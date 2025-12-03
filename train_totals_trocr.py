import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import lightning as L
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torchvision.transforms as T


# ------------------------------------------------------------
# Dataset SOLO TOTAL con data augmentation opcional
# ------------------------------------------------------------
class CORDTotalsDataset(Dataset):
    def __init__(self, root, split, processor, transform=None):
        self.root = Path(root)
        self.processor = processor
        self.transform = transform

        ann_path = self.root / "json" / f"{split}.json"
        with open(ann_path, "r") as f:
            self.annotations = json.load(f)

        self.items = []
        for ann in self.annotations:
            total = ann["gt_parse"].get("total", {})
            t = total.get("total_price", "")
            if not t:
                continue
            num = self._clean_number(t)
            if num is None:
                continue

            self.items.append(
                {
                    "image_path": self.root / "image" / ann["file_name"],
                    "total_str": str(num),
                }
            )

    def _clean_number(self, s):
        s = s.replace("Rp", "").replace(" ", "").replace(",", "").replace(".", "")
        if not s.isdigit():
            return None
        return int(s)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        # Data augmentation (solo si hay transform definida)
        if self.transform is not None:
            image = self.transform(image)

        # El processor espera PIL o tensor; si la transform da tensor, lo pasamos directo
        if isinstance(image, torch.Tensor):
            enc = self.processor(images=image, text=item["total_str"], return_tensors="pt")
        else:
            enc = self.processor(images=image, text=item["total_str"], return_tensors="pt")

        return {
            "pixel_values": enc.pixel_values.squeeze(0),
            "labels": enc.labels.squeeze(0),
            "total_str": item["total_str"],
        }


# ------------------------------------------------------------
# DataModule
# ------------------------------------------------------------
class TotalsDataModule(L.LightningDataModule):
    def __init__(
        self,
        root,
        processor,
        batch_size=2,
        num_workers=4,
        train_transform=None,
        val_transform=None,
    ):
        super().__init__()
        self.root = root
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage=None):
        self.train_ds = CORDTotalsDataset(
            self.root,
            "train",
            self.processor,
            transform=self.train_transform,
        )
        self.val_ds = CORDTotalsDataset(
            self.root,
            "validation",
            self.processor,
            transform=self.val_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )


# ------------------------------------------------------------
# Modelo
# ------------------------------------------------------------
class TotalsTrOCRModel(LightningModule):
    def __init__(self, lr=3e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed"
        )
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-printed"
        )

        # Congelamos el encoder; solo entrenamos decoder
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

    # --------------------------------------------------------
    # Data augmentation para tickets (entrenamiento)
    # --------------------------------------------------------
    train_transform = T.Compose(
        [
            T.Resize((384, 384)),  # tamaño razonable para recibos
            T.RandomApply(
                [T.ColorJitter(brightness=0.3, contrast=0.3)],
                p=0.5,
            ),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3)],
                p=0.3,
            ),
            T.RandomRotation(
                degrees=3,  # pequeñas inclinaciones tipo foto de celular
                expand=False,
                fill=(255, 255, 255),
            ),
        ]
    )

    # En validación no conviene alterar imágenes
    val_transform = T.Resize((384, 384))

    dm = TotalsDataModule(
        root=args.root,
        processor=processor,
        batch_size=args.batch,
        num_workers=args.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    model = TotalsTrOCRModel(lr=args.lr)

    ckpt = ModelCheckpoint(
        dirpath="trocr_checkpoints/totals/",
        filename="totals-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=16,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt],
    )

    trainer.fit(model, dm)
