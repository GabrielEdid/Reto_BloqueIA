import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import lightning as L
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torchvision.transforms as T
from datasets import load_dataset


# ------------------------------------------------------------
# Dataset sobre Hugging Face CORD-v2 (totals-only)
# ------------------------------------------------------------
class CORDTotalsHFDataset(Dataset):
    def __init__(self, hf_split, processor, transform=None):
        """
        hf_split: dataset de Hugging Face (por ejemplo cord['train'])
        """
        self.ds = hf_split
        self.processor = processor
        self.transform = transform

        self.indices = []
        self.totals = []

        for i, ex in enumerate(self.ds):
            ground_truth_str = ex.get("ground_truth", "")
            if not ground_truth_str:
                continue

            try:
                gt = json.loads(ground_truth_str)
            except Exception:
                continue

            gt_parse = gt.get("gt_parse", {})
            total = gt_parse.get("total", {})

            if not isinstance(total, dict):
                continue

            raw_total = str(total.get("total_price", "")).strip()
            if not raw_total:
                continue

            num = self._clean_number(raw_total)
            if num is None:
                continue

            self.indices.append(i)
            self.totals.append(str(num))

        print(
            f"[CORDTotalsHFDataset] ejemplos válidos: "
            f"{len(self.indices)} de {len(self.ds)}"
        )

    def _clean_number(self, s: str):
        """
        Limpia el total para quedarnos con un entero:
        - quita 'Rp', espacios, comas y puntos
        - si no son solo dígitos al final, descarta
        """
        s = s.replace("Rp", "").replace(" ", "").replace(",", "").replace(".", "")
        if not s.isdigit():
            return None
        return int(s)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.ds[real_idx]

        image = ex["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        total_str = self.totals[idx]

        enc = self.processor(images=image, text=total_str, return_tensors="pt")

        pixel_values = enc.pixel_values.squeeze(0)
        labels = enc.labels.squeeze(0)

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "total_str": total_str,
        }


# ------------------------------------------------------------
# DataModule usando Hugging Face
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
        self.root = root  # se mantiene para no romper la CLI
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage=None):
        cord = load_dataset("naver-clova-ix/cord-v2")

        train_split = cord["train"]
        val_split = cord["validation"]

        self.train_ds = CORDTotalsHFDataset(
            train_split, self.processor, transform=self.train_transform
        )
        self.val_ds = CORDTotalsHFDataset(
            val_split, self.processor, transform=self.val_transform
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
# Modelo Lightning
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

        tokenizer = self.processor.tokenizer
        config = self.model.config

        # Aseguramos pad_token_id
        if config.pad_token_id is None and tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id

        # Aseguramos decoder_start_token_id
        if config.decoder_start_token_id is None:
            if getattr(tokenizer, "bos_token_id", None) is not None:
                config.decoder_start_token_id = tokenizer.bos_token_id
            elif getattr(tokenizer, "cls_token_id", None) is not None:
                config.decoder_start_token_id = tokenizer.cls_token_id

        # Aseguramos eos_token_id
        if config.eos_token_id is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                config.eos_token_id = tokenizer.eos_token_id
            elif getattr(tokenizer, "sep_token_id", None) is not None:
                config.eos_token_id = tokenizer.sep_token_id

        # Congelamos el encoder; solo entrenamos el decoder
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        torch.set_float32_matmul_precision("high")

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        out = self(pixel_values=pixel_values, labels=labels)
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        out = self(pixel_values=pixel_values, labels=labels)
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

    train_transform = T.Compose(
        [
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                    )
                ],
                p=0.5,
            ),
            T.RandomApply(
                [
                    T.GaussianBlur(kernel_size=3),
                ],
                p=0.3,
            ),
            T.RandomRotation(
                degrees=3,
                expand=False,
                fill=(255, 255, 255),
            ),
        ]
    )

    val_transform = None

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
