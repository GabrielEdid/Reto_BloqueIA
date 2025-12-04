#!/usr/bin/env python3
import os
import json
import argparse
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import torchmetrics

from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    get_cosine_schedule_with_warmup,
)

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------


def set_seed(seed: int = 42):
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")


def extract_totals_text(ground_truth_str: str) -> str:
    """
    Toma el campo 'ground_truth' del dataset CORD y construye una etiqueta
    del tipo:

      "TOTAL <total_price> [CASH <cashprice>] [CHANGE <changeprice>] [CARD <creditcardprice>]"

    Si no hay nada usable, regresa cadena vacía.
    """
    try:
        gt = json.loads(ground_truth_str)
    except Exception:
        return ""

    parse = gt.get("gt_parse", {})
    total = parse.get("total")

    total_price = ""
    cash = ""
    change = ""
    card = ""

    if isinstance(total, dict):
        total_price = str(total.get("total_price", "")).strip()
        cash = str(total.get("cashprice", "")).strip()
        change = str(total.get("changeprice", "")).strip()
        card = str(total.get("creditcardprice", "")).strip()
    elif isinstance(total, str):
        total_price = total.strip()

    parts: List[str] = []
    if total_price:
        parts.append(f"TOTAL {total_price}")
    if cash:
        parts.append(f"CASH {cash}")
    if change:
        parts.append(f"CHANGE {change}")
    if card:
        parts.append(f"CARD {card}")

    return " ".join(parts).strip()


# ---------------------------------------------------------------------
# Dataset a partir de HuggingFace
# ---------------------------------------------------------------------


class CORDTotalsHFDataset(Dataset):
    def __init__(
        self,
        hf_split,
        processor: TrOCRProcessor,
        max_length: int = 64,
        use_augmentation: bool = False,
        split_name: str = "train",
    ):
        super().__init__()
        self.processor = processor
        self.max_length = max_length

        if use_augmentation:
            self.transform = transforms.ColorJitter(brightness=0.05, contrast=0.05)
        else:
            self.transform = None

        self.samples: List[Dict[str, Any]] = []
        for sample in hf_split:
            img = sample["image"]
            gt_str = sample["ground_truth"]
            text = extract_totals_text(gt_str)
            if text.strip():
                self.samples.append({"image": img, "text": text})

        print(
            f"[CORDTotalsHFDataset] ejemplos válidos: {len(self.samples)} de {len(hf_split)} ({split_name})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image: Image.Image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        if self.transform is not None:
            image = self.transform(image)

        text: str = sample["text"]

        enc = self.processor(
            image,
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        pixel_values = enc.pixel_values.squeeze(0)
        labels = enc.labels.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text,
        }


# ---------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------


class CORDTotalsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hf_dataset,
        processor: TrOCRProcessor,
        batch_size: int = 2,
        num_workers: int = 4,
        max_length: int = 64,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = CORDTotalsHFDataset(
            hf_split=self.hf_dataset["train"],
            processor=self.processor,
            max_length=self.max_length,
            use_augmentation=True,
            split_name="train",
        )

        self.val_dataset = CORDTotalsHFDataset(
            hf_split=self.hf_dataset["validation"],
            processor=self.processor,
            max_length=self.max_length,
            use_augmentation=False,
            split_name="validation",
        )

        self.test_dataset = CORDTotalsHFDataset(
            hf_split=self.hf_dataset["test"],
            processor=self.processor,
            max_length=self.max_length,
            use_augmentation=False,
            split_name="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------


class TrOCRTotalsModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        learning_rate: float = 3e-5,
        warmup_steps: int = 500,
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = TrOCRProcessor.from_pretrained(model_name)

        tokenizer = self.processor.tokenizer
        config = self.model.config

        if config.pad_token_id is None and tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id

        if config.decoder_start_token_id is None:
            if getattr(tokenizer, "bos_token_id", None) is not None:
                config.decoder_start_token_id = tokenizer.bos_token_id
            elif getattr(tokenizer, "cls_token_id", None) is not None:
                config.decoder_start_token_id = tokenizer.cls_token_id

        if config.eos_token_id is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                config.eos_token_id = tokenizer.eos_token_id
            elif getattr(tokenizer, "sep_token_id", None) is not None:
                config.eos_token_id = tokenizer.sep_token_id

        # Estrategia de congelar encoder
        if freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

            if unfreeze_last_n_layers > 0:
                encoder_layers = self.model.encoder.encoder.layer
                for layer in encoder_layers[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        # Decoder siempre entrenable
        for p in self.model.decoder.parameters():
            p.requires_grad = True

        vocab_size = len(self.processor.tokenizer)

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-100
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=-100
        )

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)

        self.train_acc(preds_flat, labels_flat)

        bs = labels.size(0)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)

        self.val_acc(preds_flat, labels_flat)

        bs = labels.size(0)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )

        return loss

    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        mask = labels_flat != -100
        if mask.sum() > 0:
            acc = (preds_flat[mask] == labels_flat[mask]).float().mean()
        else:
            acc = torch.tensor(0.0, device=self.device)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        if self.trainer is not None:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = 1000

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        epoch = int(self.current_epoch)

        train_loss = metrics.get("train_loss_epoch")
        train_acc = metrics.get("train_acc")
        val_loss = metrics.get("val_loss")
        val_acc = metrics.get("val_acc")

        def _to_float(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().item()
            return float(x)

        train_loss = _to_float(train_loss)
        train_acc = _to_float(train_acc)
        val_loss = _to_float(val_loss)
        val_acc = _to_float(val_acc)

        print(
            f"[Resumen] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning de TrOCR en CORD-v2 para campos de totales."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data_set/CORD_v2",
        help="No se usa cuando cargamos desde HuggingFace, se deja por compatibilidad.",
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    set_seed(42)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Usando dispositivo: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Cargando dataset CORD-v2 desde HuggingFace...")
    hf_dataset = load_dataset("naver-clova-ix/cord-v2")

    print("Cargando processor de TrOCR...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

    print("Creando DataModule...")
    dm = CORDTotalsDataModule(
        hf_dataset=hf_dataset,
        processor=processor,
        batch_size=args.batch,
        num_workers=args.num_workers,
        max_length=64,
    )
    dm.setup()

    print("Inicializando modelo Lightning...")
    model = TrOCRTotalsModule(
        model_name="microsoft/trocr-base-printed",
        learning_rate=args.lr,
        warmup_steps=500,
        freeze_encoder=True,
        unfreeze_last_n_layers=0,
    )

    ckpt_dir = "trocr_checkpoints/totals"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="totals-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=8,
        verbose=True,
    )

    csv_logger = CSVLogger(
        save_dir="trocr_logs",
        name="totals",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        precision="16-mixed" if device == "cuda" else "32-true",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=csv_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Si quieres reanudar desde el último checkpoint:
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None

    print(f"Checkpoint dir: {ckpt_dir}")
    if ckpt_path is not None:
        print(f"Reanudando desde {ckpt_path}")
    else:
        print("Entrenando desde cero (no se encontró last.ckpt)")

    trainer.fit(model, dm, ckpt_path=ckpt_path)

    print("Entrenamiento terminado. Evaluando en test...")
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
