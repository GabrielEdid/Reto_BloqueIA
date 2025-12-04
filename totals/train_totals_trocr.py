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


def extract_total_info(ground_truth_str: str) -> Optional[Dict[str, Any]]:
    """
    Extrae el campo 'total_price' del dataset CORD junto con su bounding box.
    Retorna un dict con 'text' y 'bbox' (x, y, w, h) o None si no hay información válida.
    
    El bbox permite recortar la región específica del total en la imagen.
    """
    try:
        gt = json.loads(ground_truth_str)
    except Exception:
        return None

    valid_lines = gt.get("valid_line", [])
    
    # Buscar el campo "total.total_price" en valid_line
    for line in valid_lines:
        words = line.get("words", [])
        for word in words:
            text = word.get("text", "").strip()
            quad = word.get("quad", {})
            
            # Verificar si este word es el total_price
            # En CORD, el quad tiene formato {"x1": ..., "y1": ..., "x2": ..., "y2": ..., ...}
            if text and quad:
                # Intentar parsear como precio (contiene dígitos)
                clean_text = text.replace("Rp", "").replace(".", "").replace(",", "").replace(" ", "").strip()
                if clean_text.isdigit() and len(clean_text) >= 4:  # Al menos 4 dígitos para ser un total
                    # Extraer bounding box del quad
                    try:
                        x_coords = [quad.get(f"x{i}", 0) for i in range(1, 5)]
                        y_coords = [quad.get(f"y{i}", 0) for i in range(1, 5)]
                        
                        x_min = min(x_coords)
                        y_min = min(y_coords)
                        x_max = max(x_coords)
                        y_max = max(y_coords)
                        
                        # Expandir bbox para dar contexto
                        margin = 10
                        
                        return {
                            "text": clean_text,
                            "bbox": {
                                "x_min": max(0, x_min - margin),
                                "y_min": max(0, y_min - margin),
                                "x_max": x_max + margin,
                                "y_max": y_max + margin,
                            }
                        }
                    except Exception:
                        continue
    
    # Fallback: buscar en gt_parse.total.total_price sin bbox
    parse = gt.get("gt_parse", {})
    total = parse.get("total")
    
    if isinstance(total, dict):
        total_price = str(total.get("total_price", "")).strip()
    elif isinstance(total, str):
        total_price = total.strip()
    else:
        return None
        
    if total_price:
        clean = total_price.replace("Rp", "").replace(".", "").replace(",", "").replace(" ", "").strip()
        if clean.isdigit():
            return {"text": clean, "bbox": None}  # Sin bbox, usar imagen completa
    
    return None


# ---------------------------------------------------------------------
# Dataset a partir de HuggingFace
# ---------------------------------------------------------------------


class CORDTotalsHFDataset(Dataset):
    def __init__(
        self,
        hf_split,
        processor: TrOCRProcessor,
        max_length: int = 32,
        use_augmentation: bool = False,
        split_name: str = "train",
    ):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        self.split_name = split_name

        # Augmentación más agresiva para training
        if use_augmentation:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
                ], p=0.3),
            ])
        else:
            self.transform = None

        self.samples: List[Dict[str, Any]] = []
        skipped_no_bbox = 0
        skipped_no_text = 0
        
        for sample in hf_split:
            img = sample["image"]
            gt_str = sample["ground_truth"]
            info = extract_total_info(gt_str)
            
            if info is None:
                skipped_no_text += 1
                continue
                
            text = info["text"]
            bbox = info["bbox"]
            
            if bbox is None:
                skipped_no_bbox += 1
                # Aún podemos usar la imagen completa, pero no es ideal
            
            self.samples.append({
                "image": img,
                "text": text,
                "bbox": bbox
            })

        print(
            f"[CORDTotalsHFDataset] {split_name}: {len(self.samples)} válidos de {len(hf_split)} "
            f"(sin bbox: {skipped_no_bbox}, sin texto: {skipped_no_text})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image: Image.Image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        # Crop a la región del total si hay bbox
        bbox = sample.get("bbox")
        if bbox is not None:
            try:
                x_min = int(bbox["x_min"])
                y_min = int(bbox["y_min"])
                x_max = int(bbox["x_max"])
                y_max = int(bbox["y_max"])
                
                # Asegurar que el bbox es válido
                w, h = image.size
                x_min = max(0, min(x_min, w))
                y_min = max(0, min(y_min, h))
                x_max = max(x_min + 1, min(x_max, w))
                y_max = max(y_min + 1, min(y_max, h))
                
                image = image.crop((x_min, y_min, x_max, y_max))
            except Exception as e:
                # Si falla el crop, usar imagen completa
                print(f"Warning: Failed to crop image in {self.split_name}, using full image: {e}")

        # Aplicar augmentación después del crop
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
        max_length: int = 32,
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
        learning_rate: float = 5e-5,
        warmup_steps: int = 300,
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 2,
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
    parser.add_argument("--epochs", type=int, required=True, help="Número de épocas")
    parser.add_argument("--batch", type=int, required=True, help="Batch size (recomendado: 8-16 en GPU potente)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (por defecto: 5e-5)")
    parser.add_argument("--num_workers", type=int, default=4, help="Workers para DataLoader")

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
        max_length=32,
    )
    dm.setup()

    print("Inicializando modelo Lightning...")
    model = TrOCRTotalsModule(
        model_name="microsoft/trocr-base-printed",
        learning_rate=args.lr,
        warmup_steps=300,
        freeze_encoder=True,
        unfreeze_last_n_layers=2,  # Descongelar últimas 2 capas del encoder
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
        patience=5,  # Reducir patience para detener antes si no mejora
        verbose=True,
        min_delta=0.001,  # Mejora mínima significativa
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
