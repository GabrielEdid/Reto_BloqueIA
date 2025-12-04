#!/usr/bin/env python3
"""
TrOCR Training Script for RTX 4070 Ti (12GB VRAM)
Optimized for SSH execution with detailed logging and checkpoint management
Conservative batch size configuration: 2/2/2 for frozen/partial/full

En este ajuste, la etiqueta de entrenamiento se simplifica a campos de total:
  TOTAL <total_price> [CASH <cashprice>] [CHANGE <changeprice>] [CARD <creditcardprice>]
para que la tarea sea más sencilla y consistente con datos limitados.
"""

import os
import argparse
import logging
import json
from typing import Callable, Optional, Dict
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import torchmetrics

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from datasets import load_dataset

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_cosine_schedule_with_warmup


def setup_logging(strategy_name: str) -> logging.Logger:
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"trocr_{strategy_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)

        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe_img = clahe.apply(gray)

        if len(img_np.shape) == 3:
            clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(clahe_img)


class SharpenTransform:
    def __init__(self, amount: float = 1.0):
        self.amount = amount

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)

        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ],
            dtype=np.float32
        ) * self.amount

        sharpened = cv2.filter2D(img_np, -1, kernel)

        return Image.fromarray(sharpened)


class TrOCRDataset(Dataset):
    """
    Dataset para fine-tuning de TrOCR en CORD-v2.

    Ahora la etiqueta solo incluye los totales:

    TOTAL <total_price> [CASH <cashprice>] [CHANGE <changeprice>] [CARD <creditcardprice>]

    Ejemplos posibles:
      "TOTAL 45500 CASH 50000 CHANGE 4500"
      "TOTAL 123000 CARD 123000"
      "TOTAL 8000 CASH 8000"
    """

    def __init__(
        self,
        hf_dataset,
        processor: TrOCRProcessor,
        image_transform: Optional[Callable] = None,
        max_length: int = 64,
        logger: Optional[logging.Logger] = None
    ):
        self.processor = processor
        self.image_transform = image_transform
        self.max_length = max_length
        self.logger = logger

        self.samples = []
        for sample in hf_dataset:
            ground_truth_str = sample["ground_truth"]
            text = self.extract_text_from_ground_truth(ground_truth_str)
            if text.strip():
                self.samples.append(
                    {
                        "image": sample["image"],
                        "ground_truth": ground_truth_str,
                        "text": text
                    }
                )

        if self.logger:
            self.logger.info(
                f"TrOCRDataset initialized: {len(self.samples)} samples with non-empty text "
                f"out of {len(hf_dataset)} original examples"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def extract_text_from_ground_truth(self, ground_truth_str: str) -> str:
        """
        Construye una cadena de texto solamente con campos de total.

        Estructura esperada:
        {
          "gt_parse": {
            ...
            "total": {
              "total_price": "...",
              "cashprice": "...",
              "changeprice": "...",
              "creditcardprice": "..."
            }
          }
        }

        Formato de salida:
          "TOTAL <total_price> [CASH <cashprice>] [CHANGE <changeprice>] [CARD <creditcardprice>]"
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

        parts = []
        if total_price:
            parts.append(f"TOTAL {total_price}")
        if cash:
            parts.append(f"CASH {cash}")
        if change:
            parts.append(f"CHANGE {change}")
        if card:
            parts.append(f"CARD {card}")

        return " ".join(parts).strip()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image = sample["image"]
        if self.image_transform:
            image = self.image_transform(image)

        text = sample["text"]

        enc = self.processor(
            image,
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        pixel_values = enc.pixel_values.squeeze(0)
        labels = enc.labels.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text
        }


class TrOCRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hf_dataset,
        processor: TrOCRProcessor,
        batch_size: int = 2,
        num_workers: int = 4,
        max_length: int = 64,
        use_augmentation: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        self.logger = logger

        if use_augmentation:
            self.train_transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.05, contrast=0.05)
                ]
            )
        else:
            self.train_transform = None

        self.val_transform = None

    def setup(self, stage: str = None):
        self.train_dataset = TrOCRDataset(
            hf_dataset=self.hf_dataset["train"],
            processor=self.processor,
            image_transform=self.train_transform,
            max_length=self.max_length,
            logger=self.logger
        )

        self.val_dataset = TrOCRDataset(
            hf_dataset=self.hf_dataset["validation"],
            processor=self.processor,
            image_transform=self.val_transform,
            max_length=self.max_length,
            logger=self.logger
        )

        self.test_dataset = TrOCRDataset(
            hf_dataset=self.hf_dataset["test"],
            processor=self.processor,
            image_transform=self.val_transform,
            max_length=self.max_length,
            logger=self.logger
        )

        if self.logger:
            self.logger.info(
                f"DataModule setup - "
                f"Train: {len(self.train_dataset)}, "
                f"Val: {len(self.val_dataset)}, "
                f"Test: {len(self.test_dataset)}"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        texts = [item["text"] for item in batch]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": texts
        }


class TrOCRLightningModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
        logger_obj: Optional[logging.Logger] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["logger_obj"])
        self.logger_obj = logger_obj

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

        self._apply_freezing_strategy(freeze_encoder, unfreeze_last_n_layers)

        vocab_size = len(self.processor.tokenizer)
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=vocab_size,
            ignore_index=-100
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=vocab_size,
            ignore_index=-100
        )

    def _apply_freezing_strategy(self, freeze_encoder: bool, unfreeze_last_n_layers: int):
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            if self.logger_obj:
                self.logger_obj.info("Encoder frozen (transfer learning mode)")

            if unfreeze_last_n_layers > 0:
                encoder_layers = self.model.encoder.encoder.layer
                for layer in encoder_layers[-unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                if self.logger_obj:
                    self.logger_obj.info(
                        f"Unfroze last {unfreeze_last_n_layers} encoder layers"
                    )
        else:
            if self.logger_obj:
                self.logger_obj.info("Encoder unfrozen (full fine-tuning mode)")

        for param in self.model.decoder.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if self.logger_obj:
            self.logger_obj.info(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss

        with torch.no_grad():
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            self.train_acc(preds_flat, labels_flat)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

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

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self(pixel_values, labels=labels)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        mask = labels_flat != -100
        if mask.sum() > 0:
            acc = (preds_flat[mask] == labels_flat[mask]).sum().float() / mask.sum()
        else:
            acc = torch.tensor(0.0, device=self.device)

        self.log("test_acc", acc, prog_bar=True)

        return {"test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description="Train TrOCR model on CORD-v2 dataset (RTX 4070 Ti optimized, totals only)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["frozen", "partial", "full"],
        help="Training strategy: frozen, partial, or full"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for training"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)"
    )
    parser.add_argument(
        "--accumulate_grad",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum sequence length for labels (default: 64, suitable for totals)"
    )

    args = parser.parse_args()

    logger = setup_logging(args.strategy)
    logger.info("=" * 80)
    logger.info(f"TrOCR Training on RTX 4070 Ti - Strategy: {args.strategy}")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Gradient accumulation: {args.accumulate_grad}")
    logger.info(f"  - Effective batch size: {args.batch_size * args.accumulate_grad}")
    logger.info(f"  - Learning rate: 5e-5")
    logger.info(f"  - GPU ID: {args.gpu_id}")
    logger.info(f"  - Max length: {args.max_length}")
    logger.info(f"  - Num workers: {args.num_workers}")
    logger.info("=" * 80)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    logger.info("Loading CORD-v2 dataset from Hugging Face...")
    hf_dataset = load_dataset("naver-clova-ix/cord-v2")
    logger.info(
        f"Dataset loaded: {len(hf_dataset['train'])} train, "
        f"{len(hf_dataset['validation'])} val, {len(hf_dataset['test'])} test"
    )

    logger.info("Loading TrOCR processor...")
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

    logger.info("Creating data module...")
    data_module = TrOCRDataModule(
        hf_dataset=hf_dataset,
        processor=trocr_processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        use_augmentation=True,
        logger=logger
    )
    data_module.setup()

    logger.info(f"Initializing TrOCR model with {args.strategy} strategy...")

    if args.strategy == "frozen":
        freeze_encoder = True
        unfreeze_last_n = 0
    elif args.strategy == "partial":
        freeze_encoder = True
        unfreeze_last_n = 3
    else:
        freeze_encoder = False
        unfreeze_last_n = 0

    lightning_model = TrOCRLightningModel(
        model_name="microsoft/trocr-base-printed",
        learning_rate=5e-5,
        warmup_steps=500,
        freeze_encoder=freeze_encoder,
        unfreeze_last_n_layers=unfreeze_last_n,
        logger_obj=logger
    )

    checkpoint_dir = f"./trocr_checkpoints/strategy_{args.strategy}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="trocr-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    csv_logger = CSVLogger(
        save_dir="./trocr_logs",
        name=f"strategy_{args.strategy}"
    )

    logger.info("Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=csv_logger,
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Logs: ./trocr_logs/strategy_{args.strategy}")
    logger.info("=" * 80)

    trainer.fit(lightning_model, data_module)

    logger.info("=" * 80)
    logger.info("Training completed")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    logger.info("=" * 80)

    logger.info("Running test evaluation on best model...")
    test_results = trainer.test(lightning_model, data_module, ckpt_path="best")
    logger.info(f"Test results: {test_results}")
    logger.info("All done")


if __name__ == "__main__":
    main()
