#!/usr/bin/env python3
"""
TrOCR Training Script for CORD-v2 Receipt OCR
Trains TrOCR models with three fine-tuning strategies.
"""

import os
import glob
import json
from typing import Callable, Optional, Dict
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

import torchmetrics

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from datasets import load_dataset

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import evaluate


# ==========================
# Custom Transform Classes
# ==========================

class CLAHETransform:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image after CLAHE
        """
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Convert to grayscale for CLAHE
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe_img = clahe.apply(gray)
        
        # Convert back to RGB
        if len(img_np.shape) == 3:
            clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(clahe_img)


class SharpenTransform:
    """Apply sharpening filter to enhance text edges."""
    
    def __init__(self, amount=1.0):
        self.amount = amount
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image after sharpening
        """
        img_np = np.array(img)
        
        # Sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]) * self.amount
        
        # Apply kernel
        if len(img_np.shape) == 3:
            sharpened = cv2.filter2D(img_np, -1, kernel)
        else:
            sharpened = cv2.filter2D(img_np, -1, kernel)
        
        return Image.fromarray(sharpened)


# ==========================
# Dataset Classes
# ==========================

class TrOCRDataset(Dataset):
    """Dataset for TrOCR fine-tuning on CORD-v2."""
    
    def __init__(
        self,
        hf_dataset,
        processor,
        image_transform: Optional[Callable] = None,
        max_length: int = 512,
    ):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.image_transform = image_transform
        self.max_length = max_length
        
        print(f"TrOCRDataset initialized with {len(self.hf_dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def extract_text_from_ground_truth(self, ground_truth_str: str) -> str:
        """Extract all text from CORD ground truth JSON."""
        try:
            gt_dict = json.loads(ground_truth_str)
            
            text_lines = []
            if 'valid_line' in gt_dict:
                for line in gt_dict['valid_line']:
                    if 'words' in line:
                        for word in line['words']:
                            if 'text' in word:
                                text_lines.append(word['text'])
            
            full_text = ' '.join(text_lines)
            return full_text.strip()
        except:
            return ""
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.hf_dataset[idx]
        
        image = sample['image']
        
        if self.image_transform:
            image = self.image_transform(image)
        
        text = self.extract_text_from_ground_truth(sample['ground_truth'])
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text
        }


class TrOCRDataModule(L.LightningDataModule):
    """DataModule for TrOCR training with CORD-v2 dataset."""
    
    def __init__(
        self,
        hf_dataset,
        processor,
        batch_size: int = 1,
        num_workers: int = 0,
        use_augmentation: bool = True,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        
        self.train_transform = transforms.Compose([
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
            SharpenTransform(amount=1.0),
            transforms.RandomRotation(degrees=5, fill=255) if use_augmentation else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.1, contrast=0.1) if use_augmentation else transforms.Lambda(lambda x: x),
        ])
        
        self.val_transform = transforms.Compose([
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
            SharpenTransform(amount=1.0),
        ])
    
    def setup(self, stage: str = None):
        self.train_dataset = TrOCRDataset(
            hf_dataset=self.hf_dataset['train'],
            processor=self.processor,
            image_transform=self.train_transform,
        )
        
        self.val_dataset = TrOCRDataset(
            hf_dataset=self.hf_dataset['validation'],
            processor=self.processor,
            image_transform=self.val_transform,
        )
        
        self.test_dataset = TrOCRDataset(
            hf_dataset=self.hf_dataset['test'],
            processor=self.processor,
            image_transform=self.val_transform,
        )
        
        print(f"TrOCRDataModule Train: {len(self.train_dataset)}, "
              f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    @staticmethod
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        texts = [item['text'] for item in batch]
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'text': texts,
        }


# ==========================
# Model Class
# ==========================

class TrOCRLightningModel(L.LightningModule):
    """TrOCR model with PyTorch Lightning for OCR on receipts."""
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        learning_rate: float = 5e-5,
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        
        self._apply_freezing_strategy(freeze_encoder, unfreeze_last_n_layers)
        
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")
        
        self.train_acc = torchmetrics.MeanMetric()
        self.val_acc = torchmetrics.MeanMetric()
        
    def _apply_freezing_strategy(self, freeze_encoder: bool, unfreeze_last_n_layers: int):
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen (transfer learning mode)")
            
            if unfreeze_last_n_layers > 0:
                encoder_layers = self.model.encoder.encoder.layer
                for layer in encoder_layers[-unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"Unfroze last {unfreeze_last_n_layers} encoder layers")
        else:
            print("Encoder unfrozen (full fine-tuning mode)")
        
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        print("Decoder trainable")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=256)
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            labels_copy = labels.clone()
            labels_copy[labels_copy == -100] = self.processor.tokenizer.pad_token_id
            reference_texts = self.processor.batch_decode(labels_copy, skip_special_tokens=True)
            
            cer = self.cer_metric.compute(predictions=generated_texts, references=reference_texts)
            acc = max(0.0, 1.0 - cer)
            self.train_acc.update(acc)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        
        generated_ids = self.model.generate(pixel_values, max_length=384)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        labels_copy = labels.clone()
        labels_copy[labels_copy == -100] = self.processor.tokenizer.pad_token_id
        reference_texts = self.processor.batch_decode(labels_copy, skip_special_tokens=True)
        
        cer = self.cer_metric.compute(predictions=generated_texts, references=reference_texts)
        wer = self.wer_metric.compute(predictions=generated_texts, references=reference_texts)
        acc = max(0.0, 1.0 - cer)
        self.val_acc.update(acc)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_cer', cer, prog_bar=True, on_epoch=True)
        self.log('val_wer', wer, prog_bar=True, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True)
        
        return {'val_loss': loss, 'val_cer': cer, 'val_wer': wer, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        generated_ids = self.model.generate(pixel_values, max_length=384)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        labels_copy = labels.clone()
        labels_copy[labels_copy == -100] = self.processor.tokenizer.pad_token_id
        reference_texts = self.processor.batch_decode(labels_copy, skip_special_tokens=True)
        
        cer = self.cer_metric.compute(predictions=generated_texts, references=reference_texts)
        wer = self.wer_metric.compute(predictions=generated_texts, references=reference_texts)
        acc = max(0.0, 1.0 - cer)
        
        self.log('test_cer', cer, prog_bar=True)
        self.log('test_wer', wer, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return {'test_cer': cer, 'test_wer': wer, 'test_acc': acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


# ==========================
# Training Functions
# ==========================

def train_strategy(strategy_name, model, data_module, max_epochs=10):
    """Train a specific strategy."""
    print(f"\n{'='*60}")
    print(f"Training {strategy_name}")
    print(f"{'='*60}\n")
    
    # Configure callbacks
    checkpoint = ModelCheckpoint(
        dirpath=f'./trocr_checkpoints/{strategy_name}',
        filename=f'trocr-{strategy_name}-{{epoch:02d}}-{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True,
    )
    
    csv_logger = CSVLogger(save_dir='./trocr_logs', name=strategy_name)
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stop],
        logger=csv_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else 'auto',
        log_every_n_steps=10,
    )
    
    print(f"Checkpoints will be saved to: ./trocr_checkpoints/{strategy_name}")
    print(f"Logs will be saved to: ./trocr_logs/{strategy_name}")
    
    # Check for existing checkpoint
    checkpoint_path = f'./trocr_checkpoints/{strategy_name}/last.ckpt'
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        print("Starting training from scratch")
        trainer.fit(model, data_module)
    
    # Test best model
    print(f"\nEvaluating best model: {checkpoint.best_model_path}")
    trainer.test(model, data_module, ckpt_path=checkpoint.best_model_path)
    
    return checkpoint.best_model_path


# ==========================
# Main Execution
# ==========================

def main():
    print("="*60)
    print("TrOCR Training Script for CORD-v2 Receipt OCR")
    print("="*60)
    
    # Device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load dataset
    print("\nLoading CORD-v2 dataset from Hugging Face...")
    ds = load_dataset("naver-clova-ix/cord-v2")
    print(f"Dataset loaded: {len(ds['train'])} train, {len(ds['validation'])} val, {len(ds['test'])} test")
    
    # Load processor
    print("\nLoading TrOCR processor...")
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    
    # Initialize DataModule
    print("\nInitializing TrOCR DataModule...")
    trocr_dm = TrOCRDataModule(
        hf_dataset=ds,
        processor=trocr_processor,
        batch_size=1,
        num_workers=0,
        use_augmentation=True,
    )
    trocr_dm.setup()
    
    # Strategy 1: Frozen Encoder
    print("\n" + "="*60)
    print("Initializing Strategy 1: Frozen Encoder")
    print("="*60)
    trocr_model_frozen = TrOCRLightningModel(
        model_name="microsoft/trocr-base-printed",
        learning_rate=5e-5,
        freeze_encoder=True,
        unfreeze_last_n_layers=0,
    )
    best_model_s1 = train_strategy("strategy1_frozen", trocr_model_frozen, trocr_dm)
    
    # Strategy 2: Partial Unfreezing
    print("\n" + "="*60)
    print("Initializing Strategy 2: Unfreeze Last 3 Layers")
    print("="*60)
    trocr_model_partial = TrOCRLightningModel(
        model_name="microsoft/trocr-base-printed",
        learning_rate=3e-5,
        freeze_encoder=True,
        unfreeze_last_n_layers=3,
    )
    best_model_s2 = train_strategy("strategy2_partial", trocr_model_partial, trocr_dm)
    
    # Strategy 3: Full Fine-tuning
    print("\n" + "="*60)
    print("Initializing Strategy 3: Full Fine-tuning")
    print("="*60)
    trocr_model_full = TrOCRLightningModel(
        model_name="microsoft/trocr-base-printed",
        learning_rate=2e-5,
        freeze_encoder=False,
        unfreeze_last_n_layers=0,
    )
    best_model_s3 = train_strategy("strategy3_full", trocr_model_full, trocr_dm)
    
    print("\n" + "="*60)
    print("TrOCR Training Complete!")
    print("="*60)
    print(f"Best models saved:")
    print(f"  Strategy 1: {best_model_s1}")
    print(f"  Strategy 2: {best_model_s2}")
    print(f"  Strategy 3: {best_model_s3}")
    print("\nLogs and checkpoints saved to:")
    print(f"  Checkpoints: ./trocr_checkpoints/")
    print(f"  Logs: ./trocr_logs/")


if __name__ == "__main__":
    main()
