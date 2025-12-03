#!/usr/bin/env python3
"""
TrOCR Training Script - Optimized for 20GB VRAM
Improved for receipt OCR with better hyperparameters
Key improvements:
- TrOCR-large model
- max_length=768 (from 256)
- batch_size=8 (from 2)
- Proper beam search generation
- CER/WER metrics
- Strategy-specific learning rates
"""

import os
import argparse
import logging
import json
from typing import Callable, Optional, Dict, List
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
from metrics_utils import calculate_cer, calculate_wer


# ==========================
# Logging Setup
# ==========================

def setup_logging(strategy_name):
    """Configure detailed logging to file and console"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"trocr_{strategy_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


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
        max_length: int = 768,
        logger=None
    ):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.image_transform = image_transform
        self.max_length = max_length
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"TrOCRDataset initialized with {len(self.hf_dataset)} samples")
    
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
        
        # Tokenize - match original train_trocr.py (no add_special_tokens to avoid index errors)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        # Replace padding tokens with -100 for loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text
        }


class TrOCRDataModule(pl.LightningDataModule):
    """DataModule for TrOCR training with CORD-v2 dataset."""
    
    def __init__(
        self,
        hf_dataset,
        processor,
        batch_size: int = 8,
        num_workers: int = 6,
        max_length: int = 768,
        use_augmentation: bool = True,
        logger=None
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
            # Minimal but effective augmentation for OCR
            self.train_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ])
        else:
            self.train_transform = None
        
        self.val_transform = None
    
    def setup(self, stage: str = None):
        self.train_dataset = TrOCRDataset(
            hf_dataset=self.hf_dataset['train'],
            processor=self.processor,
            image_transform=self.train_transform,
            max_length=self.max_length,
            logger=self.logger
        )
        
        self.val_dataset = TrOCRDataset(
            hf_dataset=self.hf_dataset['validation'],
            processor=self.processor,
            image_transform=self.val_transform,
            max_length=self.max_length,
            logger=self.logger
        )
        
        self.test_dataset = TrOCRDataset(
            hf_dataset=self.hf_dataset['test'],
            processor=self.processor,
            image_transform=self.val_transform,
            max_length=self.max_length,
            logger=self.logger
        )
        
        if self.logger:
            self.logger.info(f"DataModule setup - Train: {len(self.train_dataset)}, "
                           f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
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

class TrOCRLightningModel(pl.LightningModule):
    """TrOCR model with PyTorch Lightning for OCR on receipts."""
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-large-printed",
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.1,
        freeze_encoder: bool = False,
        unfreeze_last_n_layers: int = 0,
        max_length: int = 768,
        logger_obj=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['logger_obj'])
        self.logger_obj = logger_obj
        
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        
        # Configure model for better generation
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = max_length
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        
        self._apply_freezing_strategy(freeze_encoder, unfreeze_last_n_layers)
        
        # Token-level accuracy metric
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
        
        # Store for CER/WER calculation
        self.validation_step_outputs = []
        
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
                    self.logger_obj.info(f"Unfroze last {unfreeze_last_n_layers} encoder layers")
        else:
            if self.logger_obj:
                self.logger_obj.info("Encoder unfrozen (full fine-tuning mode)")
        
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.logger_obj:
            self.logger_obj.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                               f"({100 * trainable_params / total_params:.2f}%)")
    
    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        
        # Calculate token-level accuracy
        with torch.no_grad():
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            self.train_acc(preds_flat, labels_flat)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        ground_truths = batch['text']
        
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        
        # Calculate token-level accuracy
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        self.val_acc(preds_flat, labels_flat)
        
        # Generate predictions for CER/WER (sample every 5 batches to save time)
        if batch_idx % 5 == 0:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.hparams.max_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0
                )
                predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Store for epoch-level metrics
                self.validation_step_outputs.extend([
                    {'predictions': predictions, 'ground_truths': ground_truths}
                ])
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate CER/WER at end of validation epoch"""
        if len(self.validation_step_outputs) > 0:
            all_predictions = []
            all_ground_truths = []
            
            for output in self.validation_step_outputs:
                all_predictions.extend(output['predictions'])
                all_ground_truths.extend(output['ground_truths'])
            
            if len(all_predictions) > 0:
                cer = calculate_cer(all_predictions, all_ground_truths)
                wer = calculate_wer(all_predictions, all_ground_truths)
                
                self.log('val_cer', cer, prog_bar=True, on_epoch=True)
                self.log('val_wer', wer, prog_bar=True, on_epoch=True)
                
                if self.logger_obj:
                    self.logger_obj.info(f"Validation CER: {cer:.4f}, WER: {wer:.4f}")
            
            # Clear for next epoch
            self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        ground_truths = batch['text']
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=self.hparams.max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0
            )
            predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Calculate CER/WER
        cer = calculate_cer(predictions, ground_truths)
        wer = calculate_wer(predictions, ground_truths)
        
        # Calculate token-level accuracy
        outputs = self(pixel_values, labels=labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        mask = labels_flat != -100
        acc = (preds_flat[mask] == labels_flat[mask]).sum().float() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)
        
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)
        self.log('test_cer', cer, prog_bar=True, on_epoch=True)
        self.log('test_wer', wer, prog_bar=True, on_epoch=True)
        
        return {'test_acc': acc, 'test_cer': cer, 'test_wer': wer}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Calculate warmup steps as ratio of total steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)
        
        if self.logger_obj:
            self.logger_obj.info(f"Warmup steps: {warmup_steps} / {total_steps} ({self.hparams.warmup_ratio*100:.0f}%)")
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }




# ==========================
# Main Execution with CLI
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Train TrOCR model on CORD-v2 dataset (20GB VRAM optimized)")
    parser.add_argument("--strategy", type=str, required=True, 
                        choices=["frozen", "partial", "full"],
                        help="Training strategy: frozen, partial, or full")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training (default: 8)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID (default: 0)")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of dataloader workers (default: 6)")
    parser.add_argument("--accumulate_grad", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--max_length", type=int, default=768,
                        help="Maximum sequence length (default: 768)")
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["base", "large"],
                        help="TrOCR model size: base or large (default: base)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate (default: auto based on strategy)")
    
    args = parser.parse_args()
    
    # Determine learning rate based on strategy if not specified
    if args.learning_rate is None:
        if args.strategy == "frozen":
            learning_rate = 5e-5  # Higher for frozen encoder
        elif args.strategy == "partial":
            learning_rate = 3e-5  # Medium
        else:  # full
            learning_rate = 1e-5  # Lower to avoid catastrophic forgetting
    else:
        learning_rate = args.learning_rate
    
    # Setup logging
    logger = setup_logging(args.strategy)
    logger.info("=" * 80)
    logger.info(f"TrOCR Training (20GB VRAM) - Strategy: {args.strategy}")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - Model size: {args.model_size}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Gradient accumulation: {args.accumulate_grad}")
    logger.info(f"  - Effective batch size: {args.batch_size * args.accumulate_grad}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - GPU ID: {args.gpu_id}")
    logger.info(f"  - Max length: {args.max_length}")
    logger.info(f"  - Num workers: {args.num_workers}")
    logger.info("=" * 80)
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Load dataset
    logger.info("Loading CORD-v2 dataset from Hugging Face...")
    hf_dataset = load_dataset("naver-clova-ix/cord-v2")
    logger.info(f"Dataset loaded: {len(hf_dataset['train'])} train, "
               f"{len(hf_dataset['validation'])} val, {len(hf_dataset['test'])} test")
    
    # Load processor
    model_name = f"microsoft/trocr-{args.model_size}-printed"
    logger.info(f"Loading TrOCR processor from {model_name}...")
    trocr_processor = TrOCRProcessor.from_pretrained(model_name)
    
    # Create DataModule
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
    
    # Create model with appropriate strategy
    logger.info(f"Initializing TrOCR-{args.model_size} model with {args.strategy} strategy...")
    
    if args.strategy == "frozen":
        freeze_encoder = True
        unfreeze_last_n = 0
    elif args.strategy == "partial":
        freeze_encoder = True
        unfreeze_last_n = 6  # Increased for large model
    else:  # full
        freeze_encoder = False
        unfreeze_last_n = 0
    
    lightning_model = TrOCRLightningModel(
        model_name=model_name,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        freeze_encoder=freeze_encoder,
        unfreeze_last_n_layers=unfreeze_last_n,
        max_length=args.max_length,
        logger_obj=logger
    )
    
    # Setup callbacks
    checkpoint_dir = f"./trocr_checkpoints_v3/strategy_{args.strategy}_{args.model_size}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='trocr-{epoch:02d}-{val_loss:.4f}-{val_cer:.4f}',
        monitor='val_cer',  # Monitor CER instead of loss
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_cer',  # Monitor CER instead of loss
        patience=8,  # Increased patience for small dataset
        mode='min',
        verbose=True,
        min_delta=0.01  # Require at least 1% improvement
    )
    
    # Setup CSV logger
    csv_logger = CSVLogger(
        save_dir='./trocr_logs_v3',
        name=f'strategy_{args.strategy}_{args.model_size}'
    )
    
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        accumulate_grad_batches=args.accumulate_grad,
        gradient_clip_val=0.5,  # Reduced from 1.0 for better stability
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=csv_logger,
        log_every_n_steps=5,
        val_check_interval=1.0,  # Validate every epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True  # For reproducibility
    )
    
    # Log training start
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Logs: ./trocr_logs_v3/strategy_{args.strategy}_{args.model_size}")
    logger.info(f"Dataset: 800 train, 100 val, 100 test")
    logger.info("=" * 80)
    
    # Train
    trainer.fit(lightning_model, data_module)
    
    # Log completion
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Best validation CER: {checkpoint_callback.best_model_score:.4f}")
    logger.info("=" * 80)
    
    # Test evaluation
    logger.info("Running test evaluation on best model...")
    test_results = trainer.test(lightning_model, data_module, ckpt_path='best')
    logger.info(f"Test results: {test_results}")
    
    logger.info("All done! 🎉")


if __name__ == "__main__":
    main()
