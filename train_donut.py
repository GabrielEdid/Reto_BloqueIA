#!/usr/bin/env python3
"""
Donut Training Script for A4500 (20GB VRAM)
Optimized for SSH execution with detailed logging and checkpoint management
Conservative batch size configuration: 4/2/2 for frozen/partial/full (reduced for OOM fix)
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

from transformers import DonutProcessor, VisionEncoderDecoderModel, get_cosine_schedule_with_warmup


# ==========================
# Logging Setup
# ==========================

def setup_logging(strategy_name):
    """Configure detailed logging to file and console"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"donut_{strategy_name}_{timestamp}.log"
    
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
        
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]) * self.amount
        
        if len(img_np.shape) == 3:
            sharpened = cv2.filter2D(img_np, -1, kernel)
        else:
            sharpened = cv2.filter2D(img_np, -1, kernel)
        
        return Image.fromarray(sharpened)


# ==========================
# Dataset Classes
# ==========================

class DonutDataset(Dataset):
    """Dataset for Donut fine-tuning on CORD-v2."""
    
    def __init__(
        self,
        hf_dataset,
        processor,
        image_transform: Optional[Callable] = None,
        max_length: int = 512,
        task_prompt: str = "<s_cord-v2>",
        logger=None
    ):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.image_transform = image_transform
        self.max_length = max_length
        self.task_prompt = task_prompt
        self.logger = logger
        
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [task_prompt]})
        
        if self.logger:
            self.logger.info(f"DonutDataset initialized with {len(self.hf_dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def extract_structured_info(self, ground_truth_str: str) -> str:
        """Extract structured information from CORD ground truth."""
        try:
            gt_dict = json.loads(ground_truth_str)
            
            structured_data = {}
            
            if 'valid_line' in gt_dict:
                all_text = []
                for line in gt_dict['valid_line']:
                    if 'words' in line:
                        line_text = []
                        for word in line['words']:
                            if 'text' in word:
                                line_text.append(word['text'])
                        if line_text:
                            all_text.append(' '.join(line_text))
                
                structured_data['text'] = ' '.join(all_text)
            
            return json.dumps(structured_data, ensure_ascii=False)
        except:
            return json.dumps({"text": ""})
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.hf_dataset[idx]
        
        image = sample['image']
        
        if self.image_transform:
            image = self.image_transform(image)
        
        structured_json = self.extract_structured_info(sample['ground_truth'])
        
        target_sequence = f"{self.task_prompt}{structured_json}</s>"
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "target_sequence": target_sequence,
        }


class DonutDataModule(pl.LightningDataModule):
    """DataModule for Donut training with CORD-v2 dataset."""
    
    def __init__(
        self,
        hf_dataset,
        processor,
        batch_size: int = 4,
        num_workers: int = 4,
        max_length: int = 512,
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
            # Minimal augmentation for OCR - avoid rotation/sharpening that hurt performance
            self.train_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ])
        else:
            self.train_transform = None
        
        self.val_transform = None
    
    def setup(self, stage: str = None):
        self.train_dataset = DonutDataset(
            hf_dataset=self.hf_dataset['train'],
            processor=self.processor,
            image_transform=self.train_transform,
            max_length=self.max_length,
            logger=self.logger
        )
        
        self.val_dataset = DonutDataset(
            hf_dataset=self.hf_dataset['validation'],
            processor=self.processor,
            image_transform=self.val_transform,
            max_length=self.max_length,
            logger=self.logger
        )
        
        self.test_dataset = DonutDataset(
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
        target_sequences = [item['target_sequence'] for item in batch]
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'target_sequence': target_sequences,
        }


# ==========================
# Model Class
# ==========================

class DonutLightningModel(pl.LightningModule):
    """Donut model with PyTorch Lightning for document understanding."""
    
    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
        max_length: int = 512,
        logger_obj=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['logger_obj'])
        self.logger_obj = logger_obj
        
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = DonutProcessor.from_pretrained(model_name)
        
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<s_cord-v2>"]})
        self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
        
        self._apply_freezing_strategy(freeze_encoder, unfreeze_last_n_layers)
        
        self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(["<s_cord-v2>"])[0]
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
        
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
                self.logger_obj.info("Encoder (Swin Transformer) frozen")
            
            if unfreeze_last_n_layers > 0:
                try:
                    encoder_layers = self.model.encoder.encoder.layers
                    for layer in encoder_layers[-unfreeze_last_n_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    if self.logger_obj:
                        self.logger_obj.info(f"Unfroze last {unfreeze_last_n_layers} encoder layers")
                except Exception as e:
                    if self.logger_obj:
                        self.logger_obj.warning(f"Could not unfreeze specific layers: {e}")
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
        
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        
        # Calculate token-level accuracy
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        self.val_acc(preds_flat, labels_flat)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        outputs = self(pixel_values, labels=labels)
        
        generated_ids = self.model.generate(
            pixel_values,
            max_length=self.hparams.max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        
        test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=len(self.processor.tokenizer),
            ignore_index=-100
        ).to(self.device)
        acc = test_acc(preds_flat, labels_flat)
        
        self.log('test_acc', acc, prog_bar=True)
        
        return {'predictions': generated_texts, 'test_acc': acc}
    
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
    parser = argparse.ArgumentParser(description="Train Donut model on CORD-v2 dataset (A4500 optimized)")
    parser.add_argument("--strategy", type=str, required=True, 
                        choices=["frozen", "partial", "full"],
                        help="Training strategy: frozen, partial, or full")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Batch size for training")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID (default: 0)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers (default: 4)")
    parser.add_argument("--accumulate_grad", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.strategy)
    logger.info("=" * 80)
    logger.info(f"Donut Training on A4500 - Strategy: {args.strategy}")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Gradient accumulation: {args.accumulate_grad}")
    logger.info(f"  - Effective batch size: {args.batch_size * args.accumulate_grad}")
    logger.info(f"  - Learning rate: 5e-5")
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
    logger.info("Loading Donut processor...")
    donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    
    # Create DataModule
    logger.info("Creating data module...")
    data_module = DonutDataModule(
        hf_dataset=hf_dataset,
        processor=donut_processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        use_augmentation=True,
        logger=logger
    )
    data_module.setup()
    
    # Create model with appropriate strategy
    logger.info(f"Initializing Donut model with {args.strategy} strategy...")
    
    if args.strategy == "frozen":
        freeze_encoder = True
        unfreeze_last_n = 0
    elif args.strategy == "partial":
        freeze_encoder = True
        unfreeze_last_n = 2
    else:  # full
        freeze_encoder = False
        unfreeze_last_n = 0
    
    lightning_model = DonutLightningModel(
        model_name="naver-clova-ix/donut-base",
        learning_rate=5e-5,
        warmup_steps=500,
        freeze_encoder=freeze_encoder,
        unfreeze_last_n_layers=unfreeze_last_n,
        max_length=args.max_length,
        logger_obj=logger
    )
    
    # Setup callbacks
    checkpoint_dir = f"./donut_checkpoints/strategy_{args.strategy}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='donut-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Setup CSV logger
    csv_logger = CSVLogger(
        save_dir='./donut_logs',
        name=f'strategy_{args.strategy}'
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        accumulate_grad_batches=args.accumulate_grad,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=csv_logger,
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Log training start
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Logs: ./donut_logs/strategy_{args.strategy}")
    logger.info("=" * 80)
    
    # Train
    trainer.fit(lightning_model, data_module)
    
    # Log completion
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    logger.info("=" * 80)
    
    # Test evaluation
    logger.info("Running test evaluation on best model...")
    test_results = trainer.test(lightning_model, data_module, ckpt_path='best')
    logger.info(f"Test results: {test_results}")
    
    logger.info("All done! 🎉")


if __name__ == "__main__":
    main()

