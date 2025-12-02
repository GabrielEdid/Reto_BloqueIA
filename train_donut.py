#!/usr/bin/env python3
"""
Donut Training Script for CORD-v2 Receipt Understanding
Trains Donut models with three fine-tuning strategies.
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

from transformers import DonutProcessor, VisionEncoderDecoderModel


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
        max_length: int = 768,
        task_prompt: str = "<s_cord-v2>",
    ):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.image_transform = image_transform
        self.max_length = max_length
        self.task_prompt = task_prompt
        
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [task_prompt]})
        
        print(f"DonutDataset initialized with {len(self.hf_dataset)} samples")
    
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


class DonutDataModule(L.LightningDataModule):
    """DataModule for Donut training with CORD-v2 dataset."""
    
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
            transforms.RandomRotation(degrees=3, fill=255) if use_augmentation else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.1, contrast=0.1) if use_augmentation else transforms.Lambda(lambda x: x),
        ])
        
        self.val_transform = transforms.Compose([
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
            SharpenTransform(amount=1.0),
        ])
    
    def setup(self, stage: str = None):
        self.train_dataset = DonutDataset(
            hf_dataset=self.hf_dataset['train'],
            processor=self.processor,
            image_transform=self.train_transform,
        )
        
        self.val_dataset = DonutDataset(
            hf_dataset=self.hf_dataset['validation'],
            processor=self.processor,
            image_transform=self.val_transform,
        )
        
        self.test_dataset = DonutDataset(
            hf_dataset=self.hf_dataset['test'],
            processor=self.processor,
            image_transform=self.val_transform,
        )
        
        print(f"[DonutDataModule] Train: {len(self.train_dataset)}, "
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
        target_sequences = [item['target_sequence'] for item in batch]
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'target_sequence': target_sequences,
        }


# ==========================
# Model Class
# ==========================

class DonutLightningModel(L.LightningModule):
    """Donut model with PyTorch Lightning for document understanding."""
    
    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        learning_rate: float = 3e-5,
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
        max_length: int = 768,
    ):
        super().__init__()
        self.save_hyperparameters()
        
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
            print("Encoder (Swin Transformer) frozen")
            
            if unfreeze_last_n_layers > 0:
                try:
                    encoder_layers = self.model.encoder.encoder.layers
                    for layer in encoder_layers[-unfreeze_last_n_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    print(f"Unfroze last {unfreeze_last_n_layers} encoder layers")
                except:
                    print("Could not unfreeze specific layers (model structure may vary)")
        else:
            print("Encoder unfrozen (full fine-tuning mode)")
        
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        print("Decoder (mBART) trainable")
        
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
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            
            acc = self.train_acc(preds_flat, labels_flat)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        
        generated_ids = self.model.generate(
            pixel_values,
            max_length=self.hparams.max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        labels_copy = labels.clone()
        labels_copy[labels_copy == -100] = self.processor.tokenizer.pad_token_id
        reference_texts = self.processor.batch_decode(labels_copy, skip_special_tokens=True)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        acc = self.val_acc(preds_flat, labels_flat)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
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
    
    checkpoint = ModelCheckpoint(
        dirpath=f'./donut_checkpoints/{strategy_name}',
        filename=f'donut-{strategy_name}-{{epoch:02d}}-{{val_loss:.4f}}',
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
    
    csv_logger = CSVLogger(save_dir='./donut_logs', name=strategy_name)
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stop],
        logger=csv_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else 'auto',
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    
    print(f"Checkpoints will be saved to: ./donut_checkpoints/{strategy_name}")
    print(f"Logs will be saved to: ./donut_logs/{strategy_name}")
    
    checkpoint_path = f'./donut_checkpoints/{strategy_name}/last.ckpt'
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        print("Starting training from scratch")
        trainer.fit(model, data_module)
    
    print(f"\nEvaluating best model: {checkpoint.best_model_path}")
    trainer.test(model, data_module, ckpt_path=checkpoint.best_model_path)
    
    return checkpoint.best_model_path


# ==========================
# Main Execution
# ==========================

def main():
    print("="*60)
    print("Donut Training Script for CORD-v2 Receipt Understanding")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print("\nLoading CORD-v2 dataset from Hugging Face...")
    ds = load_dataset("naver-clova-ix/cord-v2")
    print(f"Dataset loaded: {len(ds['train'])} train, {len(ds['validation'])} val, {len(ds['test'])} test")
    
    print("\nLoading Donut processor...")
    donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    
    print("\nInitializing Donut DataModule...")
    donut_dm = DonutDataModule(
        hf_dataset=ds,
        processor=donut_processor,
        batch_size=1,
        num_workers=0,
        use_augmentation=True,
    )
    donut_dm.setup()
    
    # Strategy 1: Frozen Encoder
    print("\n" + "="*60)
    print("Initializing Strategy 1: Frozen Encoder")
    print("="*60)
    donut_model_frozen = DonutLightningModel(
        model_name="naver-clova-ix/donut-base",
        learning_rate=3e-5,
        freeze_encoder=True,
        max_length=384,
        unfreeze_last_n_layers=0,
    )
    best_model_s1 = train_strategy("strategy1_frozen", donut_model_frozen, donut_dm)
    
    # Strategy 2: Partial Unfreezing
    print("\n" + "="*60)
    print("Initializing Strategy 2: Unfreeze Last 2 Layers")
    print("="*60)
    donut_model_partial = DonutLightningModel(
        model_name="naver-clova-ix/donut-base",
        learning_rate=2e-5,
        freeze_encoder=True,
        max_length=384,
        unfreeze_last_n_layers=2,
    )
    best_model_s2 = train_strategy("strategy2_partial", donut_model_partial, donut_dm)
    
    # Strategy 3: Full Fine-tuning
    print("\n" + "="*60)
    print("Initializing Strategy 3: Full Fine-tuning")
    print("="*60)
    donut_model_full = DonutLightningModel(
        model_name="naver-clova-ix/donut-base",
        learning_rate=1e-5,
        freeze_encoder=False,
        max_length=384,
        unfreeze_last_n_layers=0,
    )
    best_model_s3 = train_strategy("strategy3_full", donut_model_full, donut_dm)
    
    print("\n" + "="*60)
    print("Donut Training Complete!")
    print("="*60)
    print(f"Best models saved:")
    print(f"  Strategy 1: {best_model_s1}")
    print(f"  Strategy 2: {best_model_s2}")
    print(f"  Strategy 3: {best_model_s3}")
    print("\nLogs and checkpoints saved to:")
    print(f"  Checkpoints: ./donut_checkpoints/")
    print(f"  Logs: ./donut_logs/")


if __name__ == "__main__":
    main()
