"""
EasyOCR Training Script for CORD-v2 Receipt Dataset
Follows the same structure as TrOCR and Donut training scripts
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from datasets import load_dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import easyocr


# ==========================
# Logging Setup
# ==========================

def setup_logging(strategy: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"easyocr_{strategy}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger


# ==========================
# Dataset
# ==========================

class EasyOCRDataset(Dataset):
    """Dataset wrapper for CORD-v2 with EasyOCR-compatible format."""
    
    def __init__(
        self,
        hf_dataset,
        image_transform=None,
        logger=None
    ):
        self.hf_dataset = hf_dataset
        self.image_transform = image_transform
        self.logger = logger
        
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
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.hf_dataset[idx]
        
        image = sample['image']
        
        if self.image_transform:
            image = self.image_transform(image)
        
        # Convert PIL to numpy array for EasyOCR
        image_np = np.array(image)
        
        text = self.extract_text_from_ground_truth(sample['ground_truth'])
        
        return {
            'image': image_np,
            'text': text,
            'image_pil': image  # Keep PIL for potential display
        }


# ==========================
# Data Module
# ==========================

class EasyOCRDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for EasyOCR training."""
    
    def __init__(
        self,
        hf_dataset,
        batch_size: int = 4,
        num_workers: int = 4,
        use_augmentation: bool = True,
        logger=None
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        self.logger_obj = logger
        
    def get_image_transforms(self, augment: bool = False) -> transforms.Compose:
        """Get image transformation pipeline."""
        if augment:
            return transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            return transforms.Compose([])
    
    def setup(self, stage=None):
        """Setup train/val/test datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = EasyOCRDataset(
                self.hf_dataset['train'],
                image_transform=self.get_image_transforms(augment=self.use_augmentation),
                logger=self.logger_obj
            )
            
            self.val_dataset = EasyOCRDataset(
                self.hf_dataset['validation'],
                image_transform=self.get_image_transforms(augment=False),
                logger=self.logger_obj
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = EasyOCRDataset(
                self.hf_dataset['test'],
                image_transform=self.get_image_transforms(augment=False),
                logger=self.logger_obj
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
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
        """Custom collate function to handle variable-size images."""
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        images_pil = [item['image_pil'] for item in batch]
        
        return {
            'images': images,
            'texts': texts,
            'images_pil': images_pil
        }


# ==========================
# Lightning Module
# ==========================

class EasyOCRLightningModel(pl.LightningModule):
    """
    EasyOCR wrapper with PyTorch Lightning for receipt OCR.
    Note: EasyOCR is primarily for inference, not training.
    This script demonstrates how to use it for evaluation and comparison.
    """
    
    def __init__(
        self,
        languages: List[str] = ['en'],
        gpu: bool = True,
        learning_rate: float = 2e-5,  # Placeholder - EasyOCR doesn't train
        logger_obj=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['logger_obj'])
        self.logger_obj = logger_obj
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(
            languages,
            gpu=gpu,
            verbose=False
        )
        
        # Metrics for evaluation
        self.val_predictions = []
        self.val_ground_truths = []
        
    def forward(self, image_np):
        """Run EasyOCR inference on image."""
        # EasyOCR readtext returns list of (bbox, text, confidence)
        results = self.reader.readtext(image_np, detail=0)  # detail=0 returns only text
        
        # Join all detected text
        full_text = ' '.join(results)
        return full_text
    
    def calculate_character_accuracy(self, pred: str, gt: str) -> float:
        """Calculate character-level accuracy."""
        if len(gt) == 0:
            return 0.0
        
        # Simple character matching
        matches = sum(1 for p, g in zip(pred, gt) if p == g)
        max_len = max(len(pred), len(gt))
        
        if max_len == 0:
            return 0.0
        
        return matches / max_len
    
    def training_step(self, batch, batch_idx):
        """
        EasyOCR doesn't support training - this is just for framework compatibility.
        In practice, you would skip training and only use validation/test.
        """
        # Return dummy loss - EasyOCR is pre-trained and frozen
        loss = torch.tensor(0.0, requires_grad=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - run EasyOCR inference and compare to ground truth."""
        images = batch['images']
        texts = batch['texts']
        
        batch_acc = []
        
        for image_np, gt_text in zip(images, texts):
            # Run EasyOCR inference
            pred_text = self.forward(image_np)
            
            # Calculate accuracy
            acc = self.calculate_character_accuracy(pred_text, gt_text)
            batch_acc.append(acc)
            
            # Store for epoch-end metrics
            self.val_predictions.append(pred_text)
            self.val_ground_truths.append(gt_text)
        
        # Average accuracy for this batch
        avg_acc = sum(batch_acc) / len(batch_acc) if batch_acc else 0.0
        
        self.log('val_acc', avg_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(images))
        
        return {'val_acc': avg_acc}
    
    def on_validation_epoch_end(self):
        """Log summary statistics at end of validation epoch."""
        if len(self.val_predictions) > 0:
            # Calculate overall accuracy
            accs = [
                self.calculate_character_accuracy(pred, gt)
                for pred, gt in zip(self.val_predictions, self.val_ground_truths)
            ]
            
            avg_acc = sum(accs) / len(accs)
            
            if self.logger_obj:
                self.logger_obj.info(f"Validation Epoch Complete - Avg Accuracy: {avg_acc:.4f}")
                
                # Log some examples
                self.logger_obj.info("Sample Predictions:")
                for i in range(min(3, len(self.val_predictions))):
                    self.logger_obj.info(f"  GT:   {self.val_ground_truths[i][:80]}")
                    self.logger_obj.info(f"  Pred: {self.val_predictions[i][:80]}")
                    self.logger_obj.info(f"  Acc:  {accs[i]:.4f}")
        
        # Clear for next epoch
        self.val_predictions.clear()
        self.val_ground_truths.clear()
    
    def configure_optimizers(self):
        """
        EasyOCR doesn't train, but we need this for Lightning compatibility.
        Return a dummy optimizer.
        """
        # Dummy parameter to optimize
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.Adam([dummy_param], lr=self.hparams.learning_rate)


# ==========================
# Main Execution
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="EasyOCR Evaluation on CORD-v2 dataset (following TrOCR/Donut structure)"
    )
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of validation epochs to run (default: 1)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID (default: 0)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers (default: 4)")
    parser.add_argument("--languages", type=str, default="en",
                        help="Languages for EasyOCR (comma-separated, default: en)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("evaluation")
    logger.info("=" * 80)
    logger.info("EasyOCR Evaluation on CORD-v2 Receipt Dataset")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - GPU ID: {args.gpu_id}")
    logger.info(f"  - Languages: {args.languages}")
    logger.info(f"  - Num workers: {args.num_workers}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("NOTE: EasyOCR is a pre-trained model and doesn't support fine-tuning.")
    logger.info("This script evaluates the out-of-box EasyOCR performance on receipts.")
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
    
    # Create DataModule
    logger.info("Creating data module...")
    data_module = EasyOCRDataModule(
        hf_dataset=hf_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=False,  # No augmentation for evaluation
        logger=logger
    )
    data_module.setup()
    
    # Create model
    logger.info("Initializing EasyOCR model...")
    languages = args.languages.split(',')
    
    lightning_model = EasyOCRLightningModel(
        languages=languages,
        gpu=torch.cuda.is_available(),
        logger_obj=logger
    )
    
    # Setup callbacks
    checkpoint_dir = "./easyocr_results"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup trainer (minimal setup since we're not training)
    tb_logger = TensorBoardLogger(
        save_dir="./easyocr_logs",
        name="evaluation"
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=tb_logger,
        log_every_n_steps=10,
        enable_checkpointing=False,  # No checkpointing needed
        deterministic=False
    )
    
    # Run validation
    logger.info("=" * 80)
    logger.info("Starting EasyOCR Evaluation...")
    logger.info("=" * 80)
    
    trainer.validate(lightning_model, datamodule=data_module)
    
    logger.info("=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {checkpoint_dir}")
    logger.info(f"Logs saved to: ./easyocr_logs/")
    logger.info("")
    logger.info("To compare with TrOCR/Donut:")
    logger.info("  - EasyOCR: Pre-trained, no fine-tuning")
    logger.info("  - TrOCR/Donut: Fine-tuned on CORD-v2 receipts")
    logger.info("")
    logger.info("EasyOCR advantages:")
    logger.info("  + Zero training required")
    logger.info("  + Fast inference")
    logger.info("  + Works out-of-box")
    logger.info("")
    logger.info("TrOCR/Donut advantages:")
    logger.info("  + Domain-specific fine-tuning")
    logger.info("  + Better accuracy on receipts")
    logger.info("  + Customizable for your data")


if __name__ == "__main__":
    main()
