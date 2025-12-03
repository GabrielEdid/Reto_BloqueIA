"""
Tesseract OCR Evaluation Script for CORD-v2 Receipt Dataset
Follows the same structure as EasyOCR evaluation script
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
import pytesseract
from pytesseract import Output
import cv2


# ==========================
# Logging Setup
# ==========================

def setup_logging(strategy: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tesseract_{strategy}_{timestamp}.log")
    
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
# Image Preprocessing
# ==========================

class TesseractPreprocessor:
    """Image preprocessing optimized for Tesseract OCR."""
    
    @staticmethod
    def preprocess_image(image: Image.Image, preprocess_type: str = "default") -> np.ndarray:
        """
        Preprocess image for better Tesseract recognition.
        
        Args:
            image: PIL Image
            preprocess_type: Type of preprocessing ("default", "grayscale", "threshold", "denoise")
        
        Returns:
            Preprocessed image as numpy array
        """
        # Convert to numpy
        img_np = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        if preprocess_type == "default":
            return gray
        
        elif preprocess_type == "threshold":
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        
        elif preprocess_type == "denoise":
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            return denoised
        
        elif preprocess_type == "adaptive":
            # Adaptive thresholding
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            return adaptive
        
        elif preprocess_type == "morphology":
            # Morphological operations
            kernel = np.ones((1, 1), np.uint8)
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            return morph
        
        else:
            return gray


# ==========================
# Dataset
# ==========================

class TesseractDataset(Dataset):
    """Dataset wrapper for CORD-v2 with Tesseract-compatible format."""
    
    def __init__(
        self,
        hf_dataset,
        preprocess_type: str = "default",
        logger=None
    ):
        self.hf_dataset = hf_dataset
        self.preprocess_type = preprocess_type
        self.logger = logger
        self.preprocessor = TesseractPreprocessor()
        
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
        
        # Preprocess image for Tesseract
        image_np = self.preprocessor.preprocess_image(image, self.preprocess_type)
        
        text = self.extract_text_from_ground_truth(sample['ground_truth'])
        
        return {
            'image': image_np,
            'text': text,
            'image_pil': image  # Keep PIL for potential display
        }


# ==========================
# Data Module
# ==========================

class TesseractDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Tesseract evaluation."""
    
    def __init__(
        self,
        hf_dataset,
        batch_size: int = 4,
        num_workers: int = 4,
        preprocess_type: str = "default",
        logger=None
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess_type = preprocess_type
        self.logger_obj = logger
        
    def setup(self, stage=None):
        """Setup train/val/test datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = TesseractDataset(
                self.hf_dataset['train'],
                preprocess_type=self.preprocess_type,
                logger=self.logger_obj
            )
            
            self.val_dataset = TesseractDataset(
                self.hf_dataset['validation'],
                preprocess_type=self.preprocess_type,
                logger=self.logger_obj
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = TesseractDataset(
                self.hf_dataset['test'],
                preprocess_type=self.preprocess_type,
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

class TesseractLightningModel(pl.LightningModule):
    """
    Tesseract OCR wrapper with PyTorch Lightning for receipt OCR.
    Note: Tesseract is primarily for inference, not training.
    This script demonstrates how to use it for evaluation and comparison.
    """
    
    def __init__(
        self,
        tesseract_config: str = '--psm 6',
        learning_rate: float = 2e-5,  # Placeholder - Tesseract doesn't train
        logger_obj=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['logger_obj'])
        self.logger_obj = logger_obj
        self.tesseract_config = tesseract_config
        
        # Verify Tesseract is installed
        try:
            version = pytesseract.get_tesseract_version()
            if self.logger_obj:
                self.logger_obj.info(f"Tesseract version: {version}")
        except Exception as e:
            if self.logger_obj:
                self.logger_obj.error(f"Tesseract not found: {e}")
            raise RuntimeError("Tesseract OCR is not installed. Please install it first.")
        
        # Metrics for evaluation
        self.val_predictions = []
        self.val_ground_truths = []
        
    def forward(self, image_np):
        """Run Tesseract OCR inference on image."""
        try:
            # Run Tesseract OCR
            text = pytesseract.image_to_string(
                image_np,
                config=self.tesseract_config
            )
            
            # Clean up the text
            text = text.strip()
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            if self.logger_obj:
                self.logger_obj.warning(f"Tesseract inference failed: {e}")
            return ""
    
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
        Tesseract doesn't support training - this is just for framework compatibility.
        In practice, you would skip training and only use validation/test.
        """
        # Return dummy loss - Tesseract is pre-trained and frozen
        loss = torch.tensor(0.0, requires_grad=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - run Tesseract inference and compare to ground truth."""
        images = batch['images']
        texts = batch['texts']
        
        batch_acc = []
        
        for image_np, gt_text in zip(images, texts):
            # Run Tesseract inference
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
        Tesseract doesn't train, but we need this for Lightning compatibility.
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
        description="Tesseract OCR Evaluation on CORD-v2 dataset (following EasyOCR structure)"
    )
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of validation epochs to run (default: 1)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID (default: 0)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers (default: 4)")
    parser.add_argument("--preprocess", type=str, 
                        choices=["default", "threshold", "denoise", "adaptive", "morphology"],
                        default="default",
                        help="Image preprocessing type (default: default)")
    parser.add_argument("--psm", type=int, default=6,
                        help="Tesseract Page Segmentation Mode (default: 6 - uniform block)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("evaluation")
    logger.info("=" * 80)
    logger.info("Tesseract OCR Evaluation on CORD-v2 Receipt Dataset")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - GPU ID: {args.gpu_id}")
    logger.info(f"  - Preprocessing: {args.preprocess}")
    logger.info(f"  - PSM Mode: {args.psm}")
    logger.info(f"  - Num workers: {args.num_workers}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("NOTE: Tesseract is a pre-trained OCR engine and doesn't support fine-tuning.")
    logger.info("This script evaluates the out-of-box Tesseract performance on receipts.")
    logger.info("=" * 80)
    
    # Verify Tesseract installation
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version detected: {version}")
    except Exception as e:
        logger.error(f"Tesseract not found: {e}")
        logger.error("Please install Tesseract OCR:")
        logger.error("  - macOS: brew install tesseract")
        logger.error("  - Ubuntu: sudo apt-get install tesseract-ocr")
        logger.error("  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return
    
    # Set GPU (not used by Tesseract, but keeps consistency)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"PyTorch device: {device} (Note: Tesseract uses CPU)")
    
    # Load dataset
    logger.info("Loading CORD-v2 dataset from Hugging Face...")
    hf_dataset = load_dataset("naver-clova-ix/cord-v2")
    logger.info(f"Dataset loaded: {len(hf_dataset['train'])} train, "
               f"{len(hf_dataset['validation'])} val, {len(hf_dataset['test'])} test")
    
    # Create DataModule
    logger.info("Creating data module...")
    data_module = TesseractDataModule(
        hf_dataset=hf_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess_type=args.preprocess,
        logger=logger
    )
    data_module.setup()
    
    # Create model
    logger.info("Initializing Tesseract OCR model...")
    tesseract_config = f'--psm {args.psm}'
    
    lightning_model = TesseractLightningModel(
        tesseract_config=tesseract_config,
        logger_obj=logger
    )
    
    # Setup callbacks
    checkpoint_dir = "./tesseract_results"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup trainer (minimal setup since we're not training)
    tb_logger = TensorBoardLogger(
        save_dir="./tesseract_logs",
        name="evaluation"
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='cpu',  # Tesseract doesn't use GPU
        devices=1,
        logger=tb_logger,
        log_every_n_steps=10,
        enable_checkpointing=False,  # No checkpointing needed
        deterministic=False
    )
    
    # Run validation
    logger.info("=" * 80)
    logger.info("Starting Tesseract OCR Evaluation...")
    logger.info("=" * 80)
    
    trainer.validate(lightning_model, datamodule=data_module)
    
    logger.info("=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {checkpoint_dir}")
    logger.info(f"Logs saved to: ./tesseract_logs/")
    logger.info("")
    logger.info("OCR Engine Comparison:")
    logger.info("  - Tesseract: Classic OCR engine, good for printed text")
    logger.info("  - EasyOCR: Deep learning-based, handles multiple languages")
    logger.info("  - TrOCR/Donut: Transformer-based, fine-tunable for domain")
    logger.info("")
    logger.info("Tesseract advantages:")
    logger.info("  + Mature and stable (30+ years development)")
    logger.info("  + Fast CPU-based processing")
    logger.info("  + Low memory footprint")
    logger.info("  + Many PSM modes for different layouts")
    logger.info("")
    logger.info("Tesseract limitations:")
    logger.info("  - No fine-tuning for receipts")
    logger.info("  - Less accurate on noisy/low-quality images")
    logger.info("  - Struggles with complex layouts")
    logger.info("")
    logger.info("Recommended PSM modes for receipts:")
    logger.info("  --psm 4: Single column of text (vertical receipts)")
    logger.info("  --psm 6: Uniform block of text (default)")
    logger.info("  --psm 11: Sparse text with no particular layout")


if __name__ == "__main__":
    main()
