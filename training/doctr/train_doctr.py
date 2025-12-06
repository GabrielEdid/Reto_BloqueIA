"""
docTR (Document Text Recognition) Evaluation Script for CORD-v2 Receipt Dataset
Follows the same structure as EasyOCR and Tesseract evaluation scripts
"""

import os
import json
import argparse
import logging
import csv
from datetime import datetime
from typing import Dict, List

import difflib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import load_dataset
from PIL import Image
import numpy as np
import cv2

# docTR imports
from doctr.models import ocr_predictor


# ==========================
# Logging Setup
# ==========================

def setup_logging(strategy: str) -> logging.Logger:
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs_doctr_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"doctr_{strategy}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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

class DocTRPreprocessor:
    """Image preprocessing for docTR OCR."""

    @staticmethod
    def _ensure_rgb(img_np: np.ndarray) -> np.ndarray:
        """Ensure image is uint8 RGB HxWx3."""
        if img_np.ndim == 2:  # grayscale HxW
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        # If already HxWx3, leave as is
        return img_np.astype(np.uint8)

    @staticmethod
    def preprocess_image(image, preprocess_type: str = "raw") -> np.ndarray:
        """
        Preprocess image for docTR recognition.

        Args:
            image: PIL Image or path string
            preprocess_type: "raw", "default", "simple_threshold", "grayscale", "enhanced"

        Returns:
            Preprocessed image as numpy array H x W x 3, uint8 RGB
        """
        # Convert to numpy (handle both PIL and path strings)
        if isinstance(image, str):
            img_np = cv2.imread(image)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            # HF CORD "image" is a PIL image already
            img_np = np.array(image)

        # Treat "default" as "raw" for safety/backward compat
        if preprocess_type in ("raw", "default"):
            return DocTRPreprocessor._ensure_rgb(img_np)

        if preprocess_type == "grayscale":
            if img_np.ndim == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            return DocTRPreprocessor._ensure_rgb(gray_rgb)

        if preprocess_type == "simple_threshold":
            if img_np.ndim == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            _, threshold = cv2.threshold(
                gray, thresh=0, maxval=255,
                type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            threshold_rgb = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
            return DocTRPreprocessor._ensure_rgb(threshold_rgb)

        if preprocess_type == "enhanced":
            # Typical "receipt" pipeline: denoise + Otsu
            if img_np.ndim == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            denoised = cv2.fastNlMeansDenoising(
                gray, None, h=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
            _, threshold = cv2.threshold(
                denoised, thresh=0, maxval=255,
                type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            threshold_rgb = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
            return DocTRPreprocessor._ensure_rgb(threshold_rgb)

        # Fallback: just ensure RGB
        return DocTRPreprocessor._ensure_rgb(img_np)


# ==========================
# Dataset
# ==========================

class DocTRDataset(Dataset):
    """Dataset wrapper for CORD-v2 with docTR-compatible format."""

    def __init__(
        self,
        hf_split,
        preprocess_type: str = "raw",
        logger: logging.Logger = None
    ):
        self.hf_split = hf_split
        self.preprocess_type = preprocess_type
        self.logger = logger
        self.preprocessor = DocTRPreprocessor()

    @staticmethod
    def extract_text_from_ground_truth(ground_truth) -> str:
        """Extract all text from CORD ground truth JSON."""
        try:
            if isinstance(ground_truth, str):
                gt_dict = json.loads(ground_truth)
            else:
                gt_dict = ground_truth

            text_lines: List[str] = []
            if "valid_line" in gt_dict:
                for line in gt_dict["valid_line"]:
                    for word in line.get("words", []):
                        txt = word.get("text", "")
                        if txt:
                            text_lines.append(txt)

            full_text = " ".join(text_lines)
            return full_text.strip()
        except Exception:
            return ""

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.hf_split[idx]
        image_pil: Image.Image = sample["image"]

        # Preprocess image to RGB np.ndarray
        image_np = self.preprocessor.preprocess_image(
            image_pil, self.preprocess_type
        )

        # Save a few samples for sanity-check
        if idx < 5:
            save_dir = "./preprocessed_samples"
            os.makedirs(save_dir, exist_ok=True)

            # Save original
            original_path = os.path.join(save_dir, f"sample_{idx}_original.png")
            image_pil.save(original_path)

            # Save preprocessed
            preprocessed_path = os.path.join(save_dir, f"sample_{idx}_preprocessed.png")
            Image.fromarray(image_np).save(preprocessed_path)

            if self.logger and idx == 0:
                self.logger.info(f"Saved preprocessed samples to {save_dir}/")

        text = self.extract_text_from_ground_truth(sample["ground_truth"])

        return {
            "image": image_np,      # np.ndarray HxWx3 (RGB)
            "text": text,           # GT text
            "image_pil": image_pil  # for debugging/visualization
        }


# ==========================
# Data Module
# ==========================

class DocTRDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for docTR evaluation."""

    def __init__(
        self,
        hf_dataset,
        batch_size: int = 4,
        num_workers: int = 4,
        preprocess_type: str = "raw",
        logger: logging.Logger = None
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess_type = preprocess_type
        self.logger_obj = logger

    def setup(self, stage=None):
        """Setup train/val/test datasets."""
        if stage in ("fit", None, "validate"):
            self.train_dataset = DocTRDataset(
                self.hf_dataset["train"],
                preprocess_type=self.preprocess_type,
                logger=self.logger_obj
            )
            self.val_dataset = DocTRDataset(
                self.hf_dataset["validation"],
                preprocess_type=self.preprocess_type,
                logger=self.logger_obj
            )

        if stage in ("test", None):
            self.test_dataset = DocTRDataset(
                self.hf_dataset["test"],
                preprocess_type=self.preprocess_type,
                logger=self.logger_obj
            )

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable-size images."""
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        images_pil = [item["image_pil"] for item in batch]

        return {
            "images": images,
            "texts": texts,
            "images_pil": images_pil
        }

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)


# ==========================
# Lightning Module
# ==========================

class DocTRLightningModel(pl.LightningModule):
    """
    docTR OCR wrapper with PyTorch Lightning for receipt OCR.
    Note: docTR can be fine-tuned, but this script focuses on evaluation.
    """

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        pretrained: bool = True,
        use_gpu: bool = True,
        learning_rate: float = 2e-5,  # dummy for Lightning
        logger_obj: logging.Logger = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["logger_obj"])
        self.logger_obj = logger_obj

        if self.logger_obj:
            self.logger_obj.info(
                f"Initializing docTR with det_arch={det_arch}, reco_arch={reco_arch}"
            )

        # Initialize docTR OCR predictor (torch backend)
        self.model = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=pretrained
        )

        # Move to GPU if available
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            if self.logger_obj:
                self.logger_obj.info("docTR predictor moved to GPU")

        # Store predictions / GT for epoch-level metrics
        self.val_predictions: List[str] = []
        self.val_ground_truths: List[str] = []
        self.test_predictions: List[str] = []
        self.test_ground_truths: List[str] = []

    def forward(self, image_np: np.ndarray) -> str:
        """Run docTR OCR inference on a single numpy RGB image."""
        try:
            # Ensure uint8 HxWx3
            if isinstance(image_np, Image.Image):
                image_np = np.array(image_np)
            if image_np.ndim == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            image_np = image_np.astype(np.uint8)

            # Directly pass numpy page as in official docs
            # out = model([input_page])
            result = self.model([image_np])

            # Iterate structured result
            full_text_tokens: List[str] = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            full_text_tokens.append(word.value)

            text = " ".join(full_text_tokens).strip()
            return text

        except Exception as e:
            if self.logger_obj:
                self.logger_obj.warning(f"docTR inference failed: {e}")
            return ""

    @staticmethod
    def calculate_character_accuracy(pred: str, gt: str) -> float:
        """Character-level similarity using difflib (robust to small shifts)."""
        if not gt:
            return 0.0
        matcher = difflib.SequenceMatcher(None, gt, pred)
        return matcher.ratio()

    def training_step(self, batch, batch_idx):
        """
        docTR can be fine-tuned, but this script focuses on evaluation.
        Return a dummy loss for Lightning compatibility.
        """
        loss = torch.tensor(0.0, requires_grad=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - run docTR inference and compare to ground truth."""
        images = batch["images"]
        texts = batch["texts"]

        batch_accs: List[float] = []

        for image_np, gt_text in zip(images, texts):
            pred_text = self.forward(image_np)

            acc = self.calculate_character_accuracy(pred_text, gt_text)
            batch_accs.append(acc)

            self.val_predictions.append(pred_text)
            self.val_ground_truths.append(gt_text)

        avg_acc = float(sum(batch_accs) / len(batch_accs)) if batch_accs else 0.0
        self.log(
            "val_acc",
            avg_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(images),
        )
        return {"val_acc": avg_acc}

    def test_step(self, batch, batch_idx):
        """Test step - run docTR inference and save all predictions."""
        images = batch["images"]
        texts = batch["texts"]

        batch_accs: List[float] = []

        for image_np, gt_text in zip(images, texts):
            pred_text = self.forward(image_np)

            acc = self.calculate_character_accuracy(pred_text, gt_text)
            batch_accs.append(acc)

            self.test_predictions.append(pred_text)
            self.test_ground_truths.append(gt_text)

        avg_acc = float(sum(batch_accs) / len(batch_accs)) if batch_accs else 0.0
        self.log(
            "test_acc",
            avg_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(images),
        )
        return {"test_acc": avg_acc}

    def on_test_epoch_end(self):
        """Save all test predictions and metrics to CSV."""
        if not self.test_predictions:
            if self.logger_obj:
                self.logger_obj.warning("No test predictions collected.")
            return

        accs: List[float] = []
        for pred, gt in zip(self.test_predictions, self.test_ground_truths):
            accs.append(self.calculate_character_accuracy(pred, gt))

        avg_acc = float(sum(accs) / len(accs))

        # Save detailed predictions to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"./results_doctr_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        predictions_file = os.path.join(results_dir, "test_predictions.csv")
        with open(predictions_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "ground_truth", "prediction", "accuracy"])
            for i, (gt, pred, acc) in enumerate(zip(self.test_ground_truths, self.test_predictions, accs)):
                writer.writerow([i, gt, pred, f"{acc:.4f}"])

        # Save summary metrics
        metrics_file = os.path.join(results_dir, "metrics.csv")
        with open(metrics_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["avg_accuracy", f"{avg_acc:.4f}"])
            writer.writerow(["min_accuracy", f"{min(accs):.4f}"])
            writer.writerow(["max_accuracy", f"{max(accs):.4f}"])
            writer.writerow(["total_samples", len(accs)])

        if self.logger_obj:
            self.logger_obj.info("=" * 80)
            self.logger_obj.info("TEST SET RESULTS")
            self.logger_obj.info("=" * 80)
            self.logger_obj.info(f"Total samples: {len(accs)}")
            self.logger_obj.info(f"Average accuracy: {avg_acc:.4f}")
            self.logger_obj.info(f"Min accuracy: {min(accs):.4f}")
            self.logger_obj.info(f"Max accuracy: {max(accs):.4f}")
            self.logger_obj.info(f"Results saved to: {results_dir}/")
            self.logger_obj.info(f"  - Predictions: {predictions_file}")
            self.logger_obj.info(f"  - Metrics: {metrics_file}")
            self.logger_obj.info("=" * 80)
            self.logger_obj.info("Sample Predictions:")
            for i in range(min(5, len(self.test_predictions))):
                gt = self.test_ground_truths[i]
                pr = self.test_predictions[i]
                self.logger_obj.info(f"\n[Sample {i}]")
                self.logger_obj.info(f"  GT  : {gt[:150]}")
                self.logger_obj.info(f"  Pred: {pr[:150]}")
                self.logger_obj.info(f"  Acc : {accs[i]:.4f}")

        # Clear buffers
        self.test_predictions.clear()
        self.test_ground_truths.clear()

    def on_validation_epoch_end(self):
        """Log summary statistics at end of validation epoch."""
        if not self.val_predictions:
            if self.logger_obj:
                self.logger_obj.warning(
                    "No validation predictions were collected in this epoch."
                )
            return

        accs: List[float] = []
        for pred, gt in zip(self.val_predictions, self.val_ground_truths):
            accs.append(self.calculate_character_accuracy(pred, gt))

        avg_acc = float(sum(accs) / len(accs))

        if self.logger_obj:
            self.logger_obj.info(
                f"Validation Epoch Complete - Avg char similarity (difflib): {avg_acc:.4f}"
            )
            self.logger_obj.info("Sample Predictions (truncated to 120 chars):")
            for i in range(min(3, len(self.val_predictions))):
                gt = self.val_ground_truths[i]
                pr = self.val_predictions[i]
                self.logger_obj.info(f"[Sample {i}] GT len={len(gt)}, Pred len={len(pr)}")
                self.logger_obj.info(f"  GT  : {gt[:120]}")
                self.logger_obj.info(f"  Pred: {pr[:120]}")
                self.logger_obj.info(f"  Acc : {accs[i]:.4f}")

        # Clear buffers
        self.val_predictions.clear()
        self.val_ground_truths.clear()

    def configure_optimizers(self):
        """
        Dummy optimizer so Lightning is happy even though we don't train.
        """
        dummy_param = nn.Parameter(torch.zeros(1))
        return torch.optim.Adam([dummy_param], lr=self.hparams.learning_rate)


# ==========================
# Main Execution
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="docTR OCR Evaluation on CORD-v2 dataset (following EasyOCR/Tesseract structure)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of validation epochs to run (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        choices=["default", "simple_threshold", "grayscale", "enhanced", "raw"],
        default="raw",  # raw/default -> just pass images to docTR
        help="Image preprocessing type (default: raw)",
    )
    parser.add_argument(
        "--det_arch",
        type=str,
        default="db_resnet50",
        help="Detection architecture (default: db_resnet50)",
    )
    parser.add_argument(
        "--reco_arch",
        type=str,
        default="crnn_vgg16_bn",
        help="Recognition architecture (default: crnn_vgg16_bn)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("evaluation")
    logger.info("=" * 80)
    logger.info("docTR OCR Evaluation on CORD-v2 Receipt Dataset")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - GPU ID: {args.gpu_id}")
    logger.info(f"  - Preprocessing: {args.preprocess}")
    logger.info(f"  - Detection arch: {args.det_arch}")
    logger.info(f"  - Recognition arch: {args.reco_arch}")
    logger.info(f"  - Num workers: {args.num_workers}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("NOTE: docTR supports fine-tuning, but this script focuses on evaluation.")
    logger.info("This evaluates pre-trained docTR performance on CORD-v2 receipts.")
    logger.info("=" * 80)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    # Load dataset
    logger.info("Loading CORD-v2 dataset from Hugging Face...")
    hf_dataset = load_dataset("naver-clova-ix/cord-v2")
    logger.info(
        f"Dataset loaded: {len(hf_dataset['train'])} train, "
        f"{len(hf_dataset['validation'])} val, {len(hf_dataset['test'])} test"
    )

    # Create DataModule
    logger.info("Creating data module...")
    data_module = DocTRDataModule(
        hf_dataset=hf_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess_type=args.preprocess,
        logger=logger,
    )
    data_module.setup()

    # Create model
    logger.info("Initializing docTR OCR model...")
    lightning_model = DocTRLightningModel(
        det_arch=args.det_arch,
        reco_arch=args.reco_arch,
        pretrained=True,
        use_gpu=torch.cuda.is_available(),
        logger_obj=logger,
    )

    # TensorBoard logger with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logger = TensorBoardLogger(
        save_dir=f"./doctr_logs_{timestamp}",
        name="evaluation",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=tb_logger,
        log_every_n_steps=10,
        enable_checkpointing=False,
        deterministic=False,
    )

    # Run test (evaluates on full test set)
    logger.info("=" * 80)
    logger.info("Starting docTR OCR Test Evaluation on Full Test Set (100 samples)...")
    logger.info("=" * 80)

    trainer.test(lightning_model, datamodule=data_module)

    logger.info("=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    logger.info(f"Logs saved to: ./doctr_logs_{timestamp}/")
    logger.info(f"Check results_doctr_{timestamp}/ for predictions and metrics CSV files.")


if __name__ == "__main__":
    main()
