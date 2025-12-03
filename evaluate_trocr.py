#!/usr/bin/env python3
"""
TrOCR Model Evaluation Script (v2)
Visualizes predictions vs ground truth with improved generation parameters
Uses: max_length=768, beam_search=4, no_repeat_ngram_size=3
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrOCREvaluator:
    """Evaluates TrOCR model and shows predictions vs ground truth"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading model from: {checkpoint_path}")
        
        # Load processor
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        
        # Load model from checkpoint
        from train_trocr_v2 import TrOCRLightningModel
        model = TrOCRLightningModel.load_from_checkpoint(
            checkpoint_path,
            learning_rate=3e-5
        )
        self.model = model.model.to(device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def extract_text_from_ground_truth(self, ground_truth_str: str) -> str:
        """Extract all text from CORD ground truth JSON (same as training)."""
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
    
    def load_test_data(self, num_samples: int = 20) -> List[Dict]:
        """Load test samples from CORD-v2 dataset"""
        logger.info("Loading CORD-v2 test dataset...")
        dataset = load_dataset("naver-clova-ix/cord-v2", split="test")
        
        # Take first num_samples
        samples = []
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            samples.append({
                'image': sample['image'],
                'ground_truth': sample['ground_truth']
            })
        
        logger.info(f"Loaded {len(samples)} test samples")
        return samples
    
    @torch.no_grad()
    def predict(self, image: Image.Image, max_length: int = 768) -> str:
        """Generate prediction for a single image"""
        # Preprocess image
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate text with proper parameters
        generated_ids = self.model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0
        )
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return generated_text
    
    def print_results(self, results: List[Dict]):
        """Print evaluation results in a readable format"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Overall statistics
        total_samples = len(results)
        exact_matches = sum(1 for r in results if r['exact_match'])
        avg_char_acc = sum(r['char_accuracy'] for r in results) / total_samples
        
        print(f"\nOverall Statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Exact matches: {exact_matches}/{total_samples} ({exact_matches/total_samples*100:.1f}%)")
        print(f"  Avg char accuracy: {avg_char_acc:.2f}%")
        
        # Individual samples
        print("\n" + "-"*80)
        print("Individual Sample Analysis:")
        print("-"*80)
        
        for result in results:
            print(f"\nSample {result['index'] + 1}:")
            print(f"  Ground Truth: '{result['ground_truth']}'")
            print(f"  Prediction:   '{result['prediction']}'")
            print(f"  Lengths: GT={result['gt_len']}, Pred={result['pred_len']}")
            print(f"  Char Accuracy: {result['char_accuracy']:.1f}%")
            print(f"  Exact Match: {'✓' if result['exact_match'] else '✗'}")
            
            # Show first difference
            if not result['exact_match']:
                gt = result['ground_truth']
                pred = result['prediction']
                for i, (g, p) in enumerate(zip(gt, pred)):
                    if g != p:
                        print(f"  First diff at position {i}: GT='{g}' vs Pred='{p}'")
                        break
                else:
                    if len(gt) != len(pred):
                        print(f"  Length mismatch: GT has {len(gt)} chars, Pred has {len(pred)} chars")
        
        print("\n" + "="*80)
    
    def analyze_tokenization(self, samples: List[Dict], num_show: int = 5):
        """Analyze how tokenizer processes ground truth texts"""
        print("\n" + "="*80)
        print("TOKENIZATION ANALYSIS")
        print("="*80)
        
        for i, sample in enumerate(samples[:num_show]):
            gt = self.extract_text_from_ground_truth(sample['ground_truth'])
            
            # Tokenize
            tokens = self.processor.tokenizer(
                gt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=768
            )
            
            # Decode back
            decoded = self.processor.tokenizer.decode(
                tokens['input_ids'][0],
                skip_special_tokens=True
            )
            
            print(f"\nSample {i + 1}:")
            print(f"  Original:  '{gt}'")
            print(f"  Token IDs: {tokens['input_ids'][0].tolist()[:20]}... ({len(tokens['input_ids'][0])} tokens)")
            print(f"  Decoded:   '{decoded}'")
            print(f"  Match: {'✓' if gt == decoded else '✗ MISMATCH'}")
            
            if gt != decoded:
                print(f"  WARNING: Tokenization is lossy!")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrOCR model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (if not provided, will use best checkpoint)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["frozen", "partial", "full"],
        default="full",
        help="Which training strategy to evaluate (default: full)"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["base", "large"],
        default="large",
        help="Model size to evaluate (default: large)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of test samples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=768,
        help="Maximum generation length (default: 768)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--analyze-tokens",
        action="store_true",
        help="Also analyze tokenization of ground truth"
    )
    parser.add_argument(
        "--checkpoint_version",
        type=str,
        default="v3",
        help="Checkpoint version (v2 or v3, default: v3)"
    )
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Find best checkpoint for strategy
        ckpt_dir = f"./trocr_checkpoints_{args.checkpoint_version}/strategy_{args.strategy}_{args.model_size}"
        if not os.path.exists(ckpt_dir):
            logger.error(f"Checkpoint directory not found: {ckpt_dir}")
            logger.info(f"Available checkpoint directories:")
            base_dir = f"./trocr_checkpoints_{args.checkpoint_version}"
            if os.path.exists(base_dir):
                for d in os.listdir(base_dir):
                    logger.info(f"  - {d}")
            sys.exit(1)
        
        # Find checkpoint with lowest val_cer (or val_loss for older checkpoints)
        checkpoints = list(Path(ckpt_dir).glob("*.ckpt"))
        if not checkpoints:
            logger.error(f"No checkpoints found in {ckpt_dir}")
            sys.exit(1)
        
        # Parse val_cer from filename (prioritize CER over loss)
        best_ckpt = None
        best_metric = float('inf')
        metric_name = 'val_cer'
        
        for ckpt in checkpoints:
            try:
                # Format: trocr-epoch=XX-val_loss=X.XXXX-val_cer=X.XXXX.ckpt
                parts = ckpt.stem.split('-')
                for part in parts:
                    if part.startswith('val_cer='):
                        metric = float(part.split('=')[1])
                        if metric < best_metric:
                            best_metric = metric
                            best_ckpt = ckpt
                        break
                else:
                    # Fallback to val_loss if val_cer not found
                    for part in parts:
                        if part.startswith('val_loss='):
                            metric = float(part.split('=')[1])
                            if metric < best_metric:
                                best_metric = metric
                                best_ckpt = ckpt
                                metric_name = 'val_loss'
                            break
            except:
                continue
        
        if best_ckpt is None:
            logger.error("Could not find best checkpoint")
            sys.exit(1)
        
        checkpoint_path = str(best_ckpt)
        logger.info(f"Using best checkpoint: {best_ckpt.name} ({metric_name}={best_metric:.4f})")
    
    # Initialize evaluator
    evaluator = TrOCREvaluator(checkpoint_path, device=args.device)
    
    # Load test data
    samples = evaluator.load_test_data(num_samples=args.samples)
    
    # Analyze tokenization if requested
    if args.analyze_tokens:
        evaluator.analyze_tokenization(samples)
    
    # Evaluate with proper max_length
    logger.info(f"Generating predictions with max_length={args.max_length}, beam_search=4...")
    results = []
    for i, sample in enumerate(samples):
        pred = evaluator.predict(sample['image'], max_length=args.max_length)
        gt = evaluator.extract_text_from_ground_truth(sample['ground_truth'])
        
        char_match = sum(1 for p, g in zip(pred, gt) if p == g)
        max_len = max(len(pred), len(gt))
        char_accuracy = (char_match / max_len * 100) if max_len > 0 else 0.0
        exact_match = (pred == gt)
        
        results.append({
            'index': i,
            'prediction': pred,
            'ground_truth': gt,
            'char_accuracy': char_accuracy,
            'exact_match': exact_match,
            'pred_len': len(pred),
            'gt_len': len(gt)
        })
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(samples)} samples")
    
    # Print results
    evaluator.print_results(results)
    
    # Save results to file
    output_file = f"evaluation_results_{args.strategy}_{args.model_size}_{args.checkpoint_version}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        total_samples = len(results)
        exact_matches = sum(1 for r in results if r['exact_match'])
        avg_char_acc = sum(r['char_accuracy'] for r in results) / total_samples
        
        f.write(f"Strategy: {args.strategy}\n")
        f.write(f"Model size: {args.model_size}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Exact matches: {exact_matches}/{total_samples} ({exact_matches/total_samples*100:.1f}%)\n")
        f.write(f"Avg char accuracy: {avg_char_acc:.2f}%\n\n")
        
        f.write("-"*80 + "\n")
        for result in results:
            f.write(f"\nSample {result['index'] + 1}:\n")
            f.write(f"  GT:   {result['ground_truth']}\n")
            f.write(f"  Pred: {result['prediction']}\n")
            f.write(f"  Char Acc: {result['char_accuracy']:.1f}%\n")
    
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
