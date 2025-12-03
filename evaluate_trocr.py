#!/usr/bin/env python3
"""
TrOCR Model Evaluation Script
Visualizes predictions vs ground truth to diagnose training issues
"""

import os
import sys
import argparse
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
        from train_trocr import TrOCRLightningModel
        model = TrOCRLightningModel.load_from_checkpoint(
            checkpoint_path,
            learning_rate=3e-5
        )
        self.model = model.model.to(device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
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
    def predict(self, image: Image.Image) -> str:
        """Generate prediction for a single image"""
        # Preprocess image
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate text
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return generated_text
    
    def evaluate_samples(self, samples: List[Dict]) -> List[Dict]:
        """Evaluate multiple samples and return results"""
        results = []
        
        logger.info("Generating predictions...")
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            # Get prediction
            prediction = self.predict(sample['image'])
            ground_truth = sample['ground_truth']
            
            # Calculate character-level accuracy
            char_match = sum(1 for p, g in zip(prediction, ground_truth) if p == g)
            max_len = max(len(prediction), len(ground_truth))
            char_accuracy = (char_match / max_len * 100) if max_len > 0 else 0.0
            
            # Exact match
            exact_match = (prediction == ground_truth)
            
            results.append({
                'index': i,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'char_accuracy': char_accuracy,
                'exact_match': exact_match,
                'pred_len': len(prediction),
                'gt_len': len(ground_truth)
            })
        
        return results
    
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
            gt = sample['ground_truth']
            
            # Tokenize
            tokens = self.processor.tokenizer(
                gt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512
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
        default="frozen",
        help="Which training strategy to evaluate"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of test samples to evaluate"
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
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Find best checkpoint for strategy
        ckpt_dir = f"./trocr_checkpoints/strategy_{args.strategy}"
        if not os.path.exists(ckpt_dir):
            logger.error(f"Checkpoint directory not found: {ckpt_dir}")
            sys.exit(1)
        
        # Find checkpoint with lowest val_loss
        checkpoints = list(Path(ckpt_dir).glob("*.ckpt"))
        if not checkpoints:
            logger.error(f"No checkpoints found in {ckpt_dir}")
            sys.exit(1)
        
        # Parse val_loss from filename
        best_ckpt = None
        best_loss = float('inf')
        for ckpt in checkpoints:
            try:
                # Format: trocr-epoch=XX-val_loss=X.XXXX-val_acc=X.XXXX.ckpt
                parts = ckpt.stem.split('-')
                for part in parts:
                    if part.startswith('val_loss='):
                        loss = float(part.split('=')[1])
                        if loss < best_loss:
                            best_loss = loss
                            best_ckpt = ckpt
                        break
            except:
                continue
        
        if best_ckpt is None:
            logger.error("Could not find best checkpoint")
            sys.exit(1)
        
        checkpoint_path = str(best_ckpt)
        logger.info(f"Using best checkpoint: {best_ckpt.name} (val_loss={best_loss:.4f})")
    
    # Initialize evaluator
    evaluator = TrOCREvaluator(checkpoint_path, device=args.device)
    
    # Load test data
    samples = evaluator.load_test_data(num_samples=args.samples)
    
    # Analyze tokenization if requested
    if args.analyze_tokens:
        evaluator.analyze_tokenization(samples)
    
    # Evaluate
    results = evaluator.evaluate_samples(samples)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results to file
    output_file = f"evaluation_results_{args.strategy}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        total_samples = len(results)
        exact_matches = sum(1 for r in results if r['exact_match'])
        avg_char_acc = sum(r['char_accuracy'] for r in results) / total_samples
        
        f.write(f"Strategy: {args.strategy}\n")
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
