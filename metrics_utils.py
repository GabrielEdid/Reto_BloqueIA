"""
OCR Evaluation Metrics - Character Error Rate (CER) and Word Error Rate (WER)
"""

from typing import List
import Levenshtein


def calculate_cer(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate Character Error Rate (CER) between predictions and ground truths.
    
    CER = (Substitutions + Deletions + Insertions) / Total Characters in Reference
    
    Args:
        predictions: List of predicted text strings
        ground_truths: List of ground truth text strings
        
    Returns:
        Float representing the average CER (lower is better, 0 is perfect)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    total_distance = 0
    total_length = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Calculate Levenshtein distance (edit distance)
        distance = Levenshtein.distance(pred, gt)
        total_distance += distance
        total_length += len(gt)
    
    if total_length == 0:
        return 0.0
    
    cer = total_distance / total_length
    return cer


def calculate_wer(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate Word Error Rate (WER) between predictions and ground truths.
    
    WER = (Substitutions + Deletions + Insertions) / Total Words in Reference
    
    Args:
        predictions: List of predicted text strings
        ground_truths: List of ground truth text strings
        
    Returns:
        Float representing the average WER (lower is better, 0 is perfect)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    total_distance = 0
    total_words = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Split into words
        pred_words = pred.split()
        gt_words = gt.split()
        
        # Calculate Levenshtein distance at word level
        distance = Levenshtein.distance(' '.join(pred_words), ' '.join(gt_words))
        total_distance += distance
        total_words += len(gt_words)
    
    if total_words == 0:
        return 0.0
    
    wer = total_distance / total_words
    return wer


def calculate_exact_match(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate the percentage of exact matches between predictions and ground truths.
    
    Args:
        predictions: List of predicted text strings
        ground_truths: List of ground truth text strings
        
    Returns:
        Float between 0 and 1 representing exact match accuracy
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    exact_matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred.strip() == gt.strip())
    return exact_matches / len(predictions)


if __name__ == "__main__":
    # Test the metrics
    predictions = [
        "hello world",
        "the quick brown fox",
        "hello wrld"
    ]
    
    ground_truths = [
        "hello world",
        "the quick brown dog",
        "hello world"
    ]
    
    print("Testing OCR Metrics:")
    print(f"CER: {calculate_cer(predictions, ground_truths):.4f}")
    print(f"WER: {calculate_wer(predictions, ground_truths):.4f}")
    print(f"Exact Match: {calculate_exact_match(predictions, ground_truths):.4f}")
