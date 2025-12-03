#!/bin/bash

################################################################################
# EasyOCR Evaluation Runner
# Evaluates pre-trained EasyOCR on CORD-v2 receipts
# No training - just benchmarking out-of-box performance
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "EasyOCR Evaluation on CORD-v2 Receipt Dataset"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Check if Python script exists
if [ ! -f "train_easyocr.py" ]; then
    echo "ERROR: train_easyocr.py not found in current directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtual environment detected"
    echo "Attempting to activate ./env/bin/activate..."
    if [ -f "./env/bin/activate" ]; then
        source ./env/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "ERROR: Virtual environment not found. Please activate it manually."
        exit 1
    fi
else
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
fi

# Create necessary directories
mkdir -p ./logs
mkdir -p ./easyocr_results
mkdir -p ./easyocr_logs

# GPU Configuration
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected - will use CPU"
echo ""

################################################################################
# EasyOCR Evaluation
################################################################################

echo "================================================================================"
echo "Running EasyOCR Evaluation"
echo "================================================================================"
echo "Configuration:"
echo "  - Epochs: 1 (just evaluation, no training)"
echo "  - Batch size: 4"
echo "  - Languages: en (English)"
echo "  - GPU ID: $GPU_ID"
echo "================================================================================"
echo ""
echo "NOTE: EasyOCR is pre-trained and cannot be fine-tuned."
echo "This evaluation shows baseline performance before fine-tuning."
echo "Compare this with your TrOCR/Donut fine-tuned results!"
echo ""
echo "================================================================================"
echo ""

python train_easyocr.py \
    --epochs 1 \
    --batch_size 4 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --languages en

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ EasyOCR evaluation completed successfully"
else
    echo ""
    echo "✗ EasyOCR evaluation failed"
    exit 1
fi

################################################################################
# Evaluation Complete
################################################################################

echo ""
echo "================================================================================"
echo "EVALUATION COMPLETE!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Results: ./easyocr_results/"
echo "  - Logs: ./easyocr_logs/"
echo "  - Detailed logs: ./logs/easyocr_*.log"
echo ""
echo "To view results:"
echo "  cat ./logs/easyocr_evaluation_*.log"
echo ""
echo "To compare with fine-tuned models:"
echo "  - EasyOCR (this run): Pre-trained, no domain adaptation"
echo "  - TrOCR v2: Fine-tuned on CORD-v2 receipts"
echo "  - Donut v2: Fine-tuned on CORD-v2 receipts"
echo ""
echo "Expected performance:"
echo "  - EasyOCR: ~40-60% (general OCR, not receipt-specific)"
echo "  - TrOCR/Donut fine-tuned: ~60-90% (receipt-specific)"
echo ""
echo "================================================================================"
