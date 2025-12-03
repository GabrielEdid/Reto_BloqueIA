#!/bin/bash

################################################################################
# TrOCR Training Runner for 20GB VRAM - IMPROVED VERSION
# Sequential execution with optimized hyperparameters
# Batch sizes: 8 for all strategies
# Model: TrOCR-large (from base)
# Max length: 768 (from 256)
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "TrOCR Training Pipeline - 20GB VRAM (IMPROVED)"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Check if Python script exists
if [ ! -f "train_trocr_v2.py" ]; then
    echo "ERROR: train_trocr_v2.py not found in current directory"
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
mkdir -p ./trocr_checkpoints_v3
mkdir -p ./trocr_logs_v3

# GPU Configuration
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

################################################################################
# Strategy 1: Full Fine-tuning (RECOMMENDED FOR SMALL DATASETS)
################################################################################

echo "================================================================================"
echo "STRATEGY 1: Full Fine-tuning (Encoder + Decoder) - RECOMMENDED"
echo "================================================================================"
echo "Configuration:"
echo "  - Model: TrOCR-base"
echo "  - Epochs: 50"
echo "  - Batch size: 8"
echo "  - Gradient accumulation: 4"
echo "  - Effective batch size: 32"
echo "  - Learning rate: 1e-5 (auto)"
echo "  - Max length: 768"
echo "  - GPU ID: $GPU_ID"
echo "================================================================================"
echo ""

python train_trocr_v2.py \
    --strategy full \
    --epochs 50 \
    --batch_size 8 \
    --gpu_id $GPU_ID \
    --num_workers 6 \
    --accumulate_grad 4 \
    --max_length 768 \
    --model_size base

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Strategy 1 (Full) completed successfully"
else
    echo ""
    echo "✗ Strategy 1 (Full) failed"
    exit 1
fi

# GPU cleanup
echo ""
echo "Cleaning up GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 10

################################################################################
# Strategy 2: Partial Unfreezing (Last 6 encoder layers)
################################################################################

echo ""
echo "================================================================================"
echo "STRATEGY 2: Partial Unfreezing (Last 6 encoder layers + Decoder)"
echo "================================================================================"
echo "Configuration:"
echo "  - Model: TrOCR-base"
echo "  - Epochs: 40"
echo "  - Batch size: 8"
echo "  - Gradient accumulation: 4"
echo "  - Effective batch size: 32"
echo "  - Learning rate: 3e-5 (auto)"
echo "  - Max length: 768"
echo "  - GPU ID: $GPU_ID"
echo "================================================================================"
echo ""

python train_trocr_v2.py \
    --strategy partial \
    --epochs 40 \
    --batch_size 8 \
    --gpu_id $GPU_ID \
    --num_workers 6 \
    --accumulate_grad 4 \
    --max_length 768 \
    --model_size base

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Strategy 2 (Partial) completed successfully"
else
    echo ""
    echo "✗ Strategy 2 (Partial) failed"
    exit 1
fi

# GPU cleanup
echo ""
echo "Cleaning up GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 10

################################################################################
# Strategy 3: Frozen Encoder (For comparison)
################################################################################

echo ""
echo "================================================================================"
echo "STRATEGY 3: Frozen Encoder (Decoder-only training)"
echo "================================================================================"
echo "Configuration:"
echo "  - Model: TrOCR-base"
echo "  - Epochs: 60"
echo "  - Batch size: 8"
echo "  - Gradient accumulation: 4"
echo "  - Effective batch size: 32"
echo "  - Learning rate: 5e-5 (auto)"
echo "  - Max length: 768"
echo "  - GPU ID: $GPU_ID"
echo "================================================================================"
echo ""

python train_trocr_v2.py \
    --strategy frozen \
    --epochs 60 \
    --batch_size 8 \
    --gpu_id $GPU_ID \
    --num_workers 6 \
    --accumulate_grad 4 \
    --max_length 768 \
    --model_size base

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Strategy 3 (Frozen) completed successfully"
else
    echo ""
    echo "✗ Strategy 3 (Frozen) failed"
    exit 1
fi

################################################################################
# Training Complete
################################################################################

echo ""
echo "================================================================================"
echo "ALL STRATEGIES COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Checkpoints: ./trocr_checkpoints_v3/"
echo "  - Logs: ./trocr_logs_v3/"
echo "  - Detailed logs: ./logs/trocr_*.log"
echo ""
echo "Best models (top 3 per strategy):"
echo "  Strategy 1 (full):    ./trocr_checkpoints_v3/strategy_full_base/"
echo "  Strategy 2 (partial): ./trocr_checkpoints_v3/strategy_partial_base/"
echo "  Strategy 3 (frozen):  ./trocr_checkpoints_v3/strategy_frozen_base/"
echo ""
echo "To view training progress:"
echo "  cat ./logs/trocr_full_*.log"
echo "  cat ./logs/trocr_partial_*.log"
echo "  cat ./logs/trocr_frozen_*.log"
echo ""
echo "To view metrics:"
echo "  cat ./trocr_logs_v3/strategy_full_base/version_*/metrics.csv"
echo "  cat ./trocr_logs_v3/strategy_partial_base/version_*/metrics.csv"
echo "  cat ./trocr_logs_v3/strategy_frozen_base/version_*/metrics.csv"
echo ""
echo "RECOMMENDATION: Start with Strategy 1 (full fine-tuning) for best results"
echo "================================================================================"
