#!/bin/bash

################################################################################
# Donut Training Runner for A4500 (20GB VRAM)
# Sequential execution of all 3 strategies with GPU cleanup between runs
# Conservative batch sizes: 4/4/3 for frozen/partial/full
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "Donut Training Pipeline - A4500 (20GB VRAM)"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Check if Python script exists
if [ ! -f "train_donut.py" ]; then
    echo "ERROR: train_donut.py not found in current directory"
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
mkdir -p ./donut_checkpoints
mkdir -p ./donut_logs

# GPU Configuration
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

################################################################################
# Strategy 1: Frozen Encoder
################################################################################

echo "================================================================================"
echo "STRATEGY 1: Frozen Encoder (Decoder-only training)"
echo "================================================================================"
echo "Configuration:"
echo "  - Epochs: 30"
echo "  - Batch size: 4"
echo "  - Gradient accumulation: 8"
echo "  - Effective batch size: 32"
echo "  - Learning rate: 5e-5"
echo "  - GPU ID: $GPU_ID"
echo "================================================================================"
echo ""

python train_donut.py \
    --strategy frozen \
    --epochs 30 \
    --batch_size 4 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --accumulate_grad 8 \
    --max_length 512

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Strategy 1 completed successfully"
else
    echo ""
    echo "✗ Strategy 1 failed"
    exit 1
fi

# GPU cleanup
echo ""
echo "Cleaning up GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 10

################################################################################
# Strategy 2: Partial Unfreezing (Last 2 encoder layers)
################################################################################

echo ""
echo "================================================================================"
echo "STRATEGY 2: Partial Unfreezing (Last 2 encoder layers + Decoder)"
echo "================================================================================"
echo "Configuration:"
echo "  - Epochs: 25"
echo "  - Batch size: 4"
echo "  - Gradient accumulation: 8"
echo "  - Effective batch size: 32"
echo "  - Learning rate: 5e-5"
echo "  - GPU ID: $GPU_ID"
echo "================================================================================"
echo ""

python train_donut.py \
    --strategy partial \
    --epochs 25 \
    --batch_size 4 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --accumulate_grad 8 \
    --max_length 512

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Strategy 2 completed successfully"
else
    echo ""
    echo "✗ Strategy 2 failed"
    exit 1
fi

# GPU cleanup
echo ""
echo "Cleaning up GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 10

################################################################################
# Strategy 3: Full Fine-tuning
################################################################################

echo ""
echo "================================================================================"
echo "STRATEGY 3: Full Fine-tuning (Encoder + Decoder)"
echo "================================================================================"
echo "Configuration:"
echo "  - Epochs: 20"
echo "  - Batch size: 3"
echo "  - Gradient accumulation: 8"
echo "  - Effective batch size: 24"
echo "  - Learning rate: 5e-5"
echo "  - GPU ID: $GPU_ID"
echo "================================================================================"
echo ""

python train_donut.py \
    --strategy full \
    --epochs 20 \
    --batch_size 3 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --accumulate_grad 8 \
    --max_length 512

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Strategy 3 completed successfully"
else
    echo ""
    echo "✗ Strategy 3 failed"
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
echo "  - Checkpoints: ./donut_checkpoints/"
echo "  - Logs: ./donut_logs/"
echo "  - Detailed logs: ./logs/donut_*.log"
echo ""
echo "Best models (top 3 per strategy):"
echo "  Strategy 1 (frozen):  ./donut_checkpoints/strategy_frozen/"
echo "  Strategy 2 (partial): ./donut_checkpoints/strategy_partial/"
echo "  Strategy 3 (full):    ./donut_checkpoints/strategy_full/"
echo ""
echo "To view training progress:"
echo "  cat ./logs/donut_frozen_*.log"
echo "  cat ./logs/donut_partial_*.log"
echo "  cat ./logs/donut_full_*.log"
echo ""
echo "To view metrics:"
echo "  cat ./donut_logs/strategy_frozen/version_*/metrics.csv"
echo "  cat ./donut_logs/strategy_partial/version_*/metrics.csv"
echo "  cat ./donut_logs/strategy_full/version_*/metrics.csv"
echo ""
echo "================================================================================"
