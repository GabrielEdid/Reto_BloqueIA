#!/bin/bash

################################################################################
# docTR OCR Evaluation Runner
# Evaluates pre-trained docTR on CORD-v2 receipts
# docTR supports fine-tuning but this script focuses on evaluation
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "docTR OCR Evaluation on CORD-v2 Receipt Dataset"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Check if Python script exists
if [ ! -f "train_doctr.py" ]; then
    echo "ERROR: train_doctr.py not found in current directory"
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
mkdir -p ./doctr_results
mkdir -p ./doctr_logs

# GPU Configuration
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected - will use CPU (slower but works)"
echo ""

################################################################################
# docTR Evaluation - Default Settings
################################################################################

echo "================================================================================"
echo "Running docTR Evaluation - Default Settings"
echo "================================================================================"
echo "Configuration:"
echo "  - Epochs: 1 (just evaluation, no training)"
echo "  - Batch size: 4"
echo "  - Preprocessing: default (no preprocessing)"
echo "  - Detection: db_resnet50"
echo "  - Recognition: crnn_vgg16_bn"
echo "  - GPU ID: $GPU_ID"
echo "================================================================================"
echo ""
echo "NOTE: docTR is a deep learning OCR that supports fine-tuning."
echo "This evaluation uses pre-trained weights to establish baseline performance."
echo "Compare with Tesseract, EasyOCR, and fine-tuned TrOCR/Donut!"
echo ""
echo "================================================================================"
echo ""

python train_doctr.py \
    --epochs 1 \
    --batch_size 4 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --preprocess default \
    --det_arch db_resnet50 \
    --reco_arch crnn_vgg16_bn

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ docTR evaluation (default) completed successfully"
else
    echo ""
    echo "✗ docTR evaluation (default) failed"
    exit 1
fi

################################################################################
# Optional: docTR with Different Architecture
################################################################################

echo ""
echo "================================================================================"
echo "Running docTR Evaluation - MobileNet (Faster)"
echo "================================================================================"
echo "Configuration:"
echo "  - Detection: db_mobilenet_v3_large (faster, lighter)"
echo "  - Recognition: crnn_mobilenet_v3_small"
echo "================================================================================"
echo ""

python train_doctr.py \
    --epochs 1 \
    --batch_size 4 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --preprocess default \
    --det_arch db_mobilenet_v3_large \
    --reco_arch crnn_mobilenet_v3_small

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ docTR evaluation (MobileNet) completed successfully"
else
    echo ""
    echo "✗ docTR evaluation (MobileNet) failed"
    # Don't exit, continue
fi

################################################################################
# Optional: docTR with Preprocessing
################################################################################

echo ""
echo "================================================================================"
echo "Running docTR Evaluation - With Thresholding"
echo "================================================================================"
echo "Configuration:"
echo "  - Preprocessing: simple_threshold (binarization)"
echo "  - Detection: db_resnet50"
echo "  - Recognition: crnn_vgg16_bn"
echo "================================================================================"
echo ""

python train_doctr.py \
    --epochs 1 \
    --batch_size 4 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --preprocess simple_threshold \
    --det_arch db_resnet50 \
    --reco_arch crnn_vgg16_bn

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ docTR evaluation (threshold) completed successfully"
else
    echo ""
    echo "✗ docTR evaluation (threshold) failed"
    # Don't exit, continue
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
echo "  - Results: ./doctr_results/"
echo "  - Logs: ./doctr_logs/"
echo "  - Detailed logs: ./logs/doctr_*.log"
echo ""
echo "To view results:"
echo "  cat ./logs/doctr_evaluation_*.log"
echo ""
echo "To compare different OCR engines:"
echo "  - docTR (deep learning): End-to-end trainable, layout-aware"
echo "  - Tesseract (classic): CPU-based, fast for clean text"
echo "  - EasyOCR (deep learning): Pre-trained, multi-language"
echo "  - TrOCR/Donut (transformers): Fine-tunable, GPU-required"
echo ""
echo "Expected performance on CORD-v2 receipts:"
echo "  - Tesseract: ~30-50% (designed for clean text)"
echo "  - EasyOCR: ~40-60% (general OCR)"
echo "  - docTR: ~50-70% (end-to-end, pre-trained)"
echo "  - TrOCR/Donut fine-tuned: ~60-90% (optimized for receipts)"
echo ""
echo "docTR configurations tested:"
echo "  1. Default: ResNet50 detection + VGG16 recognition (best accuracy)"
echo "  2. MobileNet: Lighter models (faster inference)"
echo "  3. Threshold: With image preprocessing"
echo ""
echo "docTR advantages over traditional OCR:"
echo "  + Handles complex layouts (multi-column, tables)"
echo "  + Preserves spatial structure of text"
echo "  + Can be fine-tuned on custom datasets"
echo "  + Combines detection + recognition in one model"
echo ""
echo "For better docTR results:"
echo "  1. Fine-tune on receipt-specific data"
echo "  2. Use GPU for faster inference"
echo "  3. Try different architecture combinations"
echo "  4. Adjust confidence thresholds"
echo ""
echo "================================================================================"
