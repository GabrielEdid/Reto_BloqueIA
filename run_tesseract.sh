#!/bin/bash

################################################################################
# Tesseract OCR Evaluation Runner
# Evaluates pre-trained Tesseract on CORD-v2 receipts
# No training - just benchmarking out-of-box performance with different PSM modes
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "Tesseract OCR Evaluation on CORD-v2 Receipt Dataset"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Check if Python script exists
if [ ! -f "train_tesseract.py" ]; then
    echo "ERROR: train_tesseract.py not found in current directory"
    exit 1
fi

# Check if Tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "ERROR: Tesseract OCR is not installed"
    echo ""
    echo "Please install Tesseract:"
    echo "  - macOS:   brew install tesseract"
    echo "  - Ubuntu:  sudo apt-get install tesseract-ocr"
    echo "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
    exit 1
fi

# Show Tesseract version
echo "Tesseract version:"
tesseract --version | head -1
echo ""

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
mkdir -p ./tesseract_results
mkdir -p ./tesseract_logs

# GPU Configuration (not used by Tesseract, but keeps consistency)
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "System Information:"
echo "  - OS: $(uname -s)"
echo "  - CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | grep 'Model name' | cut -d':' -f2 | xargs || echo 'Unknown')"
echo "  - Tesseract runs on CPU (does not use GPU)"
echo ""

################################################################################
# Tesseract Evaluation - Default Settings
################################################################################

echo "================================================================================"
echo "Running Tesseract Evaluation - Default Settings (PSM 6)"
echo "================================================================================"
echo "Configuration:"
echo "  - Epochs: 1 (just evaluation, no training)"
echo "  - Batch size: 4"
echo "  - Preprocessing: default (grayscale)"
echo "  - PSM Mode: 6 (uniform block of text)"
echo "  - GPU ID: $GPU_ID (not used)"
echo "================================================================================"
echo ""
echo "NOTE: Tesseract is a pre-trained OCR engine and cannot be fine-tuned."
echo "This evaluation shows baseline performance using classic OCR approach."
echo "Compare this with EasyOCR and fine-tuned TrOCR/Donut results!"
echo ""
echo "================================================================================"
echo ""

python train_tesseract.py \
    --epochs 1 \
    --batch_size 4 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --preprocess default \
    --psm 6

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Tesseract evaluation (default) completed successfully"
else
    echo ""
    echo "✗ Tesseract evaluation (default) failed"
    exit 1
fi

################################################################################
# Optional: Tesseract with Thresholding
################################################################################

echo ""
echo "================================================================================"
echo "Running Tesseract Evaluation - With Thresholding (PSM 6)"
echo "================================================================================"
echo "Configuration:"
echo "  - Preprocessing: threshold (Otsu's binarization)"
echo "  - PSM Mode: 6"
echo "================================================================================"
echo ""

python train_tesseract.py \
    --epochs 1 \
    --batch_size 4 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --preprocess threshold \
    --psm 6

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Tesseract evaluation (threshold) completed successfully"
else
    echo ""
    echo "✗ Tesseract evaluation (threshold) failed"
    # Don't exit, continue to next
fi

################################################################################
# Optional: Tesseract with Different PSM Mode
################################################################################

echo ""
echo "================================================================================"
echo "Running Tesseract Evaluation - PSM 4 (Single Column)"
echo "================================================================================"
echo "Configuration:"
echo "  - Preprocessing: default"
echo "  - PSM Mode: 4 (single column of text - good for receipts)"
echo "================================================================================"
echo ""

python train_tesseract.py \
    --epochs 1 \
    --batch_size 4 \
    --gpu_id $GPU_ID \
    --num_workers 4 \
    --preprocess default \
    --psm 4

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Tesseract evaluation (PSM 4) completed successfully"
else
    echo ""
    echo "✗ Tesseract evaluation (PSM 4) failed"
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
echo "  - Results: ./tesseract_results/"
echo "  - Logs: ./tesseract_logs/"
echo "  - Detailed logs: ./logs/tesseract_*.log"
echo ""
echo "To view results:"
echo "  cat ./logs/tesseract_evaluation_*.log"
echo ""
echo "To compare different OCR engines:"
echo "  - Tesseract (classic): Pre-trained, CPU-based, fast"
echo "  - EasyOCR (deep learning): Pre-trained, GPU-optional"
echo "  - TrOCR/Donut (transformers): Fine-tunable, GPU-required"
echo ""
echo "Expected performance on CORD-v2 receipts:"
echo "  - Tesseract: ~30-50% (designed for clean printed text)"
echo "  - EasyOCR: ~40-60% (better on varied text)"
echo "  - TrOCR/Donut fine-tuned: ~60-90% (optimized for receipts)"
echo ""
echo "Tesseract PSM modes tested:"
echo "  - PSM 6 (default): Treats image as uniform block of text"
echo "  - PSM 4: Assumes single column (better for receipts)"
echo ""
echo "Preprocessing techniques tested:"
echo "  - Default: Grayscale conversion only"
echo "  - Threshold: Binary image with Otsu's method"
echo ""
echo "For better Tesseract results on receipts, try:"
echo "  1. Image preprocessing (denoising, contrast enhancement)"
echo "  2. Different PSM modes (--psm 4, 11, 12)"
echo "  3. Custom Tesseract training (advanced, requires labeled data)"
echo ""
echo "================================================================================"
