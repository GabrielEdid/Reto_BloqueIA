#!/bin/bash

echo "=========================================="
echo "TrOCR Tokenizer Debugging"
echo "=========================================="
echo ""
echo "This script will check for tokenizer/model compatibility issues"
echo ""

# Run debug script
python debug_tokenizer.py --model_size base

echo ""
echo "=========================================="
echo "If you see 'INVALID TOKENS FOUND', there's a mismatch issue."
echo "Otherwise, the tokenizer is compatible with the model."
echo "=========================================="
