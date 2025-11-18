#!/bin/bash

# Project Setup Script
# This script installs Python dependencies and creates a .env file if it doesn't exist

set -e  # Exit on error

echo "==================================="
echo "Project Setup"
echo "==================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip first."
    exit 1
fi

echo "pip version: $(pip3 --version)"
echo ""

# Create .env file first (if it doesn't exist)
if [ -f ".env" ]; then
    echo "✓ .env file already exists, skipping creation."
else
    echo "Creating .env file..."
    cat > .env << 'EOF'
# Environment variables for project configuration

# Data paths
DATA_PATH=Data_Tarea2/German_Traffic_signs
SPLIT_DATA_PATH=Data_Tarea2/German_Traffic_signs_Split

# Model configuration
BATCH_SIZE=64
LEARNING_RATE=0.001
NUM_EPOCHS=10

# Device configuration (cpu or cuda)
DEVICE=cpu

# Random seed for reproducibility
RANDOM_SEED=42
EOF
    echo "✓ .env file created successfully!"
fi

echo ""

# Install requirements
echo "Installing Python dependencies from requirements.txt..."
echo "-----------------------------------"

# Try to install with pip, handle externally-managed environment
if pip3 install -r requirements.txt 2>/dev/null; then
    echo "✓ Dependencies installed successfully!"
else
    echo ""
    echo "Note: System Python is externally managed by Homebrew."
    echo "Installing with --user flag or --break-system-packages..."
    
    # Try with --user flag first (safer option)
    if pip3 install --user -r requirements.txt 2>/dev/null; then
        echo "✓ Dependencies installed successfully (user site-packages)!"
    elif pip3 install --break-system-packages -r requirements.txt 2>/dev/null; then
        echo "✓ Dependencies installed successfully (system-wide)!"
    else
        echo ""
        echo "⚠️  Could not install packages with pip."
        echo ""
        echo "To install packages, please either:"
        echo "  1. Create and activate a virtual environment:"
        echo "     python3 -m venv venv"
        echo "     source venv/bin/activate"
        echo "     pip install -r requirements.txt"
        echo ""
        echo "  2. Or use pipx for isolated installation:"
        echo "     brew install pipx"
        echo ""
        exit 1
    fi
fi

echo ""
echo "==================================="
echo "Setup completed successfully!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Activate your virtual environment (if using one)"
echo "2. Run 'jupyter notebook' or open the notebook in VS Code"
echo "3. Start working on your project!"
echo ""
