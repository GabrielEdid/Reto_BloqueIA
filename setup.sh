#!/bin/bash

# Project Setup Script
# This script creates a Python virtual environment and installs dependencies

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

# Create virtual environment if it doesn't exist
if [ -d "env" ]; then
    echo "✓ Virtual environment already exists, skipping creation."
else
    echo "Creating virtual environment..."
    python3 -m venv env
    echo "✓ Virtual environment created successfully!"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

echo "✓ Virtual environment activated!"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

echo ""

# Install requirements
echo "Installing Python dependencies from requirements.txt..."
echo "-----------------------------------"
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed successfully!"
echo ""
echo "==================================="
echo "Setup completed successfully!"
echo "==================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source env/bin/activate"
echo ""
echo "Next steps:"
echo "1. The virtual environment is now active"
echo "2. Run 'jupyter notebook' or open the notebook in VS Code"
echo "3. Start working on your project!"
echo ""
