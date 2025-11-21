# Reto_BloqueIA

PyTorch and PyTorch Lightning project workspace for receipt OCR and information extraction using the CORD-v2 dataset.

## Dataset

This project uses the **CORD-v2 (Consolidated Receipt Dataset v2)** from Naver Clova IX:

- Dataset: `naver-clova-ix/cord-v2`
- Source: Hugging Face Datasets
- Content: Receipt images with OCR annotations, bounding boxes, and structured information

The CORD-v2 dataset contains:

- Receipt images
- OCR text annotations
- Bounding box coordinates
- Structured key-value information extraction

## Features

- Deep learning with PyTorch and PyTorch Lightning
- Data preprocessing and augmentation
- Model training with early stopping and checkpointing
- Visualization of results

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for all Python dependencies

## Quick Setup

Run the automated setup script to create a virtual environment and install all dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:

1. Create a Python virtual environment in the `env/` directory (if it doesn't exist)
2. Activate the virtual environment
3. Upgrade pip
4. Install all required Python packages from `requirements.txt`

### Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment
python3 -m venv env

# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `Reto.ipynb` - Main Jupyter notebook for the project
- `Project_Traffic_sign_classifier.ipynb` - Reference Jupyter notebook
- `requirements.txt` - Python package dependencies
- `setup.sh` - Automated setup script
- `env/` - Python virtual environment (created by setup script)

## Usage

1. Make the setup script executable and run it:

   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. The virtual environment will be automatically activated. For future sessions, activate it manually:

   ```bash
   source env/bin/activate
   ```

3. Open the Jupyter notebook:

   ```bash
   jupyter notebook Reto.ipynb
   ```

   Or open it directly in VS Code.

4. Start working on your project using the installed dependencies.

## Virtual Environment

The setup script creates a virtual environment in the `env/` directory to isolate project dependencies.

To activate the virtual environment:

```bash
source env/bin/activate
```

To deactivate:

```bash
deactivate
```

## Configuration

Configure your project settings as needed within your notebooks or by creating configuration files for your specific use case.

## Dependencies

Key libraries used in this project:

- **PyTorch** - Deep learning framework
- **PyTorch Lightning** - High-level PyTorch wrapper
- **torchvision** - Computer vision utilities
- **OpenCV** - Image processing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization

## License

See LICENSE file for details.
