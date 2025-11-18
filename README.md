# Reto_BloqueIA

PyTorch and PyTorch Lightning project workspace.

## Features

- Deep learning with PyTorch and PyTorch Lightning
- Data preprocessing and augmentation
- Model training with early stopping and checkpointing
- Visualization of results

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for all Python dependencies

## Quick Setup

Run the automated setup script to install all dependencies and create the environment file:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:

1. Create a `.env` file with default configuration (if it doesn't exist)
2. Check for Python 3 and pip installation
3. Install all required Python packages from `requirements.txt`

### Manual Setup

If you prefer to set up manually:

```bash
# Install dependencies
pip3 install -r requirements.txt

# Create .env file (optional)
cp .env.example .env  # Edit with your preferred settings
```

## Project Structure

- `Project_Traffic_sign_classifier.ipynb` - Reference Jupyter notebook
- `requirements.txt` - Python package dependencies
- `setup.sh` - Automated setup script
- `.env` - Environment variables for configuration (created by setup script)

## Usage

1. Make the setup script executable and run it:

   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. Open the Jupyter notebook:

   ```bash
   jupyter notebook Project_Traffic_sign_classifier.ipynb
   ```

   Or open it directly in VS Code.

3. Start working on your project using the installed dependencies.

## Configuration

You can modify parameters in the `.env` file to configure your environment and project settings.

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
