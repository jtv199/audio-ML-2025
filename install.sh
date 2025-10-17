#!/bin/bash

# Freesound Audio Tagging 2019 - Installation Script
# This script automates the setup process for the project

set -e  # Exit on error

echo "=== Freesound Audio Tagging 2019 Setup ==="
echo ""

# Check if miniconda is installed
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    echo "Miniconda installed successfully!"
else
    echo "Miniconda already installed at ~/miniconda3"
fi

# Initialize conda
source $HOME/miniconda3/bin/activate

# Check if freesound environment exists
if conda env list | grep -q "^freesound "; then
    echo "Conda environment 'freesound' already exists"
    read -p "Do you want to recreate it? (y/N): " recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n freesound -y
    else
        echo "Skipping environment creation"
        exit 0
    fi
fi

echo "Creating conda environment 'freesound'..."
$HOME/miniconda3/bin/conda create -n freesound python=3.8 -y

echo "Installing packages..."
$HOME/miniconda3/bin/conda install -n freesound -c conda-forge librosa fastai pandas numpy matplotlib pillow scikit-learn ipython tqdm -y

echo "Installing ipywidgets for Jupyter notebook support..."
$HOME/miniconda3/envs/freesound/bin/pip install ipywidgets
$HOME/miniconda3/envs/freesound/bin/jupyter nbextension enable --py widgetsnbextension

echo "Cleaning up conda cache..."
$HOME/miniconda3/bin/conda clean --all -y

echo ""
echo "=== Installation Complete! ==="
echo ""
echo "To activate the environment, run:"
echo "  conda activate freesound"
echo ""
echo "Next steps:"
echo "  1. Configure Kaggle credentials (see README.md)"
echo "  2. Download dataset with: kaggle competitions download -c freesound-audio-tagging-2019"
echo "  3. Extract data (see README.md)"
echo ""
