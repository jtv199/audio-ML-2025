# Audio ML 2025

Machine learning project for Freesound Audio Tagging 2019 Kaggle competition.

## Setup

### Quick Install (Recommended)

Run the automated installation script:
```bash
bash install.sh
```

This will install Miniconda and create the conda environment with all required packages. Then activate the environment:
```bash
conda activate freesound
```

### Manual Installation

#### 1. Install Miniconda

Download and install Miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts and install to `~/miniconda3/`. After installation, restart your terminal or run:
```bash
source ~/miniconda3/bin/activate
```

### 2. Create Conda Environment

Create the freesound environment with required packages:
```bash
~/miniconda3/bin/conda create -n freesound python=3.8 -y
~/miniconda3/bin/conda install -n freesound -c conda-forge librosa fastai pandas numpy matplotlib pillow scikit-learn ipython tqdm -y
```

Install Jupyter widgets for notebook support:
```bash
~/miniconda3/envs/freesound/bin/pip install ipywidgets
~/miniconda3/envs/freesound/bin/jupyter nbextension enable --py widgetsnbextension
```

Activate the environment:
```bash
conda activate freesound
```

### 3. Install Kaggle CLI
```bash
pip install kaggle
```

### 4. Configure Kaggle Credentials
- Get API token from https://www.kaggle.com/settings/account (click "Create New Token")
- Place `kaggle.json` in `~/.config/kaggle/`
```bash
mkdir -p ~/.config/kaggle
mv ~/Downloads/kaggle.json ~/.config/kaggle/
chmod 600 ~/.config/kaggle/kaggle.json
```

### 5. Download Dataset
```bash
kaggle competitions download -c freesound-audio-tagging-2019
```

### 6. Extract Data
```bash
mkdir -p input
unzip train_curated.zip -d input/
unzip test.zip -d input/
unzip sample_submission.csv.zip -d input/
unzip train_curated.csv.zip -d input/
```

## Project Structure
- `input/` - Raw competition data (gitignored)
- `work/` - Processed data and outputs (gitignored)
- Code files tracked by git

## File Directory Path
Project root: `/mnt/c/Users/HighOrder/prog/kaggle/free-sound-audio-tagging-2019`

Data paths:
- Input data: `./input/`
- Training data: `./input/trn_curated/`
- Test data: `./input/test/`
- Work directory: `./work/`
