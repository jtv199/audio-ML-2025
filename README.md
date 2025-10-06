# Audio ML 2025

Machine learning project for Freesound Audio Tagging 2019 Kaggle competition.

## Setup

### 1. Install Kaggle CLI
```bash
pip3 install kaggle
```

### 2. Configure Kaggle Credentials
- Get API token from https://www.kaggle.com/settings/account (click "Create New Token")
- Place `kaggle.json` in `~/.config/kaggle/`
```bash
mkdir -p ~/.config/kaggle
mv ~/Downloads/kaggle.json ~/.config/kaggle/
chmod 600 ~/.config/kaggle/kaggle.json
```

### 3. Download Dataset
```bash
kaggle competitions download -c freesound-audio-tagging-2019
```

### 4. Extract Data
```bash
mkdir -p work/trn_curated
unzip train_curated.zip -d work/trn_curated
```

## Project Structure
- `input/` - Raw competition data (gitignored)
- `work/` - Processed data and outputs (gitignored)
- Code files tracked by git
