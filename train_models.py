#!/usr/bin/env python3
"""
Audio Classification Training Script - Single Label
Converted from Jupyter notebook for background execution with GPU support
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for background execution
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL
import os
import csv
from datetime import datetime
import sys

# Check for GPU and set device
import torch
print(f"\n{'='*60}")
print(f"GPU/CUDA Check")
print(f"{'='*60}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print(f"WARNING: CUDA not available, using CPU (will be very slow!)")
    device = torch.device('cpu')
print(f"{'='*60}\n")

# Audio processing libraries
import librosa
import librosa.display

# FastAI
from fastai.vision.all import *
import random
import timm
import gc

# Configuration
DEBUG_MODE = False
DEBUG_TRAIN_SIZE = 1000
DEBUG_TEST_SIZE = 500

DATA = Path('./input')
CSV_TRN_CURATED = DATA/'train_curated.csv'
CSV_TRN_NOISY = DATA/'train_noisy.csv'
CSV_SUBMISSION = DATA/'sample_submission.csv'
TRN_CURATED = DATA/'trn_curated'
TRN_NOISY = DATA/'train_noisy'
TEST = DATA/'test'

WORK = Path('work')
IMG_TRN_CURATED = WORK/'image/trn_curated'
IMG_TRN_NOISY = WORK/'image/train_noisy'
IMG_TEST = WORK/'image/test'
for folder in [WORK, IMG_TRN_CURATED, IMG_TRN_NOISY, IMG_TEST, Path('models')]:
    Path(folder).mkdir(exist_ok=True, parents=True)

# Load data
print("Loading data...")
if DEBUG_MODE:
    print(f"🐛 DEBUG MODE: Loading {DEBUG_TRAIN_SIZE} training samples and {DEBUG_TEST_SIZE} test samples")
    df = pd.read_csv(CSV_TRN_CURATED, nrows=DEBUG_TRAIN_SIZE)
    test_df = pd.read_csv(CSV_SUBMISSION, nrows=DEBUG_TEST_SIZE)
else:
    print("📊 FULL MODE: Loading complete dataset")
    df = pd.read_csv(CSV_TRN_CURATED)
    test_df = pd.read_csv(CSV_SUBMISSION)

print(f"Training samples: {len(df)}")
print(f"Test samples: {len(test_df)}")

# Filter for single-labeled samples only
df['label_count'] = df['labels'].str.count(',') + 1
df_single = df[df['label_count'] == 1].copy()
df_single['label'] = df_single['labels']
print(f"\nFiltered to single-labeled samples: {len(df_single)} (from {len(df)})")
print(f"Label distribution:\n{df_single['label'].value_counts().head(10)}")
df = df_single

# Audio configuration
class conf:
    sampling_rate = 44100
    duration = 2
    hop_length = 347*duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration

# Audio processing functions
def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(str(pathname), sr=conf.sampling_rate)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)
    if len(y) > conf.samples:
        if trim_long_data:
            y = y[0:0+conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(y=audio,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def read_as_melspectrogram(conf, pathname, trim_long_data):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    return mels

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    X = np.stack([X, X, X], axis=-1)
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def convert_wav_to_image(df, source, img_dest):
    X = []
    print(f"Converting {len(df)} audio files to mel-spectrograms...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)
        x_color = mono_to_color(x)
        X.append(x_color)
    return X

# Convert audio to images
print("\nConverting training audio to images...")
X_train = convert_wav_to_image(df, source=TRN_CURATED, img_dest=IMG_TRN_CURATED)
print("Converting test audio to images...")
X_test = convert_wav_to_image(test_df, source=TEST, img_dest=IMG_TEST)

# Custom image loading for fastai
CUR_X_FILES, CUR_X = list(df.fname.values), X_train

def open_fat2019_image(fn):
    fname = fn.name if hasattr(fn, 'name') else str(fn).split('/')[-1]
    idx = CUR_X_FILES.index(fname)
    x = PIL.Image.fromarray(CUR_X[idx])
    time_dim, base_dim = x.size
    crop_x = random.randint(0, time_dim - base_dim)
    x = x.crop([crop_x, 0, crop_x+base_dim, base_dim])
    return PILImage.create(x)

# Create DataBlock
class ImageOpener(Transform):
    def encodes(self, fn):
        return open_fat2019_image(fn)

print("\nCreating DataLoaders...")
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader('fname', pref=WORK/'image/trn_curated'),
    get_y=ColReader('label'),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(224),
    batch_tfms=[*aug_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=1.0, max_warp=0.),
                Normalize.from_stats(*imagenet_stats)]
)

dblock.type_tfms[0] = ImageOpener()
dls = dblock.dataloaders(df, bs=64)

print(f"Training batches: {len(dls.train)}")
print(f"Validation batches: {len(dls.valid)}")
print(f"Number of classes: {len(dls.vocab)}")

# Training utilities
def save_epoch_results(model_name, epoch_results, filename='training_results.csv'):
    """Save epoch training results to CSV file"""
    file_exists = Path(filename).exists()

    with open(filename, 'a', newline='') as f:
        fieldnames = ['model_name', 'timestamp', 'epoch', 'train_loss', 'valid_loss', 'accuracy', 'lr']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for result in epoch_results:
            result['model_name'] = model_name
            result['timestamp'] = datetime.now().isoformat()
            writer.writerow(result)

    print(f"✓ Saved {len(epoch_results)} epoch results to {filename}")

def train_model(model_name, dls, epochs_config, save_name=None, export_name=None):
    """Train a model with specified configuration"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")

    # Create learner - fastai will automatically use CUDA if available
    learn = vision_learner(dls, model_name, pretrained=False, metrics=[accuracy])
    learn.unfreeze()

    # Verify model is on GPU
    model_device = next(learn.model.parameters()).device
    print(f"Model is on device: {model_device}")

    all_results = []

    # Training phases
    for phase_idx, (epochs, lr) in enumerate(epochs_config):
        print(f"\n--- Phase {phase_idx + 1}: {epochs} epochs with lr={lr} ---")

        if phase_idx == 0:
            print("Finding optimal learning rate...")
            learn.lr_find()

        # Train
        print(f"Training...")
        learn.fit_one_cycle(epochs, lr)

        # Extract results from recorder
        for i, values in enumerate(learn.recorder.values):
            epoch_num = len(all_results) + 1
            result = {
                'epoch': epoch_num,
                'train_loss': float(values[0]) if len(values) > 0 else None,
                'valid_loss': float(values[1]) if len(values) > 1 else None,
                'accuracy': float(values[2]) if len(values) > 2 else None,
                'lr': float(learn.recorder.lrs[i][-1]) if i < len(learn.recorder.lrs) else None
            }
            all_results.append(result)

    # Save results
    save_epoch_results(model_name, all_results)

    # Save model
    if save_name:
        learn.save(save_name)
        print(f"✓ Saved model weights to {save_name}")

    if export_name:
        learn.export(export_name)
        print(f"✓ Exported model to {export_name}")

    print(f"\n{'='*60}")
    print(f"Completed training {model_name}")
    print(f"Total epochs: {len(all_results)}")
    print(f"Best accuracy: {max([r['accuracy'] for r in all_results if r['accuracy'] is not None]):.4f}")
    print(f"{'='*60}\n")

    return learn, all_results

# Training configurations
print("\n" + "="*60)
print("Starting Model Training")
print("="*60 + "\n")

# Train ResNet18
resnet18_config = [
    (5, 1e-1),
    (10, 1e-2),
    (20, 3e-3),
    (20, 1e-3),
    (50, slice(1e-3, 3e-3)),
    (10, slice(1e-4, 1e-3))
]

try:
    learn_resnet18, results_resnet18 = train_model(
        model_name='resnet18',
        dls=dls,
        epochs_config=resnet18_config,
        save_name='resnet18_single_label',
        export_name='models/resnet18_single_label.pkl'
    )
    print("✓ ResNet18 training completed successfully")
except Exception as e:
    print(f"✗ ResNet18 training failed: {e}")
    import traceback
    traceback.print_exc()

# Train DeiT Tiny
deit_config = [
    (5, 1e-1),
    (10, 1e-2),
    (20, 3e-3),
    (20, 1e-3),
    (50, slice(1e-3, 3e-3)),
    (10, slice(1e-4, 1e-3))
]

try:
    learn_deit, results_deit = train_model(
        model_name='deit_tiny_patch16_224',
        dls=dls,
        epochs_config=deit_config,
        save_name='deit_tiny_single_label',
        export_name='models/deit_tiny_single_label.pkl'
    )
    print("✓ DeiT Tiny training completed successfully")
except Exception as e:
    print(f"✗ DeiT Tiny training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
