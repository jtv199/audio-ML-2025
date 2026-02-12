#!/usr/bin/env python
"""
ResNet18 training script for single-label audio classification
Based on mobilnet-test.ipynb but modified for single-label classification
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL
import random
import librosa
import librosa.display

from fastai.vision.all import *
import torch

# Configuration
class conf:
    # Preprocessing settings
    sampling_rate = 44100
    duration = 2
    hop_length = 347 * duration  # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration

# File/folder definitions
DATA = Path('./input')
CSV_TRN_CURATED = DATA/'train_curated.csv'
TRN_CURATED = DATA/'trn_curated'
WORK = Path('work')
IMG_TRN_CURATED = WORK/'image/trn_curated_single_label'

# Create directories
for folder in [WORK, IMG_TRN_CURATED]:
    Path(folder).mkdir(exist_ok=True, parents=True)

print("Loading and filtering dataset to single-label samples only...")
# Load data
df_all = pd.read_csv(CSV_TRN_CURATED)
print(f"Total samples: {len(df_all)}")

# Filter to only single-label samples
df = df_all[df_all['labels'].str.contains(',') == False].copy()
print(f"Single-label samples: {len(df)}")
print(f"Number of unique labels: {df['labels'].nunique()}")
print(f"\nLabel distribution:\n{df['labels'].value_counts()}")

# Audio conversion functions
def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(str(pathname), sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y):  # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples:  # long enough
        if trim_long_data:
            y = y[0:0+conf.samples]
    else:  # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
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
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Scale to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def convert_wav_to_image(df, source):
    X = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Converting audio to images"):
        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)
        x_color = mono_to_color(x)
        X.append(x_color)
    return X

print("\nConverting audio files to mel-spectrograms...")
X_train = convert_wav_to_image(df, source=TRN_CURATED)

# Custom image loading function for fastai
CUR_X_FILES, CUR_X = list(df.fname.values), X_train

class ImageOpener(Transform):
    def encodes(self, fn):
        # open
        fname = fn.name if hasattr(fn, 'name') else str(fn).split('/')[-1]
        idx = CUR_X_FILES.index(fname)
        x = PIL.Image.fromarray(CUR_X[idx])
        # crop
        time_dim, base_dim = x.size
        crop_x = random.randint(0, time_dim - base_dim)
        x = x.crop([crop_x, 0, crop_x+base_dim, base_dim])
        # return fastai PILImage
        return PILImage.create(x)

print("\nCreating DataLoaders for single-label classification...")
# Create DataBlock for single-label classification
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # CategoryBlock for single-label
    get_x=ColReader('fname', pref=IMG_TRN_CURATED),
    get_y=ColReader('labels'),  # No label_delim for single-label
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(224),
    batch_tfms=[*aug_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=1.0, max_warp=0.),
                Normalize.from_stats(*imagenet_stats)]
)

# Override the image opener
dblock.type_tfms[0] = ImageOpener()

# Create DataLoaders
dls = dblock.dataloaders(df, bs=64)

print(f"\nDataLoader created:")
print(f"  Training batches: {len(dls.train)}")
print(f"  Validation batches: {len(dls.valid)}")
print(f"  Number of classes: {len(dls.vocab)}")
print(f"  Classes: {dls.vocab}")

print("\nCreating ResNet18 model (training from scratch)...")
# Create learner with ResNet18 and accuracy metric
learn = vision_learner(dls, resnet18, pretrained=False, metrics=accuracy)
learn.unfreeze()

print("\nStarting training (20 epochs total)...")
print("=" * 60)

# Train for 20 epochs total
# Split into phases for better convergence
print("\nPhase 1: Initial training (5 epochs, lr=1e-1)")
learn.fit_one_cycle(5, 1e-1)

print("\nPhase 2: Fine-tuning (10 epochs, lr=1e-2)")
learn.fit_one_cycle(10, 1e-2)

print("\nPhase 3: Final refinement (5 epochs, lr=1e-3)")
learn.fit_one_cycle(5, 1e-3)

print("\n" + "=" * 60)
print("Training complete!")

# Save model
print("\nSaving model...")
learn.save('resnet18_single_label')
learn.export('models/resnet18_single_label.pkl')
print(f"Model saved to:")
print(f"  - work/models/resnet18_single_label.pth")
print(f"  - models/resnet18_single_label.pkl")

# Print final metrics
print("\n" + "=" * 60)
print("FINAL RESULTS:")
print("=" * 60)
valid_loss, valid_acc = learn.validate()
print(f"Validation Loss: {valid_loss:.4f}")
print(f"Validation Accuracy: {valid_acc:.4f} ({valid_acc*100:.2f}%)")
print("=" * 60)

print("\nDone!")
