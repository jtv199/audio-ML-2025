#!/usr/bin/env python3
"""
Convert test audio files to mel-spectrograms
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os
import librosa

print("="*60)
print("Converting Test Audio to Mel-Spectrograms")
print("="*60)

# Configuration
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

# Paths
DATA = Path('./input')
CSV_SUBMISSION = DATA/'sample_submission.csv'
TEST = DATA/'test'
WORK = Path('work')
IMG_TEST = WORK/'image/test'

print(f"\nTest data directory: {TEST}")
print(f"Output directory: {IMG_TEST}")

# Create output directory
IMG_TEST.mkdir(exist_ok=True, parents=True)

# Load test file list
print(f"\nLoading test file list from: {CSV_SUBMISSION}")
test_df = pd.read_csv(CSV_SUBMISSION)
print(f"Test samples: {len(test_df)}")

# Check if test audio files exist
test_files = list(TEST.glob("*.wav"))
print(f"Found {len(test_files)} test audio files")

if len(test_files) == 0:
    print(f"\nERROR: No test audio files found in {TEST}")
    exit(1)

# Convert audio files to mel-spectrograms
print(f"\nConverting {len(test_df)} test audio files to mel-spectrograms...")
converted_count = 0
skipped_count = 0

for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Converting audio"):
    fname = row['fname']
    audio_path = TEST / fname

    # Skip if audio file doesn't exist
    if not audio_path.exists():
        skipped_count += 1
        continue

    # Create output directory for this file
    filename, _ = os.path.splitext(fname)
    output_dir = IMG_TEST / filename
    output_dir.mkdir(exist_ok=True, parents=True)

    # Output path
    output_path = output_dir / 'mel.png'

    # Skip if already exists
    if output_path.exists():
        converted_count += 1
        continue

    try:
        # Convert audio to mel-spectrogram
        x = read_as_melspectrogram(conf, audio_path, trim_long_data=False)

        # Convert to color image
        x_color = mono_to_color(x)

        # Save as PNG
        Image.fromarray(x_color).save(output_path)
        converted_count += 1

    except Exception as e:
        print(f"\nError processing {fname}: {e}")
        skipped_count += 1

print(f"\n{'='*60}")
print(f"Conversion Summary:")
print(f"{'='*60}")
print(f"Total files: {len(test_df)}")
print(f"Converted: {converted_count}")
print(f"Skipped: {skipped_count}")
print(f"\n✓ Test mel-spectrograms saved to: {IMG_TEST}")
