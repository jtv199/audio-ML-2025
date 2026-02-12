#!/usr/bin/env python3
"""
ESC-50 Feature Extraction Script
Extracts the same features (STFT, MEL, CQT) as the FreeSound dataset.
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ESC-50 FEATURE EXTRACTION")
print("=" * 80)
print()

# Paths
ESC50_PATH = Path('data/esc50/ESC-50-master')
AUDIO_PATH = ESC50_PATH / 'audio'
META_PATH = ESC50_PATH / 'meta' / 'esc50.csv'
OUTPUT_PATH = Path('data/esc50/esc50_features.csv')

# Audio parameters (match FreeSound preprocessing)
SAMPLE_RATE = 44100
DURATION = 5  # seconds
N_SAMPLES = SAMPLE_RATE * DURATION

def extract_features(audio_path):
    """
    Extract STFT, MEL, and CQT features from audio file.
    Matches the feature extraction used for FreeSound dataset.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

        # Pad or trim to fixed length
        if len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)), mode='constant')
        else:
            y = y[:N_SAMPLES]

        features = {}

        # 1. STFT features
        stft = np.abs(librosa.stft(y))
        stft_mean = np.mean(stft, axis=1)
        stft_std = np.std(stft, axis=1)

        for i, val in enumerate(stft_mean):
            features[f'stft_mean_{i}'] = val
        for i, val in enumerate(stft_std):
            features[f'stft_std_{i}'] = val

        # 2. Mel spectrogram features
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)
        mel_std = np.std(mel_db, axis=1)

        for i, val in enumerate(mel_mean):
            features[f'mel_mean_{i}'] = val
        for i, val in enumerate(mel_std):
            features[f'mel_std_{i}'] = val

        # 3. CQT features
        cqt = np.abs(librosa.cqt(y, sr=sr))
        cqt_mean = np.mean(cqt, axis=1)
        cqt_std = np.std(cqt, axis=1)

        for i, val in enumerate(cqt_mean):
            features[f'cqt_mean_{i}'] = val
        for i, val in enumerate(cqt_std):
            features[f'cqt_std_{i}'] = val

        return features

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def main():
    """Main extraction pipeline"""

    # Load metadata
    print("Loading metadata...")
    meta_df = pd.read_csv(META_PATH)
    print(f"  Found {len(meta_df)} audio files")
    print(f"  Number of categories: {meta_df['target'].nunique()}")
    print()

    # Extract features for all files
    print("Extracting features...")
    all_features = []

    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Processing"):
        audio_file = AUDIO_PATH / row['filename']

        if not audio_file.exists():
            print(f"Warning: {audio_file} not found, skipping...")
            continue

        features = extract_features(audio_file)

        if features is not None:
            features['filename'] = row['filename']
            features['fold'] = row['fold']
            features['target'] = row['target']
            features['category'] = row['category']
            features['esc10'] = row['esc10']
            all_features.append(features)

    # Create DataFrame
    print(f"\nCreating feature DataFrame...")
    df_features = pd.DataFrame(all_features)

    # Reorder columns: metadata first, then features
    meta_cols = ['filename', 'fold', 'target', 'category', 'esc10']
    feature_cols = [col for col in df_features.columns if col not in meta_cols]
    df_features = df_features[meta_cols + sorted(feature_cols)]

    print(f"  Total samples: {len(df_features)}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"\nFeature breakdown:")

    # Count feature types
    stft_count = len([c for c in feature_cols if c.startswith('stft')])
    mel_count = len([c for c in feature_cols if c.startswith('mel')])
    cqt_count = len([c for c in feature_cols if c.startswith('cqt')])

    print(f"  STFT features: {stft_count}")
    print(f"  MEL features: {mel_count}")
    print(f"  CQT features: {cqt_count}")

    # Save to CSV
    print(f"\nSaving features to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 80}")
    print("FEATURE EXTRACTION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\nOutput file: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    print()


if __name__ == "__main__":
    main()
