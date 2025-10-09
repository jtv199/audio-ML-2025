#!/usr/bin/env python3
"""
Audio Length Analysis - Freesound Audio Tagging 2019
Generates histogram of audio file lengths
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Paths
DATA = Path('./input')
CSV_TRN_CURATED = DATA/'train_curated.csv'
TRN_CURATED = DATA/'trn_curated'

# Load CSV
print("Loading training metadata...")
df = pd.read_csv(CSV_TRN_CURATED)

# Analyze audio lengths
print(f"Analyzing {len(df)} audio files...")
audio_lengths = []

for fname in tqdm(df['fname'].values):
    audio_path = TRN_CURATED / fname
    try:
        # Get duration without loading full audio (faster)
        duration = librosa.get_duration(path=str(audio_path))
        audio_lengths.append(duration)
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        continue

# Convert to numpy array
audio_lengths = np.array(audio_lengths)

# Statistics
print("\n=== Audio Length Statistics ===")
print(f"Total files analyzed: {len(audio_lengths)}")
print(f"Mean duration: {audio_lengths.mean():.2f} seconds")
print(f"Median duration: {np.median(audio_lengths):.2f} seconds")
print(f"Min duration: {audio_lengths.min():.2f} seconds")
print(f"Max duration: {audio_lengths.max():.2f} seconds")
print(f"Std deviation: {audio_lengths.std():.2f} seconds")

# Percentiles
print("\nPercentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(audio_lengths, p):.2f} seconds")

# Create histogram
plt.figure(figsize=(12, 6))
plt.hist(audio_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Duration (seconds)')
plt.ylabel('Number of files')
plt.title('Distribution of Audio File Lengths\nFreesound Audio Tagging 2019 - Training Set (Curated)')
plt.grid(True, alpha=0.3)
plt.axvline(audio_lengths.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {audio_lengths.mean():.2f}s')
plt.axvline(np.median(audio_lengths), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(audio_lengths):.2f}s')
plt.legend()
plt.tight_layout()

# Save plot
output_file = 'audio_length_histogram.png'
plt.savefig(output_file, dpi=150)
print(f"\nHistogram saved to: {output_file}")
plt.show()

# Distribution by length bins
print("\n=== Distribution by Length Bins ===")
bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, float('inf'))]
for start, end in bins:
    if end == float('inf'):
        count = np.sum(audio_lengths >= start)
        print(f"{start}+ seconds: {count} files ({count/len(audio_lengths)*100:.1f}%)")
    else:
        count = np.sum((audio_lengths >= start) & (audio_lengths < end))
        print(f"{start}-{end} seconds: {count} files ({count/len(audio_lengths)*100:.1f}%)")
