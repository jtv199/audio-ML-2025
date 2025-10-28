#!/usr/bin/env python3
"""
DeiT Tiny Error Analysis
Load the trained DeiT model and perform comprehensive error analysis
Same approach as MobileNet analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from fastai.vision.all import *
import librosa
import PIL
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import random
import warnings
warnings.filterwarnings('ignore')

# LWLRAP metric (required for model loading)
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample."""
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    retrieved_classes = np.argsort(scores)[::-1]
    class_rankings = np.zeros(num_classes, dtype=np.int32)
    class_rankings[retrieved_classes] = range(num_classes)
    retrieved_class_true = np.zeros(num_classes, dtype=bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float32)))
    return pos_class_indices, precision_at_hits

def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision."""
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    return per_class_lwlrap, weight_per_class

def lwlrap(preds, targs):
    """Wrapper for fast.ai library"""
    scores = preds.cpu().numpy() if hasattr(preds, 'cpu') else preds
    truth = targs.cpu().numpy() if hasattr(targs, 'cpu') else targs
    score, weight = calculate_per_class_lwlrap(truth, scores)
    return (score * weight).sum()

# Custom image opener for FastAI (required for model loading)
class ImageOpener(Transform):
    def encodes(self, fn):
        return open_fat2019_image(fn)

# Global variables for image loading
CUR_X_FILES, CUR_X = [], []

def open_fat2019_image(fn):
    """Custom image loading function for fastai"""
    fname = fn.name if hasattr(fn, 'name') else str(fn).split('/')[-1]
    if fname in CUR_X_FILES:
        idx = CUR_X_FILES.index(fname)
        x = PIL.Image.fromarray(CUR_X[idx])
        # crop
        time_dim, base_dim = x.size
        if time_dim >= base_dim:
            crop_x = (time_dim - base_dim) // 2
            x = x.crop([crop_x, 0, crop_x+base_dim, base_dim])
        return PILImage.create(x)
    else:
        # Fallback: just return a blank image
        return PILImage.create(np.zeros((128, 128, 3), dtype=np.uint8))

print("🚀 Starting DeiT Tiny Error Analysis\n")

# Paths
DATA = Path('./input')
WORK = Path('work')
CSV_TRN_CURATED = DATA/'train_curated.csv'
TRN_CURATED = DATA/'trn_curated'
MODEL_PATH = Path('models/deit_tiny_single_label.pkl')

# Check if model exists
if not MODEL_PATH.exists():
    print(f"❌ Model not found at {MODEL_PATH}")
    print("Looking for alternative models...")
    alternatives = list(Path('models').glob('*deit*.pkl'))
    if alternatives:
        print(f"Found: {alternatives}")
        MODEL_PATH = alternatives[0]
        print(f"Using: {MODEL_PATH}")
    else:
        raise FileNotFoundError("No DeiT model files found!")

print(f"✓ Model found: {MODEL_PATH}")

# Load training data (single-label only)
print("\n📊 Loading training data...")
df = pd.read_csv(CSV_TRN_CURATED)
df['label_count'] = df['labels'].str.count(',') + 1
df_single = df[df['label_count'] == 1].copy()
df_single['label'] = df_single['labels']

print(f"Total samples: {len(df)}")
print(f"Single-label samples: {len(df_single)} ({len(df_single)/len(df)*100:.1f}%)")
print(f"Unique classes: {df_single['label'].nunique()}")

# Audio processing configuration
class conf:
    sampling_rate = 44100
    duration = 2
    hop_length = 347 * duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration

def read_audio(pathname, trim_long_data=False):
    """Read and preprocess audio file"""
    y, sr = librosa.load(str(pathname), sr=conf.sampling_rate)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)
    if len(y) > conf.samples:
        if trim_long_data:
            y = y[0:conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(audio):
    """Convert audio to mel-spectrogram"""
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=conf.sampling_rate,
        n_mels=conf.n_mels,
        hop_length=conf.hop_length,
        n_fft=conf.n_fft,
        fmin=conf.fmin,
        fmax=conf.fmax
    )
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def mono_to_color(X, eps=1e-6):
    """Convert mono mel-spectrogram to 3-channel image"""
    X = np.stack([X, X, X], axis=-1)
    mean = X.mean()
    std = X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    if (_max - _min) > eps:
        V = 255 * (Xstd - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def read_as_melspectrogram(pathname):
    """Read audio file and convert to mel-spectrogram image"""
    x = read_audio(pathname, trim_long_data=False)
    mels = audio_to_melspectrogram(x)
    return mono_to_color(mels)

print("\n🔄 Loading trained model...")
try:
    learn = load_learner(MODEL_PATH)
    print(f"✓ Model loaded successfully")
    print(f"  Vocab size: {len(learn.dls.vocab)}")
    print(f"  Classes: {learn.dls.vocab[:10]}..." if len(learn.dls.vocab) > 10 else f"  Classes: {learn.dls.vocab}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Prepare validation dataset
print("\n📝 Preparing validation set...")
# Use 20% as validation (same split as training)
np.random.seed(42)
val_indices = np.random.choice(len(df_single), size=int(0.2 * len(df_single)), replace=False)
val_df = df_single.iloc[val_indices].reset_index(drop=True)

print(f"Validation samples: {len(val_df)}")

# Make predictions
print("\n🔮 Making predictions on validation set...")
y_true = []
y_pred = []
y_pred_proba = []

for idx, row in val_df.iterrows():
    if idx % 100 == 0:
        print(f"  Processing {idx}/{len(val_df)}...")

    try:
        # Load and process audio
        audio_path = TRN_CURATED / row['fname']
        img_array = read_as_melspectrogram(audio_path)
        img_pil = PIL.Image.fromarray(img_array)

        # Random crop to 128x128 (same as training)
        time_dim, base_dim = img_pil.size
        if time_dim >= base_dim:
            crop_x = (time_dim - base_dim) // 2
            img_pil = img_pil.crop([crop_x, 0, crop_x + base_dim, base_dim])

        # Convert to FastAI PILImage
        img = PILImage.create(img_pil)

        # Make prediction
        pred_class, pred_idx, pred_probs = learn.predict(img)

        # FastAI returns L (fastcore.foundation.L) object for multi-label
        # Convert to regular Python list
        if hasattr(pred_class, 'items'):
            # L object from fastcore
            pred_list = list(pred_class.items)
        elif isinstance(pred_class, (list, tuple)):
            pred_list = list(pred_class)
        else:
            # Single string
            pred_list = [str(pred_class)]

        # Take first prediction (highest confidence for multi-label)
        if len(pred_list) > 0:
            pred_class_str = pred_list[0]
        else:
            pred_class_str = "Unknown"  # No prediction made

        y_true.append(row['label'])
        y_pred.append(pred_class_str)
        y_pred_proba.append(pred_probs.numpy())

    except Exception as e:
        print(f"    Error processing {row['fname']}: {e}")
        continue

print(f"\n✓ Predictions complete: {len(y_pred)}/{len(val_df)} samples")

# Debug: Check a few predictions
print(f"\n🔍 Sample predictions:")
for i in range(min(10, len(y_pred))):
    match = "✓" if y_true[i] == y_pred[i] else "✗"
    print(f"  {match} True: {y_true[i]:<40s} | Pred: {y_pred[i]}")

# Calculate overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\n🎯 Overall Accuracy: {accuracy*100:.2f}%")

# Per-class analysis
print("\n📊 Calculating per-class metrics...")
classes = sorted(set(y_true))
results = []

for cls in classes:
    # Get indices for this class
    true_indices = [i for i, label in enumerate(y_true) if label == cls]

    if len(true_indices) == 0:
        continue

    # Calculate metrics
    tp = sum([1 for i in true_indices if y_pred[i] == cls])
    fn = len(true_indices) - tp
    fp = sum([1 for i, pred in enumerate(y_pred) if pred == cls and y_true[i] != cls])

    total_samples = len(true_indices)
    correct = tp
    class_accuracy = correct / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results.append({
        'Class': cls,
        'Total_Samples': total_samples,
        'Correct': correct,
        'Accuracy': class_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'TP': tp,
        'FP': fp,
        'FN': fn
    })

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

# Save results
output_file = 'claude/lazypredict/findings/error_analysis_deit_tiny.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

# Print summary statistics
print("\n" + "="*70)
print("DEIT TINY ERROR ANALYSIS SUMMARY")
print("="*70)
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"Mean Per-Class Accuracy: {results_df['Accuracy'].mean()*100:.2f}%")
print(f"Median Per-Class Accuracy: {results_df['Accuracy'].median()*100:.2f}%")
print(f"Std Dev: {results_df['Accuracy'].std()*100:.2f}%")

# Top/Bottom classes
print("\n" + "="*70)
print("TOP 10 CLASSES")
print("="*70)
top10 = results_df.head(10)
for _, row in top10.iterrows():
    print(f"{row['Class']:40s} {row['Accuracy']*100:5.1f}% ({row['Correct']:2d}/{row['Total_Samples']:2d})")

print("\n" + "="*70)
print("BOTTOM 10 CLASSES")
print("="*70)
bottom10 = results_df.tail(10)
for _, row in bottom10.iterrows():
    print(f"{row['Class']:40s} {row['Accuracy']*100:5.1f}% ({row['Correct']:2d}/{row['Total_Samples']:2d})")

# Compare with MobileNet
print("\n" + "="*70)
print("COMPARISON WITH MOBILENET")
print("="*70)

try:
    mobilenet_df = pd.read_csv('claude/lazypredict/findings/error_analysis_mobilenet.csv')

    # Merge results
    comparison = results_df[['Class', 'Accuracy']].merge(
        mobilenet_df[['Class', 'Accuracy']],
        on='Class',
        suffixes=('_deit', '_mobilenet'),
        how='inner'
    )
    comparison['Difference'] = (comparison['Accuracy_deit'] - comparison['Accuracy_mobilenet']) * 100
    comparison = comparison.sort_values('Difference', ascending=False)

    print(f"\nDeiT Tiny Overall: {accuracy*100:.2f}%")
    print(f"MobileNet Overall: {mobilenet_df['Accuracy'].mean()*100:.2f}%")
    print(f"Difference: {(accuracy - mobilenet_df['Accuracy'].mean())*100:+.2f}%")

    print("\n📈 Classes where DeiT is BETTER:")
    better = comparison[comparison['Difference'] > 0]
    if len(better) > 0:
        for _, row in better.head(10).iterrows():
            print(f"  {row['Class']:40s} {row['Difference']:+6.1f}% "
                  f"({row['Accuracy_deit']*100:.1f}% vs {row['Accuracy_mobilenet']*100:.1f}%)")
    else:
        print("  (None)")

    print("\n📉 Classes where DeiT is WORSE:")
    worse = comparison[comparison['Difference'] < 0]
    if len(worse) > 0:
        for _, row in worse.tail(10).iterrows():
            print(f"  {row['Class']:40s} {row['Difference']:+6.1f}% "
                  f"({row['Accuracy_deit']*100:.1f}% vs {row['Accuracy_mobilenet']*100:.1f}%)")

    # Save comparison
    comparison.to_csv('claude/lazypredict/findings/deit_vs_mobilenet_comparison.csv', index=False)
    print(f"\n✓ Comparison saved to: claude/lazypredict/findings/deit_vs_mobilenet_comparison.csv")

except Exception as e:
    print(f"  Could not compare with MobileNet: {e}")

# Sample size analysis
print("\n" + "="*70)
print("SAMPLE SIZE ANALYSIS")
print("="*70)

bins = [0, 5, 10, 15, 20, 100]
labels = ['1-4', '5-9', '10-14', '15-19', '20+']
results_df['sample_range'] = pd.cut(results_df['Total_Samples'], bins=bins, labels=labels)

for label in labels:
    subset = results_df[results_df['sample_range'] == label]
    if len(subset) > 0:
        print(f"{label:10s}: {len(subset):2d} classes | Avg: {subset['Accuracy'].mean()*100:5.1f}%")

print(f"\nCorrelation (samples vs accuracy): {results_df['Total_Samples'].corr(results_df['Accuracy']):.3f}")

print("\n✅ Analysis complete!")
