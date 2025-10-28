#!/usr/bin/env python3
"""
Per-class error analysis for DeiT Tiny vision transformer model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# FastAI imports
from fastai.vision.all import *
import PIL
import random

# Define the custom ImageOpener class that was used during training
class ImageOpener(Transform):
    """Custom image opener for DeiT model - must match training setup"""
    def encodes(self, fn):
        # For inference, we just load the image directly from file
        return PILImage.create(fn)

print("="*80)
print("DEIT TINY PER-CLASS ERROR ANALYSIS")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load validation data
print("Loading validation data...")
val_df = pd.read_csv('work/tokenized/tokenized_train_curated.csv')
print(f"Validation data loaded: {val_df.shape}")

# Filter to single-label samples only
val_df['num_labels'] = val_df['labels'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
single_label_df = val_df[val_df['num_labels'] == 1].copy()
print(f"Single-label samples: {len(single_label_df)} ({len(single_label_df)/len(val_df)*100:.1f}%)")

# Reset index
single_label_df = single_label_df.reset_index(drop=True)

# Take 20% as validation (matching the training split)
from sklearn.model_selection import train_test_split
_, val_idx = train_test_split(
    range(len(single_label_df)),
    test_size=0.2,
    random_state=42,
    stratify=single_label_df['labels']
)

val_df_subset = single_label_df.iloc[val_idx].copy()
print(f"Validation subset: {len(val_df_subset)} samples")

# Load the DeiT Tiny model
print("\nLoading DeiT Tiny model...")
model_path = Path('models/deit_tiny_single_label.pkl')
learn = load_learner(model_path)
print(f"Model loaded from: {model_path}")

# Create predictions
print("\nGenerating predictions...")
y_pred = []
y_true = []

errors = 0
for idx, row in val_df_subset.iterrows():
    fname = row['fname']
    true_label = row['labels']

    # Construct image path (mel-spectrogram)
    img_path = Path('work/mel_spectrograms') / f"{fname.replace('.wav', '.png')}"

    if not img_path.exists():
        errors += 1
        continue

    try:
        # Predict
        pred_class, pred_idx, pred_probs = learn.predict(img_path)

        # Handle multi-label predictions: take the first/highest confidence label
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
            pred_class_str = "Unknown"

        y_pred.append(pred_class_str)
        y_true.append(true_label)

    except Exception as e:
        errors += 1
        print(f"Error on {fname}: {e}")
        continue

print(f"\nPredictions completed: {len(y_pred)} samples")
print(f"Errors/skipped: {errors}")

# Calculate overall accuracy
overall_acc = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

# Per-class analysis
print("\n" + "="*80)
print("PER-CLASS ANALYSIS")
print("="*80)

unique_classes = sorted(set(y_true))
print(f"Total classes: {len(unique_classes)}")

class_results = []
for class_name in unique_classes:
    # Get indices for this class
    class_mask = [i for i, label in enumerate(y_true) if label == class_name]

    if len(class_mask) == 0:
        continue

    # Calculate metrics for this class
    y_true_class = [1 if label == class_name else 0 for label in y_true]
    y_pred_class = [1 if label == class_name else 0 for label in y_pred]

    precision = precision_score(y_true_class, y_pred_class, zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, zero_division=0)

    # Calculate per-class accuracy
    correct = sum([1 for i in class_mask if y_pred[i] == y_true[i]])
    total = len(class_mask)
    accuracy = correct / total if total > 0 else 0

    class_results.append({
        'Class': class_name,
        'Total_Samples': total,
        'Correct': correct,
        'Incorrect': total - correct,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    })

# Create DataFrame
results_df = pd.DataFrame(class_results)
results_df = results_df.sort_values('Accuracy', ascending=False)

# Summary statistics
perfect_classes = len(results_df[results_df['Accuracy'] == 1.0])
failed_classes = len(results_df[results_df['Accuracy'] == 0.0])
mean_acc = results_df['Accuracy'].mean()
median_acc = results_df['Accuracy'].median()

print(f"\nSummary Statistics:")
print(f"  Total classes: {len(results_df)}")
print(f"  Perfect classes (100%): {perfect_classes}")
print(f"  Failed classes (0%): {failed_classes}")
print(f"  Mean accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
print(f"  Median accuracy: {median_acc:.4f} ({median_acc*100:.2f}%)")

# Top 10 classes
print("\n" + "="*80)
print("TOP 10 CLASSES BY ACCURACY")
print("="*80)
print(results_df.head(10).to_string(index=False))

# Bottom 10 classes
print("\n" + "="*80)
print("BOTTOM 10 CLASSES BY ACCURACY")
print("="*80)
print(results_df.tail(10).to_string(index=False))

# Save results
output_file = f"deit_tiny_per_class_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

# Also save to the findings folder for consistency
findings_file = 'claude/lazypredict/findings/error_analysis_deit_tiny.csv'
results_df.to_csv(findings_file, index=False)
print(f"✓ Results also saved to: {findings_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nDeiT Tiny Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
