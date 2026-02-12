#!/usr/bin/env python3
"""
Test ResNet18 Single-Label Model on Test Data
Loads the trained model and generates predictions for submission
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os

# FastAI imports
from fastai.vision.all import *

print("="*60)
print("Testing ResNet18 Single-Label Model")
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

# Paths
DATA = Path('./input')
CSV_SUBMISSION = DATA/'sample_submission.csv'
TEST = DATA/'test'
WORK = Path('work')
IMG_TEST = WORK/'image/test'

print(f"\nLoading test data from: {CSV_SUBMISSION}")
test_df = pd.read_csv(CSV_SUBMISSION)
print(f"Test samples: {len(test_df)}")

# Load trained data info to get classes
CSV_TRN_CURATED = DATA/'train_curated.csv'
df_all = pd.read_csv(CSV_TRN_CURATED)
df = df_all[df_all['labels'].str.contains(',') == False].copy()

print(f"\nNumber of classes (from training): {df['labels'].nunique()}")
classes = sorted(df['labels'].unique())
print(f"Classes: {classes[:10]}... (showing first 10)")

# Check if test images exist
print(f"\nChecking test images in: {IMG_TEST}")
if not IMG_TEST.exists():
    print(f"ERROR: Test image directory not found: {IMG_TEST}")
    print("Please run the mel-spectrogram conversion for test data first.")
    exit(1)

# Count existing test images
test_images = list(IMG_TEST.glob("*/*.png"))
print(f"Found {len(test_images)} test images")

if len(test_images) == 0:
    print("\nERROR: No test images found!")
    print("You need to convert test audio files to mel-spectrograms first.")
    exit(1)

# Create DataBlock for testing (single-label)
IMG_TRN_CURATED = WORK/'image/trn_curated'

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # Single-label
    get_x=ColReader('fname', pref=IMG_TRN_CURATED),
    get_y=ColReader('labels'),  # Single-label, no delim
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(224),
    batch_tfms=[*aug_transforms(size=224, min_scale=0.75),
                Normalize.from_stats(*imagenet_stats)]
)

print("\nCreating DataLoaders...")
dls = dblock.dataloaders(df, bs=64)
print(f"Number of classes in DataLoader: {dls.c}")
print(f"Vocabulary size: {len(dls.vocab)}")

# Load the trained model
print("\nLoading trained ResNet18 model...")
learn = vision_learner(dls, resnet18, pretrained=False, metrics=accuracy)

# Try to load the model
model_path = 'work/models/resnet18_single_label.pth'
if not os.path.exists(model_path):
    model_path = 'models/resnet18_single_label.pkl'
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found!")
        print(f"Checked: work/models/resnet18_single_label.pth")
        print(f"Checked: models/resnet18_single_label.pkl")
        exit(1)

if model_path.endswith('.pth'):
    learn.load('resnet18_single_label')
    print(f"✓ Model loaded from: {model_path}")
else:
    learn = load_learner(model_path)
    print(f"✓ Model loaded from: {model_path}")

print("\nModel loaded successfully!")
print(f"Model vocab: {dls.vocab[:10]}... (showing first 10)")

# Create test dataset
print("\nPreparing test data...")
test_items = []
for idx, row in test_df.iterrows():
    fname = row['fname']
    # Find the image path
    img_base = fname.replace('.wav', '')
    img_path = IMG_TEST / img_base / 'mel.png'

    if not img_path.exists():
        # Try mel2.png
        img_path = IMG_TEST / img_base / 'mel2.png'

    if img_path.exists():
        test_items.append(img_path)
    else:
        print(f"Warning: Image not found for {fname}")

print(f"Valid test items: {len(test_items)}")

if len(test_items) == 0:
    print("\nERROR: No valid test items found!")
    exit(1)

# Create test DataLoader
print("\nGenerating predictions...")
test_dl = learn.dls.test_dl(test_items, with_labels=False)

# Get predictions
preds, _ = learn.get_preds(dl=test_dl)
print(f"Predictions shape: {preds.shape}")

# Convert predictions to class indices (single-label)
pred_classes = preds.argmax(dim=1)
print(f"Predicted classes shape: {pred_classes.shape}")

# Get confidence scores
pred_confidences = preds.max(dim=1)[0]

# Map back to class names
pred_labels = [dls.vocab[idx] for idx in pred_classes]

# Create submission DataFrame (single-label format)
print("\nCreating predictions DataFrame...")
results_df = pd.DataFrame({
    'fname': [item.parent.name + '.wav' for item in test_items],
    'predicted_label': pred_labels,
    'confidence': pred_confidences.cpu().numpy()
})

# Show sample predictions
print("\n" + "="*60)
print("Sample Predictions (Single-Label):")
print("="*60)
print(results_df.head(20).to_string(index=False))

# Show prediction statistics
print("\n" + "="*60)
print("Prediction Statistics:")
print("="*60)
print(f"Total predictions: {len(results_df)}")
print(f"Unique predicted classes: {results_df['predicted_label'].nunique()}")
print(f"Average confidence: {results_df['confidence'].mean():.4f}")
print(f"Min confidence: {results_df['confidence'].min():.4f}")
print(f"Max confidence: {results_df['confidence'].max():.4f}")

# Top predicted classes
print("\n" + "="*60)
print("Top 10 Predicted Classes (Frequency):")
print("="*60)
top_classes = results_df['predicted_label'].value_counts().head(10)
for label, count in top_classes.items():
    percentage = (count / len(results_df)) * 100
    print(f"{label:40s}: {count:4d} ({percentage:5.2f}%)")

# Save results
output_file = 'test_predictions_single_label.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Predictions saved to: {output_file}")

# Also save confidence distribution
print("\n" + "="*60)
print("Confidence Distribution:")
print("="*60)
confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
confidence_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
results_df['confidence_bin'] = pd.cut(results_df['confidence'], bins=confidence_bins, labels=confidence_labels)
print(results_df['confidence_bin'].value_counts().sort_index())

print("\n" + "="*60)
print("Testing Complete!")
print("="*60)
