#!/usr/bin/env python3
"""
VGGish+Tabular Confusion Analysis
Load the trained VGGish+Tabular model and analyze which classes are confused with each other
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# FastAI imports
from fastai.tabular.all import *

print("="*80)
print("VGGISH+TABULAR CONFUSION ANALYSIS")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load the model
model_path = Path('models/vggish_tabular_2layers_20251016_182328.pkl')
print(f"Loading model from: {model_path}")

try:
    learn = load_learner(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Vocab size: {len(learn.dls.vocab)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Load data
print("\n📊 Loading training data...")
vggish_train = pd.read_csv('work/tokenized/vggish_embeddings_train_curated.csv')
tabular_train = pd.read_csv('work/trn_curated_feature.csv')

# Rename column to match
tabular_train = tabular_train.rename(columns={'file': 'fname'})

# Merge datasets
train_df = vggish_train.merge(tabular_train, on='fname', how='inner')
print(f"Merged data: {train_df.shape}")

# Filter to single-label samples
train_df['num_labels'] = train_df['labels'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
single_label_df = train_df[train_df['num_labels'] == 1].copy()
single_label_df['label'] = single_label_df['labels']

print(f"Single-label samples: {len(single_label_df)} ({len(single_label_df)/len(train_df)*100:.1f}%)")

# Use same validation split as training (20%, seed 42)
from sklearn.model_selection import train_test_split
single_label_df = single_label_df.reset_index(drop=True)
_, val_idx = train_test_split(
    range(len(single_label_df)),
    test_size=0.2,
    random_state=42,
    stratify=single_label_df['label']
)

val_df = single_label_df.iloc[val_idx].copy()
print(f"Validation samples: {len(val_df)}")

# Get feature columns
vggish_cols = [f'emb_{i}' for i in range(128)]
tabular_cols = [col for col in tabular_train.columns if col != 'fname']
feature_cols = vggish_cols + tabular_cols

print(f"\nFeatures: {len(feature_cols)} total")
print(f"  VGGish: {len(vggish_cols)}")
print(f"  Tabular: {len(tabular_cols)}")

# Make predictions
print("\n🔮 Making predictions on validation set...")

# Make predictions row by row
y_pred = []
y_true = []

print(f"Processing {len(val_df)} samples...")
for i in range(len(val_df)):
    if i % 100 == 0:
        print(f"  {i}/{len(val_df)}...")

    row = val_df.iloc[i]

    # Get features
    features = row[feature_cols].values

    # Create a single-row DataFrame for prediction
    test_row = pd.DataFrame([features], columns=feature_cols)

    # Predict
    pred_class, pred_idx, pred_probs = learn.predict(test_row.iloc[0])

    y_pred.append(str(pred_class))
    y_true.append(row['label'])

print(f"✓ Predictions complete")

# Get class names (vocab)
vocab = learn.dls.vocab

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\n🎯 Overall Accuracy: {accuracy*100:.2f}%")

# Build confusion analysis
print("\n📊 Analyzing class confusions...")

# Dictionary to store confusion patterns
confusion_data = defaultdict(lambda: {'correct': 0, 'total': 0, 'confused_with': Counter()})

for true_label, pred_label in zip(y_true, y_pred):
    confusion_data[true_label]['total'] += 1
    if true_label == pred_label:
        confusion_data[true_label]['correct'] += 1
    else:
        confusion_data[true_label]['confused_with'][pred_label] += 1

# Create detailed results
results = []
for class_name in sorted(confusion_data.keys()):
    data = confusion_data[class_name]
    total = data['total']
    correct = data['correct']
    accuracy = correct / total if total > 0 else 0

    # Get top 3 confused classes
    top_confusions = data['confused_with'].most_common(3)

    result = {
        'Class': class_name,
        'Total_Samples': total,
        'Correct': correct,
        'Incorrect': total - correct,
        'Accuracy': accuracy,
    }

    # Add top confusions
    for i, (confused_class, count) in enumerate(top_confusions, 1):
        result[f'Confused_With_{i}'] = confused_class
        result[f'Confused_Count_{i}'] = count

    # Fill empty confusion slots
    for i in range(len(top_confusions) + 1, 4):
        result[f'Confused_With_{i}'] = ''
        result[f'Confused_Count_{i}'] = 0

    results.append(result)

# Create DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

# Save results
output_file = 'claude/lazypredict/findings/vggish_tabular_confusion_analysis.csv'
results_df.to_csv(output_file, index=False)
print(f"✓ Results saved to: {output_file}")

# Analysis: Most Confused Classes
print("\n" + "="*80)
print("MOST CONFUSED CLASSES (High False Negatives)")
print("="*80)
print("\nClasses that are MOST OFTEN MISCLASSIFIED:\n")

worst_classes = results_df[results_df['Incorrect'] > 0].nlargest(15, 'Incorrect')
for _, row in worst_classes.iterrows():
    acc = row['Accuracy'] * 100
    total = row['Total_Samples']
    incorrect = row['Incorrect']

    confusions = []
    for i in range(1, 4):
        if row[f'Confused_Count_{i}'] > 0:
            confusions.append(f"{row[f'Confused_With_{i}']} ({row[f'Confused_Count_{i}']})")

    confusion_str = ", ".join(confusions) if confusions else "N/A"

    print(f"{row['Class']:40s} {acc:5.1f}% ({incorrect}/{total} errors)")
    print(f"  → Confused with: {confusion_str}\n")

# Analysis: Most Mistaken For (False Positives)
print("\n" + "="*80)
print("CLASSES MOST OFTEN PREDICTED INCORRECTLY (High False Positives)")
print("="*80)
print("\nClasses that other sounds are MISTAKEN FOR:\n")

# Count how many times each class is predicted incorrectly
false_positive_counter = Counter()
for true_label, pred_label in zip(y_true, y_pred):
    if true_label != pred_label:
        false_positive_counter[pred_label] += 1

most_fp = false_positive_counter.most_common(15)
for class_name, fp_count in most_fp:
    # Find which classes are confused AS this class
    confused_from = []
    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label == class_name and true_label != class_name:
            confused_from.append(true_label)

    from_counter = Counter(confused_from).most_common(3)
    from_str = ", ".join([f"{c} ({cnt})" for c, cnt in from_counter])

    print(f"{class_name:40s} {fp_count:3d} false positives")
    print(f"  ← Mistaken from: {from_str}\n")

# Perfect classes
perfect = results_df[results_df['Accuracy'] == 1.0]
print("\n" + "="*80)
print(f"PERFECT CLASSES ({len(perfect)}/74): 100% Accuracy")
print("="*80)
for _, row in perfect.iterrows():
    print(f"  ✓ {row['Class']:40s} ({row['Total_Samples']} samples)")

# Failed classes
failed = results_df[results_df['Accuracy'] == 0.0]
print("\n" + "="*80)
print(f"FAILED CLASSES ({len(failed)}/74): 0% Accuracy")
print("="*80)
for _, row in failed.iterrows():
    confusions = []
    for i in range(1, 4):
        if row[f'Confused_Count_{i}'] > 0:
            confusions.append(f"{row[f'Confused_With_{i}']} ({row[f'Confused_Count_{i}']})")
    confusion_str = ", ".join(confusions) if confusions else "N/A"

    print(f"  ✗ {row['Class']:40s} ({row['Total_Samples']} samples)")
    print(f"    → All mistaken as: {confusion_str}")

# Sample size correlation
print("\n" + "="*80)
print("SAMPLE SIZE ANALYSIS")
print("="*80)

bins = [0, 5, 10, 15, 20, 100]
labels = ['1-4', '5-9', '10-14', '15-19', '20+']
results_df['sample_range'] = pd.cut(results_df['Total_Samples'], bins=bins, labels=labels)

for label in labels:
    subset = results_df[results_df['sample_range'] == label]
    if len(subset) > 0:
        avg_acc = subset['Accuracy'].mean() * 100
        avg_errors = subset['Incorrect'].mean()
        print(f"{label:10s}: {len(subset):2d} classes | Avg Acc: {avg_acc:5.1f}% | Avg Errors: {avg_errors:.1f}")

corr = results_df['Total_Samples'].corr(results_df['Accuracy'])
print(f"\nCorrelation (samples vs accuracy): {corr:.3f}")

# Acoustic similarity patterns
print("\n" + "="*80)
print("ACOUSTIC SIMILARITY PATTERNS")
print("="*80)

# Group confused pairs by acoustic category
acoustic_groups = {
    'Percussion/Transients': ['Knock', 'Finger_snapping', 'Hi-hat', 'Bass_drum', 'Clapping'],
    'Strings/Plucked': ['Acoustic_guitar', 'Bass_guitar', 'Electric_guitar', 'Banjo', 'Ukulele'],
    'Human/Vocal': ['Whispering', 'Breathing', 'Sigh', 'Gasp', 'Male_speech_and_man_speaking'],
    'Metallic': ['Keys_jangling', 'Scissors', 'Cutlery_and_silverware', 'Bicycle_bell'],
    'Ambient/Texture': ['Hiss', 'Crackle', 'Waves_and_surf', 'Raindrop', 'Trickle_and_dribble'],
}

print("\nChecking for within-group confusions (acoustically similar sounds):\n")

for group_name, classes in acoustic_groups.items():
    within_group_confusions = 0
    examples = []

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in classes and pred_label in classes and true_label != pred_label:
            within_group_confusions += 1
            examples.append(f"{true_label} → {pred_label}")

    if within_group_confusions > 0:
        print(f"{group_name}: {within_group_confusions} within-group confusions")
        # Show top 3 examples
        example_counter = Counter(examples).most_common(3)
        for ex, count in example_counter:
            print(f"  • {ex} ({count}x)")
        print()

print("\n✅ Analysis complete!")
print(f"\nSummary:")
print(f"  Overall Accuracy: {accuracy*100:.2f}%")
print(f"  Perfect Classes: {len(perfect)}")
print(f"  Failed Classes: {len(failed)}")
print(f"  Most confused class: {worst_classes.iloc[0]['Class']} ({worst_classes.iloc[0]['Incorrect']} errors)")
