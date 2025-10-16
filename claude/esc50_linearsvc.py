#!/usr/bin/env python3
"""
ESC-50 LinearSVC Training Script
Trains LinearSVC on ESC-50 features and compares with FreeSound results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ESC-50 LINEAR SVC TRAINING")
print("=" * 80)
print()

# Load features
FEATURES_PATH = Path('data/esc50/esc50_features.csv')
print("Loading features...")
df = pd.read_csv(FEATURES_PATH)
print(f"  Dataset shape: {df.shape}")
print(f"  Number of samples: {len(df)}")
print(f"  Number of categories: {df['category'].nunique()}")
print()

# Prepare data using fold-based split (ESC-50 has 5 folds for cross-validation)
# We'll use fold 5 as test set, folds 1-4 as train set
print("Preparing train/test split...")
print("  Using ESC-50 fold structure:")
print("    Train: Folds 1-4")
print("    Test: Fold 5")
print()

train_df = df[df['fold'] != 5]
test_df = df[df['fold'] == 5]

# Extract features and labels
meta_cols = ['filename', 'fold', 'target', 'category', 'esc10']
feature_cols = [col for col in df.columns if col not in meta_cols]

X_train = train_df[feature_cols].values
y_train = train_df['category'].values
X_test = test_df[feature_cols].values
y_test = test_df['category'].values

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Number of features: {len(feature_cols)}")
print()

# Train LinearSVC (using same configuration as FreeSound analysis)
print("Training LinearSVC...")
print("  Configuration: LinearSVC(random_state=42, max_iter=1000, dual=False)")
print()

model = LinearSVC(random_state=42, max_iter=1000, dual=False)
model.fit(X_train, y_train)

print("✓ Training complete!")
print()

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)
print("✓ Done!")
print()

# Calculate metrics
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Overall Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Balanced Accuracy:        {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
print(f"F1 Score (weighted):      {f1:.4f}")
print()

# Per-class analysis
print("=" * 80)
print("PER-CLASS PERFORMANCE")
print("=" * 80)
print()

classes = sorted(df['category'].unique())
results = []

for cls in classes:
    cls_indices = y_test == cls
    if cls_indices.sum() == 0:
        continue

    tp = ((y_test == cls) & (y_pred == cls)).sum()
    fp = ((y_test != cls) & (y_pred == cls)).sum()
    fn = ((y_test == cls) & (y_pred != cls)).sum()

    total_actual = cls_indices.sum()
    correct = (y_test[cls_indices] == y_pred[cls_indices]).sum()
    class_accuracy = correct / total_actual if total_actual > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_class = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results.append({
        'Class': cls,
        'Samples': total_actual,
        'Correct': correct,
        'Accuracy': class_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_class
    })

df_results = pd.DataFrame(results)

# Display top and bottom performers
print("Top 10 Most Accurate Classes:")
print("-" * 80)
top_10 = df_results.nlargest(10, 'Accuracy')[['Class', 'Accuracy', 'Samples', 'Precision', 'Recall']]
print(top_10.to_string(index=False))
print()

print("Bottom 10 Least Accurate Classes:")
print("-" * 80)
bottom_10 = df_results.nsmallest(10, 'Accuracy')[['Class', 'Accuracy', 'Samples', 'Precision', 'Recall']]
print(bottom_10.to_string(index=False))
print()

# Summary statistics
print("=" * 80)
print("STATISTICS")
print("=" * 80)
print()
print(f"Mean per-class accuracy:    {df_results['Accuracy'].mean():.4f}")
print(f"Median per-class accuracy:  {df_results['Accuracy'].median():.4f}")
print(f"Std dev of accuracy:        {df_results['Accuracy'].std():.4f}")
print(f"Min accuracy:               {df_results['Accuracy'].min():.4f} ({df_results['Accuracy'].idxmin()})")
print(f"Max accuracy:               {df_results['Accuracy'].max():.4f} ({df_results['Accuracy'].idxmax()})")
print()

# Comparison with FreeSound
print("=" * 80)
print("COMPARISON: ESC-50 vs FreeSound")
print("=" * 80)
print()
print("ESC-50 Dataset:")
print(f"  Overall Accuracy:         {accuracy*100:.2f}%")
print(f"  Number of Classes:        50")
print(f"  Samples per Class:        40 (train: 32, test: 8)")
print(f"  Total Features:           2,474")
print()
print("FreeSound Dataset (from previous analysis):")
print(f"  Overall Accuracy:         56.56%")
print(f"  Number of Classes:        74")
print(f"  Samples (single-label):   4,268")
print(f"  Total Features:           2,474")
print()

if accuracy > 0.5656:
    diff = (accuracy - 0.5656) * 100
    print(f"✓ ESC-50 performs BETTER by {diff:.2f} percentage points!")
else:
    diff = (0.5656 - accuracy) * 100
    print(f"✗ ESC-50 performs WORSE by {diff:.2f} percentage points")

print()

# Save results
output_dir = Path('claude/lazypredict/findings')
output_dir.mkdir(parents=True, exist_ok=True)

results_file = output_dir / 'esc50_linearsvc_results.csv'
df_results.to_csv(results_file, index=False)
print(f"Per-class results saved to: {results_file}")

# Save overall metrics
summary = {
    'Dataset': 'ESC-50',
    'Model': 'LinearSVC',
    'Overall_Accuracy': accuracy,
    'Balanced_Accuracy': balanced_acc,
    'F1_Score': f1,
    'Num_Classes': 50,
    'Num_Features': len(feature_cols),
    'Train_Samples': len(X_train),
    'Test_Samples': len(X_test)
}

summary_file = output_dir / 'esc50_summary.csv'
pd.DataFrame([summary]).to_csv(summary_file, index=False)
print(f"Summary saved to: {summary_file}")
print()

print("=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
