#!/usr/bin/env python3
"""
VGGish + Tabular 2-Layer NN Error Analysis
Load the trained model and perform comprehensive error analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
from fastai.tabular.all import *
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting VGGish+Tabular Error Analysis\n")

# Paths
WORK = Path('work')
MODEL_PATH = Path('models/vggish_tabular_2layers_20251016_182328.pkl')

# Check if model exists
if not MODEL_PATH.exists():
    print(f"❌ Model not found at {MODEL_PATH}")
    alternatives = list(Path('models').glob('vggish_tabular*.pkl'))
    if alternatives:
        MODEL_PATH = sorted(alternatives)[-1]  # Get latest
        print(f"Using: {MODEL_PATH}")
    else:
        raise FileNotFoundError("No VGGish tabular model files found!")

print(f"✓ Model found: {MODEL_PATH}")

# Load embeddings and features
print("\n📊 Loading data...")
vggish_train = pd.read_csv(WORK/'tokenized/vggish_embeddings_train_curated.csv')
tabular_train = pd.read_csv(WORK/'trn_curated_feature.csv')

# Merge
train_df = vggish_train.merge(tabular_train, on='fname', how='inner')
print(f"Total samples: {len(train_df)}")

# Filter single-label
train_df['label_count'] = train_df['labels'].str.count(',') + 1
df_single = train_df[train_df['label_count'] == 1].copy()
df_single['label'] = df_single['labels']

print(f"Single-label samples: {len(df_single)} ({len(df_single)/len(train_df)*100:.1f}%)")
print(f"Unique classes: {df_single['label'].nunique()}")

# Load model
print("\n🔄 Loading trained model...")
try:
    learn = load_learner(MODEL_PATH)
    print(f"✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Prepare validation dataset (20% split, same seed)
print("\n📝 Preparing validation set...")
np.random.seed(42)
val_indices = np.random.choice(len(df_single), size=int(0.2 * len(df_single)), replace=False)
val_df = df_single.iloc[val_indices].reset_index(drop=True)
print(f"Validation samples: {len(val_df)}")

# Make predictions
print("\n🔮 Making predictions on validation set...")
y_true = []
y_pred = []

for idx, row in val_df.iterrows():
    if idx % 100 == 0:
        print(f"  Processing {idx}/{len(val_df)}...")

    try:
        # Get the row data (excluding metadata columns)
        # The learner expects the feature columns only
        pred_class, pred_idx, pred_probs = learn.predict(row)

        y_true.append(row['label'])
        y_pred.append(str(pred_class))

    except Exception as e:
        print(f"    Error processing row {idx}: {e}")
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
output_file = 'claude/lazypredict/findings/error_analysis_vggish_tabular.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

# Print summary statistics
print("\n" + "="*70)
print("VGGISH+TABULAR ERROR ANALYSIS SUMMARY")
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

# Compare with LinearSVC and MobileNet
print("\n" + "="*70)
print("COMPARISON WITH OTHER MODELS")
print("="*70)

try:
    linearsvc_df = pd.read_csv('claude/lazypredict/findings/error_analysis_linearsvc.csv')
    mobilenet_df = pd.read_csv('claude/lazypredict/findings/error_analysis_mobilenet.csv')

    # Merge results
    comparison = results_df[['Class', 'Accuracy']].merge(
        linearsvc_df[['Class', 'Accuracy']],
        on='Class',
        suffixes=('_vggish', '_linearsvc'),
        how='inner'
    ).merge(
        mobilenet_df[['Class', 'Accuracy']],
        on='Class',
        how='inner'
    )
    comparison.rename(columns={'Accuracy': 'Accuracy_mobilenet'}, inplace=True)

    comparison['Improvement_vs_LinearSVC'] = (comparison['Accuracy_vggish'] - comparison['Accuracy_linearsvc']) * 100
    comparison['Improvement_vs_MobileNet'] = (comparison['Accuracy_vggish'] - comparison['Accuracy_mobilenet']) * 100

    print(f"\nVGGish+Tabular Overall: {accuracy*100:.2f}%")
    print(f"LinearSVC Overall: {linearsvc_df['Accuracy'].mean()*100:.2f}%")
    print(f"MobileNet Overall: {mobilenet_df['Accuracy'].mean()*100:.2f}%")

    print("\n📈 Top 10 improvements vs LinearSVC:")
    comp_svc = comparison.nlargest(10, 'Improvement_vs_LinearSVC')
    for _, row in comp_svc.iterrows():
        print(f"  {row['Class']:40s} {row['Improvement_vs_LinearSVC']:+6.1f}% "
              f"({row['Accuracy_vggish']*100:.1f}% vs {row['Accuracy_linearsvc']*100:.1f}%)")

    print("\n📊 Comparison with MobileNet (Top 10 better):")
    comp_mob = comparison.nlargest(10, 'Improvement_vs_MobileNet')
    for _, row in comp_mob.iterrows():
        print(f"  {row['Class']:40s} {row['Improvement_vs_MobileNet']:+6.1f}% "
              f"({row['Accuracy_vggish']*100:.1f}% vs {row['Accuracy_mobilenet']*100:.1f}%)")

    # Save comparison
    comparison.to_csv('claude/lazypredict/findings/vggish_vs_all_comparison.csv', index=False)
    print(f"\n✓ Comparison saved to: claude/lazypredict/findings/vggish_vs_all_comparison.csv")

except Exception as e:
    print(f"  Could not compare with other models: {e}")

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
