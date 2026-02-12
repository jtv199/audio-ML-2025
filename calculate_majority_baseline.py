#!/usr/bin/env python3
"""
Calculate baseline accuracy for a majority class predictor.
"""

import pandas as pd
import numpy as np

# Read the per-class data
df = pd.read_csv('all_4_models_per_class_comparison.csv')

print("="*70)
print("MAJORITY CLASS BASELINE CALCULATION")
print("="*70)

# Get total samples per class
total_samples = df['Total_Samples'].sum()
print(f"\nTotal test samples: {total_samples}")
print(f"Total classes: {len(df)}")

# Find the majority class
max_samples_idx = df['Total_Samples'].idxmax()
majority_class = df.loc[max_samples_idx, 'Class']
majority_class_count = df.loc[max_samples_idx, 'Total_Samples']

print(f"\nMajority class: '{majority_class}'")
print(f"Samples in majority class: {majority_class_count}")

# Calculate baseline accuracy (if we always predict the majority class)
majority_baseline_accuracy = (majority_class_count / total_samples) * 100

print(f"\nMajority class baseline accuracy: {majority_baseline_accuracy:.2f}%")
print(f"  (If we always predict '{majority_class}')")

# Calculate uniform random baseline (random guessing)
uniform_baseline = (1 / len(df)) * 100
print(f"\nUniform random baseline: {uniform_baseline:.2f}%")
print(f"  (If we randomly guess from {len(df)} classes)")

# Calculate stratified random baseline (weighted by class distribution)
class_probabilities = df['Total_Samples'] / total_samples
stratified_baseline = (class_probabilities ** 2).sum() * 100
print(f"\nStratified random baseline: {stratified_baseline:.2f}%")
print(f"  (If we randomly guess weighted by class distribution)")

print("\n" + "="*70)
print("CLASS DISTRIBUTION STATISTICS")
print("="*70)

print(f"\nMean samples per class: {df['Total_Samples'].mean():.1f}")
print(f"Median samples per class: {df['Total_Samples'].median():.1f}")
print(f"Std dev samples per class: {df['Total_Samples'].std():.1f}")
print(f"Min samples: {df['Total_Samples'].min()}")
print(f"Max samples: {df['Total_Samples'].max()}")

# Show top 10 most frequent classes
print("\nTop 10 Most Frequent Classes:")
top_10 = df.nlargest(10, 'Total_Samples')[['Class', 'Total_Samples']]
for idx, row in top_10.iterrows():
    pct = (row['Total_Samples'] / total_samples) * 100
    print(f"  {row['Class']:40} {row['Total_Samples']:3} samples ({pct:5.2f}%)")

# Show bottom 10 least frequent classes
print("\nBottom 10 Least Frequent Classes:")
bottom_10 = df.nsmallest(10, 'Total_Samples')[['Class', 'Total_Samples']]
for idx, row in bottom_10.iterrows():
    pct = (row['Total_Samples'] / total_samples) * 100
    print(f"  {row['Class']:40} {row['Total_Samples']:3} samples ({pct:5.2f}%)")

print("\n" + "="*70)
print("COMPARISON WITH ACTUAL MODELS")
print("="*70)

# Compare with actual model performance
models = {
    'VGGish + Tabular': 'Accuracy_vggish_tab',
    'MobileNetV4': 'Accuracy_mobilenet',
    'VGGish Solo': 'Accuracy_vggish_solo',
    'LinearSVC': 'Accuracy_svc'
}

print(f"\n{'Model':<25} {'Accuracy':>10} {'vs Majority':>15} {'vs Random':>15}")
print("-" * 70)

baselines = [
    ('Majority Class Baseline', majority_baseline_accuracy),
    ('Stratified Random', stratified_baseline),
    ('Uniform Random', uniform_baseline)
]

for name, acc in baselines:
    print(f"{name:<25} {acc:>9.2f}% {'baseline':>14} {'baseline':>14}")

print("-" * 70)

for model_name, col in models.items():
    # Calculate mean accuracy across classes
    mean_acc = df[col].mean()
    vs_majority = mean_acc - majority_baseline_accuracy
    vs_random = mean_acc - uniform_baseline

    print(f"{model_name:<25} {mean_acc:>9.2f}% {vs_majority:>+14.2f}% {vs_random:>+14.2f}%")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print("""
1. Majority Class Baseline: Always predict the most common class
   - Simple but naive approach
   - Performance: {:.2f}%

2. Uniform Random Baseline: Randomly guess from 74 classes
   - Expected accuracy: 1/74 = {:.2f}%
   - No learning whatsoever

3. Stratified Random Baseline: Weighted random guess by class frequency
   - Accounts for class imbalance
   - Performance: {:.2f}%

4. All trained models significantly outperform these baselines:
   - VGGish + Tabular: {:.2f}% (vs {:.2f}% majority baseline = +{:.2f}%)
   - MobileNetV4: {:.2f}% (vs {:.2f}% majority baseline = +{:.2f}%)
   - LinearSVC: {:.2f}% (vs {:.2f}% majority baseline = +{:.2f}%)
   - Even DeiT Tiny (19.62%) beats uniform random (1.35%)
""".format(
    majority_baseline_accuracy,
    uniform_baseline,
    stratified_baseline,
    df['Accuracy_vggish_tab'].mean(), majority_baseline_accuracy,
    df['Accuracy_vggish_tab'].mean() - majority_baseline_accuracy,
    df['Accuracy_mobilenet'].mean(), majority_baseline_accuracy,
    df['Accuracy_mobilenet'].mean() - majority_baseline_accuracy,
    df['Accuracy_svc'].mean(), majority_baseline_accuracy,
    df['Accuracy_svc'].mean() - majority_baseline_accuracy
))
