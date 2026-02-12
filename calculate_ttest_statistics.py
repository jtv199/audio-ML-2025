#!/usr/bin/env python3
"""
Calculate t-test statistics comparing models:
1. All models vs DeiT Tiny (baseline)
2. VGGish solo vs VGGish + Tabular
"""

import pandas as pd
import numpy as np
from scipy import stats

# Read the data files
all_models_df = pd.read_csv('all_4_models_per_class_comparison.csv')
deit_df = pd.read_csv('claude/lazypredict/findings/error_analysis_deit_tiny.csv')

# Merge dataframes on Class
merged_df = pd.merge(
    all_models_df,
    deit_df[['Class', 'Accuracy']],
    on='Class',
    how='inner'
)

# Rename columns for clarity
merged_df.rename(columns={
    'Accuracy_svc': 'LinearSVC',
    'Accuracy_mobilenet': 'MobileNetV4',
    'Accuracy_vggish_solo': 'VGGish_Solo',
    'Accuracy_vggish_tab': 'VGGish_Tabular',
    'Accuracy': 'DeiT_Tiny'
}, inplace=True)

print(f"Total classes matched: {len(merged_df)}")
print(f"\nClass overlap check:")
print(f"  All models CSV: {len(all_models_df)} classes")
print(f"  DeiT CSV: {len(deit_df)} classes")
print(f"  Merged: {len(merged_df)} classes")

# Convert percentages to proportions (0-1 scale) for LinearSVC, MobileNetV4, VGGish models
# DeiT is already in 0-1 scale
for col in ['LinearSVC', 'MobileNetV4', 'VGGish_Solo', 'VGGish_Tabular']:
    merged_df[col] = merged_df[col] / 100.0

# Multiply DeiT by 100 for display consistency
merged_df['DeiT_Tiny_pct'] = merged_df['DeiT_Tiny'] * 100

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS (in %)")
print("="*80)

models = ['LinearSVC', 'MobileNetV4', 'VGGish_Solo', 'VGGish_Tabular', 'DeiT_Tiny']
stats_data = []

for model in models:
    values = merged_df[model] * 100  # Convert to percentage for display
    stats_data.append({
        'Model': model,
        'Mean': values.mean(),
        'Std Dev': values.std(),
        'Min': values.min(),
        'Max': values.max(),
        'Median': values.median()
    })

stats_df = pd.DataFrame(stats_data)
print(stats_df.to_string(index=False))

print("\n" + "="*80)
print("T-TEST RESULTS: ALL MODELS vs DEIT TINY (BASELINE)")
print("="*80)
print("H0: Model performance = DeiT Tiny performance")
print("Ha: Model performance ≠ DeiT Tiny performance")
print()

ttest_results = []

for model in ['LinearSVC', 'MobileNetV4', 'VGGish_Solo', 'VGGish_Tabular']:
    # Paired t-test (same classes)
    t_stat, p_value = stats.ttest_rel(merged_df[model], merged_df['DeiT_Tiny'])

    # Calculate mean difference
    mean_diff = (merged_df[model].mean() - merged_df['DeiT_Tiny'].mean()) * 100

    # Effect size (Cohen's d for paired samples)
    diff = merged_df[model] - merged_df['DeiT_Tiny']
    cohens_d = diff.mean() / diff.std()

    # Significance level
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "ns"

    ttest_results.append({
        'Model': model,
        'Mean Diff (%)': mean_diff,
        't-statistic': t_stat,
        'p-value': p_value,
        'Cohen\'s d': cohens_d,
        'Significance': sig
    })

    print(f"{model:20} vs DeiT Tiny:")
    print(f"  Mean difference: {mean_diff:+6.2f}% (percentage points)")
    print(f"  t-statistic:     {t_stat:7.3f}")
    print(f"  p-value:         {p_value:.2e}  {sig}")
    print(f"  Cohen's d:       {cohens_d:7.3f} (effect size)")

    if p_value < 0.05:
        direction = "better" if mean_diff > 0 else "worse"
        print(f"  Result: {model} performs significantly {direction} than DeiT Tiny")
    else:
        print(f"  Result: No significant difference from DeiT Tiny")
    print()

# Save t-test results to CSV
ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_csv('model_vs_deit_ttest_results.csv', index=False)
print(f"Saved t-test results to: model_vs_deit_ttest_results.csv")

print("\n" + "="*80)
print("T-TEST RESULTS: VGGISH SOLO vs VGGISH + TABULAR")
print("="*80)
print("H0: VGGish Solo = VGGish + Tabular")
print("Ha: VGGish Solo ≠ VGGish + Tabular")
print()

# Paired t-test for VGGish models
t_stat, p_value = stats.ttest_rel(merged_df['VGGish_Solo'], merged_df['VGGish_Tabular'])

# Calculate mean difference
mean_diff = (merged_df['VGGish_Tabular'].mean() - merged_df['VGGish_Solo'].mean()) * 100

# Effect size (Cohen's d for paired samples)
diff = merged_df['VGGish_Tabular'] - merged_df['VGGish_Solo']
cohens_d = diff.mean() / diff.std()

# Significance level
if p_value < 0.001:
    sig = "***"
elif p_value < 0.01:
    sig = "**"
elif p_value < 0.05:
    sig = "*"
else:
    sig = "ns"

print(f"VGGish + Tabular vs VGGish Solo:")
print(f"  Mean difference: {mean_diff:+6.2f}% (percentage points)")
print(f"  t-statistic:     {t_stat:7.3f}")
print(f"  p-value:         {p_value:.2e}  {sig}")
print(f"  Cohen's d:       {cohens_d:7.3f} (effect size)")

if p_value < 0.05:
    direction = "better" if mean_diff > 0 else "worse"
    print(f"  Result: VGGish + Tabular performs significantly {direction} than VGGish Solo")
else:
    print(f"  Result: No significant difference between models")

# Count how many classes improved/worsened
improved = (merged_df['VGGish_Tabular'] > merged_df['VGGish_Solo']).sum()
worsened = (merged_df['VGGish_Tabular'] < merged_df['VGGish_Solo']).sum()
same = (merged_df['VGGish_Tabular'] == merged_df['VGGish_Solo']).sum()

print(f"\nPer-class comparison:")
print(f"  Classes improved with tabular features: {improved}")
print(f"  Classes worsened with tabular features: {worsened}")
print(f"  Classes unchanged: {same}")

# Save VGGish comparison
vggish_comparison = pd.DataFrame({
    'Model': ['VGGish Solo', 'VGGish + Tabular'],
    'Mean Accuracy (%)': [
        merged_df['VGGish_Solo'].mean() * 100,
        merged_df['VGGish_Tabular'].mean() * 100
    ],
    't-statistic': [np.nan, t_stat],
    'p-value': [np.nan, p_value],
    'Cohen\'s d': [np.nan, cohens_d],
    'Significance': ['baseline', sig]
})
vggish_comparison.to_csv('vggish_comparison_ttest.csv', index=False)
print(f"\nSaved VGGish comparison to: vggish_comparison_ttest.csv")

print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("Significance levels:")
print("  *** p < 0.001 (highly significant)")
print("  **  p < 0.01  (very significant)")
print("  *   p < 0.05  (significant)")
print("  ns  p ≥ 0.05  (not significant)")
print()
print("Cohen's d (effect size):")
print("  |d| < 0.2   : Small effect")
print("  |d| 0.2-0.5 : Medium effect")
print("  |d| 0.5-0.8 : Large effect")
print("  |d| > 0.8   : Very large effect")
print()
print("Note: Using paired t-tests (same classes compared across models)")
