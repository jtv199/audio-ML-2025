#!/usr/bin/env python3
"""
All Models Comparison Plot
Sample Size vs Accuracy - All models on same graph with trend lines
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read datasets
df_svc = pd.read_csv('claude/lazypredict/findings/error_analysis_linearsvc.csv')
df_mob = pd.read_csv('claude/lazypredict/findings/error_analysis_mobilenet.csv')

# For VGGish, we'll use the overall accuracy and create a synthetic comparison
# Since we don't have per-class data, we'll show it as a horizontal line
vggish_accuracy = 68.46  # From the results file

# Merge SVC and MobileNet
df = df_svc[['Class', 'Total_Samples', 'Accuracy']].merge(
    df_mob[['Class', 'Accuracy']],
    on='Class',
    suffixes=('_svc', '_mob')
)

# Convert to percentage
df['Accuracy_svc'] = df['Accuracy_svc'] * 100
df['Accuracy_mob'] = df['Accuracy_mob'] * 100

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))

# Plot individual points (light, in background)
ax.scatter(df['Total_Samples'], df['Accuracy_svc'],
          s=80, alpha=0.3, c='#FF6B6B', edgecolors='none',
          label=None, zorder=1)

ax.scatter(df['Total_Samples'], df['Accuracy_mob'],
          s=80, alpha=0.3, c='#4ECDC4', edgecolors='none',
          label=None, zorder=1)

# Calculate and plot trend lines
x_range = np.linspace(df['Total_Samples'].min(), df['Total_Samples'].max(), 100)

# LinearSVC trend line (polynomial fit)
z_svc = np.polyfit(df['Total_Samples'], df['Accuracy_svc'], 2)
p_svc = np.poly1d(z_svc)
ax.plot(x_range, p_svc(x_range),
       color='#FF6B6B', linestyle='-', linewidth=4,
       label=f'LinearSVC (Mean: {df["Accuracy_svc"].mean():.1f}%)', zorder=3, alpha=0.9)

# MobileNet trend line
z_mob = np.polyfit(df['Total_Samples'], df['Accuracy_mob'], 2)
p_mob = np.poly1d(z_mob)
ax.plot(x_range, p_mob(x_range),
       color='#4ECDC4', linestyle='-', linewidth=4,
       label=f'MobileNetV4 (Mean: {df["Accuracy_mob"].mean():.1f}%)', zorder=3, alpha=0.9)

# VGGish+Tabular - horizontal line (overall accuracy)
ax.axhline(y=vggish_accuracy, color='#FFE66D', linestyle='-', linewidth=4,
          label=f'VGGish+Tabular (Mean: {vggish_accuracy:.1f}%)', zorder=3, alpha=0.9)

# Add reference lines
ax.axvline(x=15, color='blue', linestyle='--', alpha=0.4, linewidth=2,
          label='15-sample threshold', zorder=2)
ax.axhline(y=90, color='green', linestyle=':', alpha=0.2, linewidth=1.5, zorder=0)
ax.axhline(y=70, color='yellowgreen', linestyle=':', alpha=0.2, linewidth=1.5, zorder=0)
ax.axhline(y=50, color='orange', linestyle=':', alpha=0.2, linewidth=1.5, zorder=0)

# Labels and title
ax.set_xlabel('Number of Test Samples per Class', fontsize=15, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=15, fontweight='bold')
ax.set_title('Model Comparison: Sample Size vs Accuracy\nFreeSound Audio Classification (74 Classes)',
            fontsize=18, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', zorder=0)

# Legend
ax.legend(loc='lower right', fontsize=13, framealpha=0.95, shadow=True)

# Set limits
ax.set_xlim(0, df['Total_Samples'].max() + 2)
ax.set_ylim(-5, 105)

# Add statistics box
stats_text = f"""Model Performance:
LinearSVC: {df['Accuracy_svc'].mean():.1f}% ± {df['Accuracy_svc'].std():.1f}%
VGGish+Tab: {vggish_accuracy:.1f}%
MobileNet: {df['Accuracy_mob'].mean():.1f}% ± {df['Accuracy_mob'].std():.1f}%

Best: MobileNetV4 (+{df['Accuracy_mob'].mean() - df['Accuracy_svc'].mean():.1f}% vs LinearSVC)"""

ax.text(0.02, 0.98, stats_text,
       transform=ax.transAxes,
       fontsize=12,
       verticalalignment='top',
       fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='black', linewidth=2))

plt.tight_layout()

# Save
output_file = 'claude/lazypredict/findings/all_models_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved to: {output_file}")

# Print comparative statistics
print("\n" + "="*80)
print("ALL MODELS COMPARISON SUMMARY")
print("="*80)

print("\nOverall Performance:")
print(f"  1. MobileNetV4:     {df['Accuracy_mob'].mean():.2f}% ± {df['Accuracy_mob'].std():.2f}% 🥇")
print(f"  2. VGGish+Tabular:  {vggish_accuracy:.2f}%")
print(f"  3. LinearSVC:       {df['Accuracy_svc'].mean():.2f}% ± {df['Accuracy_svc'].std():.2f}%")

print(f"\nImprovement over LinearSVC:")
print(f"  MobileNetV4:    +{df['Accuracy_mob'].mean() - df['Accuracy_svc'].mean():.2f}%")
print(f"  VGGish+Tabular: +{vggish_accuracy - df['Accuracy_svc'].mean():.2f}%")

print(f"\nCorrelation with Sample Size:")
print(f"  LinearSVC:  {df['Total_Samples'].corr(df['Accuracy_svc']):.3f} (moderate)")
print(f"  MobileNet:  {df['Total_Samples'].corr(df['Accuracy_mob']):.3f} (weak)")

print(f"\nConsistency (Lower std is better):")
print(f"  MobileNetV4:  {df['Accuracy_mob'].std():.2f}% 🥇 Most consistent")
print(f"  LinearSVC:    {df['Accuracy_svc'].std():.2f}%")

# Performance by sample size
print("\n" + "="*80)
print("PERFORMANCE BY SAMPLE SIZE")
print("="*80)

bins = [0, 5, 10, 15, 100]
labels = ['1-4', '5-9', '10-14', '15+']
df['sample_range'] = pd.cut(df['Total_Samples'], bins=bins, labels=labels)

print(f"\n{'Range':<12} {'Classes':<10} {'LinearSVC':<15} {'MobileNet':<15} {'Improvement'}")
print("-" * 80)
for label in labels:
    subset = df[df['sample_range'] == label]
    if len(subset) > 0:
        svc_mean = subset['Accuracy_svc'].mean()
        mob_mean = subset['Accuracy_mob'].mean()
        improvement = mob_mean - svc_mean
        print(f"{label:<12} {len(subset):<10} {svc_mean:>6.1f}% ± {subset['Accuracy_svc'].std():4.1f}% "
              f"{mob_mean:>6.1f}% ± {subset['Accuracy_mob'].std():4.1f}%  {improvement:+6.1f}%")

# Best/Worst for each model
print("\n" + "="*80)
print("BEST CLASSES FOR EACH MODEL")
print("="*80)

print("\nLinearSVC Top 5:")
top_svc = df.nlargest(5, 'Accuracy_svc')
for _, row in top_svc.iterrows():
    print(f"  {row['Class']:40s} {row['Accuracy_svc']:5.1f}%")

print("\nMobileNet Top 5:")
top_mob = df.nlargest(5, 'Accuracy_mob')
for _, row in top_mob.iterrows():
    print(f"  {row['Class']:40s} {row['Accuracy_mob']:5.1f}%")

print("\n" + "="*80)
print("CLASSES WHERE LINEARSVC BEATS MOBILENET")
print("="*80)

df['svc_better'] = df['Accuracy_svc'] - df['Accuracy_mob']
svc_wins = df[df['svc_better'] > 0].nlargest(10, 'svc_better')

if len(svc_wins) > 0:
    for _, row in svc_wins.iterrows():
        print(f"  {row['Class']:40s} LinearSVC: {row['Accuracy_svc']:5.1f}% vs MobileNet: {row['Accuracy_mob']:5.1f}% ({row['svc_better']:+5.1f}%)")
else:
    print("  None! MobileNet wins or ties on all classes.")

print("\n✓ Analysis complete!")
