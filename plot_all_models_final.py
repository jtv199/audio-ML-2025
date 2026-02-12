#!/usr/bin/env python3
"""
Final All Models Comparison Plot
With corrected VGGish+Tabular per-class data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read all datasets
df_svc = pd.read_csv('claude/lazypredict/findings/error_analysis_linearsvc.csv')
df_mob = pd.read_csv('claude/lazypredict/findings/error_analysis_mobilenet.csv')
df_vgg = pd.read_csv('claude/lazypredict/findings/error_analysis_vggish_tabular.csv')

# Merge all three
df = df_svc[['Class', 'Total_Samples', 'Accuracy']].merge(
    df_mob[['Class', 'Accuracy']],
    on='Class',
    suffixes=('_svc', '_mob')
).merge(
    df_vgg[['Class', 'Accuracy']],
    on='Class'
)

df.rename(columns={'Accuracy': 'Accuracy_vgg'}, inplace=True)

# Convert to percentage
df['Accuracy_svc'] = df['Accuracy_svc'] * 100
df['Accuracy_mob'] = df['Accuracy_mob'] * 100
df['Accuracy_vgg'] = df['Accuracy_vgg'] * 100

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))

# Plot individual points (light, in background)
ax.scatter(df['Total_Samples'], df['Accuracy_svc'],
          s=80, alpha=0.3, c='#FF6B6B', edgecolors='none', zorder=1)
ax.scatter(df['Total_Samples'], df['Accuracy_mob'],
          s=80, alpha=0.3, c='#4ECDC4', edgecolors='none', zorder=1)
ax.scatter(df['Total_Samples'], df['Accuracy_vgg'],
          s=80, alpha=0.3, c='#FFE66D', edgecolors='none', zorder=1)

# Calculate and plot trend lines
x_range = np.linspace(df['Total_Samples'].min(), df['Total_Samples'].max(), 100)

# LinearSVC trend line
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

# VGGish+Tabular trend line
z_vgg = np.polyfit(df['Total_Samples'], df['Accuracy_vgg'], 2)
p_vgg = np.poly1d(z_vgg)
ax.plot(x_range, p_vgg(x_range),
       color='#FFE66D', linestyle='-', linewidth=4,
       label=f'VGGish+Tabular (Mean: {df["Accuracy_vgg"].mean():.1f}%) 🏆', zorder=3, alpha=0.9)

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
🥇 VGGish+Tab: {df['Accuracy_vgg'].mean():.1f}% ± {df['Accuracy_vgg'].std():.1f}%
🥈 MobileNet: {df['Accuracy_mob'].mean():.1f}% ± {df['Accuracy_mob'].std():.1f}%
🥉 LinearSVC: {df['Accuracy_svc'].mean():.1f}% ± {df['Accuracy_svc'].std():.1f}%

Best: VGGish+Tabular (+{df['Accuracy_vgg'].mean() - df['Accuracy_mob'].mean():.1f}% vs MobileNet)"""

ax.text(0.02, 0.98, stats_text,
       transform=ax.transAxes,
       fontsize=12,
       verticalalignment='top',
       fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=2))

plt.tight_layout()

# Save
output_file = 'claude/lazypredict/findings/all_models_comparison_final.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Final comparison plot saved to: {output_file}")

# Print comparative statistics
print("\n" + "="*80)
print("FINAL MODEL COMPARISON SUMMARY")
print("="*80)

print("\nOverall Performance:")
print(f"  🥇 VGGish+Tabular:  {df['Accuracy_vgg'].mean():.2f}% ± {df['Accuracy_vgg'].std():.2f}%")
print(f"  🥈 MobileNetV4:     {df['Accuracy_mob'].mean():.2f}% ± {df['Accuracy_mob'].std():.2f}%")
print(f"  🥉 LinearSVC:       {df['Accuracy_svc'].mean():.2f}% ± {df['Accuracy_svc'].std():.2f}%")

print(f"\nImprovement over MobileNet:")
print(f"  VGGish+Tabular: +{df['Accuracy_vgg'].mean() - df['Accuracy_mob'].mean():.2f}%")

print(f"\nImprovement over LinearSVC:")
print(f"  VGGish+Tabular: +{df['Accuracy_vgg'].mean() - df['Accuracy_svc'].mean():.2f}%")
print(f"  MobileNetV4:    +{df['Accuracy_mob'].mean() - df['Accuracy_svc'].mean():.2f}%")

print(f"\nCorrelation with Sample Size:")
print(f"  VGGish+Tabular: {df['Total_Samples'].corr(df['Accuracy_vgg']):.3f}")
print(f"  MobileNet:      {df['Total_Samples'].corr(df['Accuracy_mob']):.3f}")
print(f"  LinearSVC:      {df['Total_Samples'].corr(df['Accuracy_svc']):.3f}")

print(f"\nConsistency (Lower std is better):")
print(f"  VGGish+Tabular: {df['Accuracy_vgg'].std():.2f}% 🥇")
print(f"  MobileNetV4:    {df['Accuracy_mob'].std():.2f}%")
print(f"  LinearSVC:      {df['Accuracy_svc'].std():.2f}%")

# Count perfect classes
perfect_vgg = len(df[df['Accuracy_vgg'] >= 100])
perfect_mob = len(df[df['Accuracy_mob'] >= 100])
perfect_svc = len(df[df['Accuracy_svc'] >= 100])

print(f"\nPerfect Classes (100% accuracy):")
print(f"  VGGish+Tabular: {perfect_vgg} classes 🏆")
print(f"  MobileNetV4:    {perfect_mob} classes")
print(f"  LinearSVC:      {perfect_svc} classes")

print("\n✓ Analysis complete!")
