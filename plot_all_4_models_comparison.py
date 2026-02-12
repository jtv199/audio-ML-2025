#!/usr/bin/env python3
"""
Plot sample size vs accuracy for all 4 models:
- LinearSVC (baseline)
- MobileNetV4 (CNN)
- VGGish Standalone (embeddings only)
- VGGish+Tabular (best model)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read all model results
print("Loading model results...")

# LinearSVC
svc_df = pd.read_csv('claude/lazypredict/findings/error_analysis_linearsvc.csv')
print(f"LinearSVC: {len(svc_df)} classes, Mean accuracy: {svc_df['Accuracy'].mean():.2%}")

# MobileNet
mobilenet_df = pd.read_csv('claude/lazypredict/findings/error_analysis_mobilenet.csv')
print(f"MobileNet: {len(mobilenet_df)} classes, Mean accuracy: {mobilenet_df['Accuracy'].mean():.2%}")

# VGGish Standalone
vggish_standalone_df = pd.read_csv('vggish_single_label_per_class_2layers_20251017_103236.csv')
print(f"VGGish Standalone: {len(vggish_standalone_df)} classes, Mean accuracy: {vggish_standalone_df['Accuracy'].mean():.2%}")

# VGGish+Tabular
vggish_tabular_df = pd.read_csv('claude/lazypredict/findings/error_analysis_vggish_tabular.csv')
print(f"VGGish+Tabular: {len(vggish_tabular_df)} classes, Mean accuracy: {vggish_tabular_df['Accuracy'].mean():.2%}")

# Merge all dataframes on class name
df = svc_df[['Class', 'Total_Samples', 'Accuracy']].copy()
df.columns = ['Class', 'Total_Samples', 'Accuracy_svc']

df = df.merge(
    mobilenet_df[['Class', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_mobilenet'}),
    on='Class',
    how='outer'
)

df = df.merge(
    vggish_standalone_df[['Class', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_vggish_solo'}),
    on='Class',
    how='outer'
)

df = df.merge(
    vggish_tabular_df[['Class', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_vggish_tab'}),
    on='Class',
    how='outer'
)

# Fill NaN with 0 for classes that might be missing
df = df.fillna(0)

# Convert to percentages
df['Accuracy_svc'] *= 100
df['Accuracy_mobilenet'] *= 100
df['Accuracy_vggish_solo'] *= 100
df['Accuracy_vggish_tab'] *= 100

print(f"\nMerged data: {len(df)} classes")

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))

# Define colors for each model
colors = {
    'LinearSVC': '#FF6B6B',           # Red
    'MobileNetV4': '#4ECDC4',         # Teal
    'VGGish Solo': '#95E1D3',         # Light teal
    'VGGish+Tab': '#FFE66D'           # Yellow/Gold (winner)
}

# Plot scatter points for each model
alpha_scatter = 0.4
size_scatter = 80

ax.scatter(df['Total_Samples'], df['Accuracy_svc'],
          s=size_scatter, alpha=alpha_scatter, c=colors['LinearSVC'],
          label='LinearSVC (52.78%)', edgecolors='white', linewidth=0.5)

ax.scatter(df['Total_Samples'], df['Accuracy_mobilenet'],
          s=size_scatter, alpha=alpha_scatter, c=colors['MobileNetV4'],
          label='MobileNetV4 (75.26%)', edgecolors='white', linewidth=0.5)

ax.scatter(df['Total_Samples'], df['Accuracy_vggish_solo'],
          s=size_scatter, alpha=alpha_scatter, c=colors['VGGish Solo'],
          label='VGGish Standalone (64.43%)', edgecolors='white', linewidth=0.5)

ax.scatter(df['Total_Samples'], df['Accuracy_vggish_tab'],
          s=size_scatter, alpha=alpha_scatter, c=colors['VGGish+Tab'],
          label='VGGish+Tabular (90.25%) 🏆', edgecolors='white', linewidth=0.5)

# Add trend lines (polynomial fit)
x_range = np.linspace(df['Total_Samples'].min(), df['Total_Samples'].max(), 100)

# LinearSVC trend
valid_svc = df['Accuracy_svc'] > 0
z_svc = np.polyfit(df.loc[valid_svc, 'Total_Samples'], df.loc[valid_svc, 'Accuracy_svc'], 2)
p_svc = np.poly1d(z_svc)
ax.plot(x_range, p_svc(x_range), color=colors['LinearSVC'], linewidth=3,
        alpha=0.8, linestyle='-')

# MobileNet trend
valid_mob = df['Accuracy_mobilenet'] > 0
z_mob = np.polyfit(df.loc[valid_mob, 'Total_Samples'], df.loc[valid_mob, 'Accuracy_mobilenet'], 2)
p_mob = np.poly1d(z_mob)
ax.plot(x_range, p_mob(x_range), color=colors['MobileNetV4'], linewidth=3,
        alpha=0.8, linestyle='-')

# VGGish Standalone trend
valid_vgg_solo = df['Accuracy_vggish_solo'] > 0
z_vgg_solo = np.polyfit(df.loc[valid_vgg_solo, 'Total_Samples'], df.loc[valid_vgg_solo, 'Accuracy_vggish_solo'], 2)
p_vgg_solo = np.poly1d(z_vgg_solo)
ax.plot(x_range, p_vgg_solo(x_range), color=colors['VGGish Solo'], linewidth=3,
        alpha=0.8, linestyle='-')

# VGGish+Tabular trend (winner - make it bold)
valid_vgg_tab = df['Accuracy_vggish_tab'] > 0
z_vgg_tab = np.polyfit(df.loc[valid_vgg_tab, 'Total_Samples'], df.loc[valid_vgg_tab, 'Accuracy_vggish_tab'], 2)
p_vgg_tab = np.poly1d(z_vgg_tab)
ax.plot(x_range, p_vgg_tab(x_range), color=colors['VGGish+Tab'], linewidth=4,
        alpha=0.9, linestyle='-')

# Calculate correlations
corr_svc = stats.spearmanr(df.loc[valid_svc, 'Total_Samples'], df.loc[valid_svc, 'Accuracy_svc'])[0]
corr_mob = stats.spearmanr(df.loc[valid_mob, 'Total_Samples'], df.loc[valid_mob, 'Accuracy_mobilenet'])[0]
corr_vgg_solo = stats.spearmanr(df.loc[valid_vgg_solo, 'Total_Samples'], df.loc[valid_vgg_solo, 'Accuracy_vggish_solo'])[0]
corr_vgg_tab = stats.spearmanr(df.loc[valid_vgg_tab, 'Total_Samples'], df.loc[valid_vgg_tab, 'Accuracy_vggish_tab'])[0]

print(f"\nCorrelations (Spearman):")
print(f"  LinearSVC:        {corr_svc:.3f}")
print(f"  MobileNetV4:      {corr_mob:.3f}")
print(f"  VGGish Solo:      {corr_vgg_solo:.3f}")
print(f"  VGGish+Tabular:   {corr_vgg_tab:.3f}")

# Add horizontal lines for mean accuracies
mean_svc = df.loc[valid_svc, 'Accuracy_svc'].mean()
mean_mob = df.loc[valid_mob, 'Accuracy_mobilenet'].mean()
mean_vgg_solo = df.loc[valid_vgg_solo, 'Accuracy_vggish_solo'].mean()
mean_vgg_tab = df.loc[valid_vgg_tab, 'Accuracy_vggish_tab'].mean()

ax.axhline(y=mean_svc, color=colors['LinearSVC'], linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=mean_mob, color=colors['MobileNetV4'], linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=mean_vgg_solo, color=colors['VGGish Solo'], linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=mean_vgg_tab, color=colors['VGGish+Tab'], linestyle='--', alpha=0.3, linewidth=2)

# Styling
ax.set_xlabel('Number of Training Samples per Class', fontsize=14, fontweight='bold')
ax.set_ylabel('Per-Class Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Comparison: Sample Size vs Accuracy (All 4 Models)\nFreeSound Audio Classification',
             fontsize=16, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim(-5, 105)
ax.set_xlim(0, df['Total_Samples'].max() + 5)

# Legend with correlations
legend_labels = [
    f'LinearSVC (52.78% avg, corr={corr_svc:.3f})',
    f'MobileNetV4 (75.26% avg, corr={corr_mob:.3f})',
    f'VGGish Standalone (64.43% avg, corr={corr_vgg_solo:.3f})',
    f'VGGish+Tabular (90.25% avg, corr={corr_vgg_tab:.3f}) 🏆'
]

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, legend_labels, fontsize=11, loc='lower right',
         framealpha=0.95, edgecolor='black', fancybox=True)

# Add text box with key insights
textstr = '\n'.join([
    'Key Insights:',
    '• VGGish+Tabular: Best overall (90.25%)',
    '• MobileNetV4: Best on few samples',
    '• VGGish Solo: Good baseline (64.43%)',
    '• LinearSVC: Baseline (52.78%)',
    '',
    'Sample Dependency:',
    f'• VGGish+Tab: {corr_vgg_tab:.3f} (strong)',
    f'• LinearSVC: {corr_svc:.3f} (moderate)',
    f'• MobileNet: {corr_mob:.3f} (weak)',
])

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()

# Save figure
output_file = 'all_4_models_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Also save the merged data
df.to_csv('all_4_models_per_class_comparison.csv', index=False)
print(f"✓ Data saved to: all_4_models_per_class_comparison.csv")

# Print summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")
print(f"\nModel Performance (Mean ± Std):")
print(f"  LinearSVC:        {mean_svc:.2f}% ± {df.loc[valid_svc, 'Accuracy_svc'].std():.2f}%")
print(f"  MobileNetV4:      {mean_mob:.2f}% ± {df.loc[valid_mob, 'Accuracy_mobilenet'].std():.2f}%")
print(f"  VGGish Solo:      {mean_vgg_solo:.2f}% ± {df.loc[valid_vgg_solo, 'Accuracy_vggish_solo'].std():.2f}%")
print(f"  VGGish+Tabular:   {mean_vgg_tab:.2f}% ± {df.loc[valid_vgg_tab, 'Accuracy_vggish_tab'].std():.2f}%")

print(f"\nPerfect Classes (100% accuracy):")
print(f"  LinearSVC:        {len(df[df['Accuracy_svc'] == 100])}")
print(f"  MobileNetV4:      {len(df[df['Accuracy_mobilenet'] == 100])}")
print(f"  VGGish Solo:      {len(df[df['Accuracy_vggish_solo'] == 100])}")
print(f"  VGGish+Tabular:   {len(df[df['Accuracy_vggish_tab'] == 100])}")

print(f"\nFailed Classes (0% accuracy):")
print(f"  LinearSVC:        {len(df[df['Accuracy_svc'] == 0])}")
print(f"  MobileNetV4:      {len(df[df['Accuracy_mobilenet'] == 0])}")
print(f"  VGGish Solo:      {len(df[df['Accuracy_vggish_solo'] == 0])}")
print(f"  VGGish+Tabular:   {len(df[df['Accuracy_vggish_tab'] == 0])}")

# plt.show()  # Comment out to prevent hanging
