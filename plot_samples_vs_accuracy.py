#!/usr/bin/env python3
"""
Plot: Number of Samples vs Accuracy per Class
Shows the relationship between data quantity and model performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the error analysis data
df = pd.read_csv('claude/lazypredict/findings/error_analysis_linearsvc.csv')

# Extract data
classes = df['Class']
samples = df['Total_Samples']
accuracy = df['Accuracy'] * 100  # Convert to percentage

# Create the plot
plt.figure(figsize=(14, 8))
scatter = plt.scatter(samples, accuracy,
                     s=100,  # Size of dots
                     alpha=0.6,
                     c=accuracy,  # Color by accuracy
                     cmap='RdYlGn',  # Red-Yellow-Green colormap
                     edgecolors='black',
                     linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontsize=12)

# Annotate interesting points
# Perfect classes
perfect = df[df['Accuracy'] == 1.0]
for _, row in perfect.iterrows():
    plt.annotate(row['Class'],
                xy=(row['Total_Samples'], row['Accuracy']*100),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Failed classes (0% accuracy)
failed = df[df['Accuracy'] == 0.0]
for _, row in failed.iterrows():
    plt.annotate(row['Class'],
                xy=(row['Total_Samples'], row['Accuracy']*100),
                xytext=(5, -5), textcoords='offset points',
                fontsize=7, color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

# Interesting outliers: High samples, low accuracy
low_acc_high_samples = df[(df['Accuracy'] < 0.2) & (df['Total_Samples'] >= 15)]
for _, row in low_acc_high_samples.iterrows():
    plt.annotate(row['Class'],
                xy=(row['Total_Samples'], row['Accuracy']*100),
                xytext=(5, 5), textcoords='offset points',
                fontsize=7, color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Add trend line
z = np.polyfit(samples, accuracy, 2)  # 2nd degree polynomial
p = np.poly1d(z)
x_trend = np.linspace(samples.min(), samples.max(), 100)
plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend (polynomial fit)')

# Add horizontal lines for performance tiers
plt.axhline(y=90, color='green', linestyle=':', alpha=0.3, linewidth=1.5, label='Excellent (90%)')
plt.axhline(y=70, color='yellowgreen', linestyle=':', alpha=0.3, linewidth=1.5, label='Good (70%)')
plt.axhline(y=50, color='orange', linestyle=':', alpha=0.3, linewidth=1.5, label='Fair (50%)')
plt.axhline(y=30, color='red', linestyle=':', alpha=0.3, linewidth=1.5, label='Poor (30%)')

# Add vertical line for the "15-sample threshold"
plt.axvline(x=15, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='15-sample threshold')

# Labels and title
plt.xlabel('Number of Test Samples per Class', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.title('LinearSVC: Accuracy vs Number of Samples per Class\n(FreeSound Audio Classification)',
         fontsize=16, fontweight='bold', pad=20)

# Grid
plt.grid(True, alpha=0.3, linestyle='--')

# Legend
plt.legend(loc='lower right', fontsize=10)

# Set limits
plt.xlim(0, max(samples) + 2)
plt.ylim(-5, 105)

# Add statistics text box
stats_text = f"""Dataset Statistics:
Total Classes: {len(df)}
Mean Accuracy: {accuracy.mean():.1f}%
Median Accuracy: {accuracy.median():.1f}%
Perfect Classes: {len(perfect)}
Failed Classes: {len(failed)}
Std Dev: {accuracy.std():.1f}%"""

plt.text(0.02, 0.98, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Tight layout
plt.tight_layout()

# Save
output_file = 'claude/lazypredict/findings/samples_vs_accuracy_scatter.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")

# Show
plt.show()

# Print some statistics
print("\n" + "="*70)
print("SAMPLE SIZE vs ACCURACY ANALYSIS")
print("="*70)

# Group by sample size ranges
bins = [0, 5, 10, 15, 20]
labels = ['1-4', '5-9', '10-14', '15+']
df['sample_range'] = pd.cut(df['Total_Samples'], bins=bins, labels=labels, include_lowest=True)

print("\nAccuracy by Sample Size Range:")
print("-" * 70)
for label in labels:
    subset = df[df['sample_range'] == label]
    if len(subset) > 0:
        print(f"{label:10s} samples: {len(subset):2d} classes | "
              f"Avg Acc: {subset['Accuracy'].mean()*100:5.1f}% | "
              f"Min: {subset['Accuracy'].min()*100:5.1f}% | "
              f"Max: {subset['Accuracy'].max()*100:5.1f}%")

print("\n" + "="*70)
print("OUTLIER ANALYSIS")
print("="*70)

# High samples, low accuracy (anomalies)
print("\nClasses with 15+ samples but <30% accuracy:")
outliers = df[(df['Total_Samples'] >= 15) & (df['Accuracy'] < 0.3)]
for _, row in outliers.iterrows():
    print(f"  • {row['Class']:40s} - {row['Total_Samples']:2d} samples, {row['Accuracy']*100:5.1f}% accuracy")

# Correlation
correlation = df['Total_Samples'].corr(df['Accuracy'])
print(f"\nCorrelation (samples vs accuracy): {correlation:.3f}")

print("\n✓ Analysis complete!")
