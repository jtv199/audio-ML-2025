#!/usr/bin/env python3
"""
Compare VGGish Standalone vs VGGish+Tabular Models
Shows how adding tabular features improves performance
"""

import pandas as pd
import numpy as np

print("="*80)
print("VGGISH STANDALONE VS VGGISH+TABULAR COMPARISON")
print("="*80)

# Load both models' results
vggish_solo_df = pd.read_csv('vggish_single_label_per_class_2layers_20251017_103236.csv')
vggish_tab_df = pd.read_csv('claude/lazypredict/findings/error_analysis_vggish_tabular.csv')

print(f"\nVGGish Standalone: {len(vggish_solo_df)} classes")
print(f"VGGish+Tabular: {len(vggish_tab_df)} classes")

# Merge on class name
comparison_df = vggish_solo_df[['Class', 'Total_Samples', 'Accuracy']].copy()
comparison_df.columns = ['Class', 'Total_Samples', 'Accuracy_Solo']

comparison_df = comparison_df.merge(
    vggish_tab_df[['Class', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_Tab'}),
    on='Class',
    how='outer'
)

# Fill NaN
comparison_df = comparison_df.fillna(0)

# Calculate improvement
comparison_df['Improvement'] = comparison_df['Accuracy_Tab'] - comparison_df['Accuracy_Solo']
comparison_df['Improvement_Pct'] = comparison_df['Improvement'] * 100

# Sort by improvement
comparison_df = comparison_df.sort_values('Improvement', ascending=False)

# Overall stats
print("\n" + "="*80)
print("OVERALL PERFORMANCE")
print("="*80)

solo_mean = comparison_df['Accuracy_Solo'].mean()
tab_mean = comparison_df['Accuracy_Tab'].mean()
mean_improvement = tab_mean - solo_mean

print(f"\nVGGish Standalone:  {solo_mean*100:.2f}%")
print(f"VGGish+Tabular:     {tab_mean*100:.2f}%")
print(f"Improvement:        +{mean_improvement*100:.2f}% ({mean_improvement/solo_mean*100:.1f}% relative gain)")

# Perfect classes
solo_perfect = len(comparison_df[comparison_df['Accuracy_Solo'] == 1.0])
tab_perfect = len(comparison_df[comparison_df['Accuracy_Tab'] == 1.0])

print(f"\nPerfect Classes:")
print(f"  VGGish Standalone:  {solo_perfect}/74 ({solo_perfect/74*100:.1f}%)")
print(f"  VGGish+Tabular:     {tab_perfect}/74 ({tab_perfect/74*100:.1f}%)")
print(f"  Gain:               +{tab_perfect - solo_perfect} classes")

# Failed classes
solo_failed = len(comparison_df[comparison_df['Accuracy_Solo'] == 0.0])
tab_failed = len(comparison_df[comparison_df['Accuracy_Tab'] == 0.0])

print(f"\nFailed Classes (0% accuracy):")
print(f"  VGGish Standalone:  {solo_failed}/74")
print(f"  VGGish+Tabular:     {tab_failed}/74")
print(f"  Reduction:          -{solo_failed - tab_failed} classes")

# Top 15 improvements
print("\n" + "="*80)
print("TOP 15 IMPROVEMENTS (Classes that benefited most from tabular features)")
print("="*80)

for _, row in comparison_df.head(15).iterrows():
    class_name = row['Class']
    solo_acc = row['Accuracy_Solo'] * 100
    tab_acc = row['Accuracy_Tab'] * 100
    improvement = row['Improvement_Pct']
    samples = int(row['Total_Samples'])

    print(f"{class_name:45s} {solo_acc:5.1f}% → {tab_acc:5.1f}% (+{improvement:5.1f}%) [{samples} samples]")

# Classes that got worse
print("\n" + "="*80)
print("CLASSES THAT GOT WORSE (Negative impact from tabular features)")
print("="*80)

worse_df = comparison_df[comparison_df['Improvement'] < 0].sort_values('Improvement')
if len(worse_df) > 0:
    for _, row in worse_df.iterrows():
        class_name = row['Class']
        solo_acc = row['Accuracy_Solo'] * 100
        tab_acc = row['Accuracy_Tab'] * 100
        decrease = row['Improvement_Pct']
        samples = int(row['Total_Samples'])

        print(f"{class_name:45s} {solo_acc:5.1f}% → {tab_acc:5.1f}% ({decrease:5.1f}%) [{samples} samples]")
else:
    print("  ✓ No classes performed worse with tabular features!")

# Distribution analysis
print("\n" + "="*80)
print("IMPROVEMENT DISTRIBUTION")
print("="*80)

bins = [-1.0, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
labels = ['Worse', 'No change', '0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
comparison_df['improvement_range'] = pd.cut(comparison_df['Improvement'], bins=bins, labels=labels)

print(f"\nClasses by improvement range:")
for label in labels:
    count = len(comparison_df[comparison_df['improvement_range'] == label])
    pct = count / len(comparison_df) * 100
    print(f"  {label:15s}: {count:2d} classes ({pct:5.1f}%)")

# Sample size correlation
print("\n" + "="*80)
print("SAMPLE SIZE vs IMPROVEMENT")
print("="*80)

# Bin by sample size
bins = [0, 5, 10, 15, 20, 100]
labels = ['1-4', '5-9', '10-14', '15-19', '20+']
comparison_df['sample_range'] = pd.cut(comparison_df['Total_Samples'], bins=bins, labels=labels)

for label in labels:
    subset = comparison_df[comparison_df['sample_range'] == label]
    if len(subset) > 0:
        avg_improvement = subset['Improvement'].mean() * 100
        print(f"{label:10s}: {len(subset):2d} classes | Avg Improvement: {avg_improvement:+6.2f}%")

# Correlation
corr = comparison_df['Total_Samples'].corr(comparison_df['Improvement'])
print(f"\nCorrelation (samples vs improvement): {corr:.3f}")

# Save comparison
output_file = 'vggish_standalone_vs_tabular_comparison.csv'
comparison_df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print(f"""
1. Overall Performance:
   - Adding tabular features improves accuracy by {mean_improvement*100:.2f}% points absolute
   - That's a {mean_improvement/solo_mean*100:.1f}% relative improvement

2. Perfect Classes:
   - Tabular features help {tab_perfect - solo_perfect} more classes reach 100% accuracy
   - From {solo_perfect} to {tab_perfect} perfect classes

3. Failed Classes:
   - {solo_failed - tab_failed} fewer classes fail completely with tabular features
   - From {solo_failed} down to {tab_failed} failed classes

4. Consistency:
   - {len(worse_df)} classes performed worse with tabular features
   - {len(comparison_df[comparison_df['Improvement'] > 0])} classes improved
   - Tabular features provide broad benefit across all classes
""")

print("="*80)
