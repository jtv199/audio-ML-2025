#!/usr/bin/env python3
"""
Plot: Number of Samples vs Accuracy per Class
Simplified to 5 acoustic clusters maximum
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the error analysis data
df = pd.read_csv('claude/lazypredict/findings/error_analysis_linearsvc.csv')

# Define 5 MAIN acoustic similarity clusters
clusters_5 = {
    'Percussion/Transients': [
        'Finger_snapping', 'Hi-hat', 'Knock', 'Clapping', 'Tap', 'Bass_drum',
        'Bicycle_bell', 'Glockenspiel', 'Tambourine', 'Cowbell', 'Slam'
    ],
    'Instruments/Music': [
        'Acoustic_guitar', 'Electric_guitar', 'Bass_guitar', 'Harmonica',
        'Marimba_and_xylophone', 'Clarinet', 'Accordion',
        'Male_singing', 'Female_singing'
    ],
    'Human/Vocal': [
        'Whispering', 'Breathing', 'Sigh', 'Gasp', 'Burping_and_eructation',
        'Chewing_and_mastication', 'Sneeze', 'Yell', 'Screaming',
        'Male_speech_and_man_speaking', 'Female_speech_and_woman_speaking',
        'Child_speech_and_kid_speaking'
    ],
    'Mechanical/Ambient': [
        'Hiss', 'Crackle', 'Buzz', 'Tick-tock', 'Mechanical_fan',
        'Microwave_oven', 'Printer', 'Computer_keyboard',
        'Cupboard_open_or_close', 'Drawer_open_or_close',
        'Accelerating_and_revving_and_vroom', 'Bus', 'Car_passing_by',
        'Motorcycle', 'Race_car_and_auto_racing', 'Skateboard',
        'Traffic_noise_and_roadway_noise', 'Keys_jangling', 'Scissors',
        'Cutlery_and_silverware', 'Chink_and_clink', 'Zipper_(clothing)',
        'Writing', 'Shatter'
    ],
    'Nature/Animals': [
        'Bark', 'Meow', 'Purr', 'Cricket', 'Chirp_and_tweet',
        'Stream', 'Waves_and_surf', 'Trickle_and_dribble',
        'Raindrop', 'Drip', 'Gurgling', 'Fill_(with_liquid)',
        'Bathtub_(filling_or_washing)', 'Toilet_flush',
        'Water_tap_and_faucet', 'Frying_(food)', 'Applause',
        'Crowd', 'Church_bell', 'Gong', 'Fart', 'Run',
        'Walk_and_footsteps', 'Squeak'
    ]
}

# Assign cluster to each class
def get_cluster_5(class_name):
    for cluster_name, classes in clusters_5.items():
        if class_name in classes:
            return cluster_name
    return 'Other'

df['cluster'] = df['Class'].apply(get_cluster_5)

# Color map for 5 clusters (distinct colors)
cluster_colors_5 = {
    'Percussion/Transients': '#FF6B6B',      # Red
    'Instruments/Music': '#4ECDC4',          # Teal
    'Human/Vocal': '#FFE66D',                # Yellow
    'Mechanical/Ambient': '#95E1D3',         # Mint
    'Nature/Animals': '#A8E6CF'              # Light green
}

# Extract data
samples = df['Total_Samples']
accuracy = df['Accuracy'] * 100
colors = df['cluster'].map(cluster_colors_5)

# Create the plot
fig, ax = plt.subplots(figsize=(16, 10))

# Plot each cluster separately for legend
for cluster_name, color in cluster_colors_5.items():
    cluster_df = df[df['cluster'] == cluster_name]
    if len(cluster_df) > 0:
        ax.scatter(cluster_df['Total_Samples'],
                  cluster_df['Accuracy'] * 100,
                  s=150,
                  alpha=0.7,
                  c=color,
                  edgecolors='black',
                  linewidth=1.2,
                  label=f'{cluster_name} ({len(cluster_df)})',
                  zorder=3)

# Add trend line
z = np.polyfit(samples, accuracy, 2)
p = np.poly1d(z)
x_trend = np.linspace(samples.min(), samples.max(), 100)
ax.plot(x_trend, p(x_trend), "r--", alpha=0.6, linewidth=3, label='Trend (polynomial)', zorder=1)

# Add the "15-sample threshold" line
ax.axvline(x=15, color='blue', linestyle='--', alpha=0.5, linewidth=2.5, label='15-sample threshold', zorder=2)

# Add performance tier lines
ax.axhline(y=90, color='green', linestyle=':', alpha=0.25, linewidth=2)
ax.axhline(y=70, color='yellowgreen', linestyle=':', alpha=0.25, linewidth=2)
ax.axhline(y=50, color='orange', linestyle=':', alpha=0.25, linewidth=2)
ax.axhline(y=30, color='red', linestyle=':', alpha=0.25, linewidth=2)

# Labels and title
ax.set_xlabel('Number of Test Samples per Class', fontsize=15, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=15, fontweight='bold')
ax.set_title('LinearSVC Performance: Samples vs Accuracy (5 Acoustic Clusters)\nFreeSound Audio Classification (74 Classes)',
            fontsize=17, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', zorder=0)

# Legend
ax.legend(loc='lower right', fontsize=12, ncol=1, framealpha=0.95, shadow=True)

# Set limits
ax.set_xlim(0, max(samples) + 2)
ax.set_ylim(-5, 105)

# Add statistics box
stats_text = f"""Overall Stats:
Classes: {len(df)}
Mean Acc: {accuracy.mean():.1f}%
Median Acc: {accuracy.median():.1f}%
Correlation: {samples.corr(accuracy):.3f}"""

ax.text(0.02, 0.98, stats_text,
       transform=ax.transAxes,
       fontsize=12,
       verticalalignment='top',
       fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))

plt.tight_layout()

# Save
output_file = 'claude/lazypredict/findings/samples_vs_accuracy_5clusters.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")

# Print cluster statistics
print("\n" + "="*70)
print("5-CLUSTER PERFORMANCE ANALYSIS")
print("="*70)

cluster_stats = df.groupby('cluster').agg({
    'Accuracy': ['mean', 'std', 'min', 'max', 'count']
}).round(3)

cluster_stats.columns = ['Mean_Acc', 'Std_Acc', 'Min_Acc', 'Max_Acc', 'Count']
cluster_stats = cluster_stats.sort_values('Mean_Acc', ascending=False)

print("\nCluster Performance Summary:")
print("-" * 70)
for cluster_name in cluster_stats.index:
    row = cluster_stats.loc[cluster_name]
    print(f"{cluster_name:25s}: {row['Count']:2.0f} classes | "
          f"Avg: {row['Mean_Acc']*100:5.1f}% | "
          f"Range: {row['Min_Acc']*100:4.1f}%-{row['Max_Acc']*100:5.1f}%")

print("\n" + "="*70)
print("SAMPLE SIZE ANALYSIS")
print("="*70)

bins = [0, 5, 10, 15, 20]
labels = ['1-4', '5-9', '10-14', '15+']
df['sample_range'] = pd.cut(df['Total_Samples'], bins=bins, labels=labels, include_lowest=True)

for label in labels:
    subset = df[df['sample_range'] == label]
    if len(subset) > 0:
        print(f"{label:10s}: {len(subset):2d} classes | Avg: {subset['Accuracy'].mean()*100:5.1f}%")

print(f"\nCorrelation: {samples.corr(accuracy):.3f}")
print("\n✓ Complete!")
