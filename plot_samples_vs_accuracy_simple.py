#!/usr/bin/env python3
"""
Plot: Number of Samples vs Accuracy per Class
Color-coded by acoustic similarity clusters
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the error analysis data
df = pd.read_csv('claude/lazypredict/findings/error_analysis_linearsvc.csv')

# Define acoustic similarity clusters based on confusion patterns
clusters = {
    'Metallic': ['Keys_jangling', 'Scissors', 'Cutlery_and_silverware', 'Chink_and_clink',
                 'Glockenspiel', 'Hi-hat', 'Cowbell', 'Tambourine'],
    'Ambient/Texture': ['Hiss', 'Crackle', 'Stream', 'Waves_and_surf', 'Trickle_and_dribble',
                        'Frying_(food)', 'Raindrop'],
    'Human_Vocal': ['Whispering', 'Breathing', 'Sigh', 'Gasp', 'Burping_and_eructation',
                    'Chewing_and_mastication', 'Sneeze', 'Yell', 'Screaming'],
    'Speech': ['Male_speech_and_man_speaking', 'Female_speech_and_woman_speaking',
               'Child_speech_and_kid_speaking'],
    'Music/Instruments': ['Acoustic_guitar', 'Electric_guitar', 'Bass_guitar', 'Harmonica',
                         'Marimba_and_xylophone', 'Clarinet'],
    'Singing': ['Male_singing', 'Female_singing'],
    'Percussion': ['Bass_drum', 'Knock', 'Clapping', 'Finger_snapping', 'Tap', 'Slam'],
    'Animals': ['Bark', 'Meow', 'Purr', 'Cricket', 'Chirp_and_tweet', 'Crow'],
    'Mechanical': ['Mechanical_fan', 'Microwave_oven', 'Printer', 'Computer_keyboard',
                   'Cupboard_open_or_close', 'Drawer_open_or_close'],
    'Vehicles': ['Accelerating_and_revving_and_vroom', 'Bus', 'Car_passing_by', 'Motorcycle',
                 'Race_car_and_auto_racing', 'Skateboard', 'Bicycle_bell'],
    'Bells': ['Church_bell', 'Gong'],
    'Fabric/Texture': ['Zipper_(clothing)', 'Writing', 'Shatter'],
    'Water_Actions': ['Fill_(with_liquid)', 'Bathtub_(filling_or_washing)', 'Toilet_flush',
                      'Drip', 'Gurgling'],
    'Other': ['Buzz', 'Tick-tock', 'Traffic_noise_and_roadway_noise', 'Crowd',
             'Fart', 'Run', 'Walk_and_footsteps', 'Squeak']
}

# Assign cluster to each class
def get_cluster(class_name):
    for cluster_name, classes in clusters.items():
        if class_name in classes:
            return cluster_name
    return 'Other'

df['cluster'] = df['Class'].apply(get_cluster)

# Color map for clusters
cluster_colors = {
    'Metallic': '#FF6B6B',
    'Ambient/Texture': '#4ECDC4',
    'Human_Vocal': '#FFE66D',
    'Speech': '#95E1D3',
    'Music/Instruments': '#F38181',
    'Singing': '#AA96DA',
    'Percussion': '#FCBAD3',
    'Animals': '#A8E6CF',
    'Mechanical': '#FFD3B6',
    'Vehicles': '#DCEDC1',
    'Bells': '#FFC8DD',
    'Fabric/Texture': '#BDB2FF',
    'Water_Actions': '#A0C4FF',
    'Other': '#CCCCCC'
}

# Extract data
samples = df['Total_Samples']
accuracy = df['Accuracy'] * 100  # Convert to percentage
colors = df['cluster'].map(cluster_colors)

# Create the plot
fig, ax = plt.subplots(figsize=(16, 10))

# Plot each cluster separately for legend
for cluster_name, color in cluster_colors.items():
    cluster_df = df[df['cluster'] == cluster_name]
    if len(cluster_df) > 0:
        ax.scatter(cluster_df['Total_Samples'],
                  cluster_df['Accuracy'] * 100,
                  s=120,
                  alpha=0.7,
                  c=color,
                  edgecolors='black',
                  linewidth=0.8,
                  label=f'{cluster_name} ({len(cluster_df)})')

# Add trend line
z = np.polyfit(samples, accuracy, 2)
p = np.poly1d(z)
x_trend = np.linspace(samples.min(), samples.max(), 100)
ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2.5, label='Trend', zorder=1)

# Add the "15-sample threshold" line
ax.axvline(x=15, color='blue', linestyle='--', alpha=0.4, linewidth=2, label='15-sample threshold')

# Add performance tier lines
ax.axhline(y=90, color='green', linestyle=':', alpha=0.2, linewidth=1.5)
ax.axhline(y=70, color='yellowgreen', linestyle=':', alpha=0.2, linewidth=1.5)
ax.axhline(y=50, color='orange', linestyle=':', alpha=0.2, linewidth=1.5)

# Labels and title
ax.set_xlabel('Number of Test Samples per Class', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('LinearSVC Performance: Samples vs Accuracy by Acoustic Cluster\nFreeSound Audio Classification (74 Classes)',
            fontsize=16, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', zorder=0)

# Legend
ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.9)

# Set limits
ax.set_xlim(0, max(samples) + 2)
ax.set_ylim(-5, 105)

# Add statistics box
stats_text = f"""Overall Stats:
Classes: {len(df)}
Mean Acc: {accuracy.mean():.1f}%
Correlation: {samples.corr(accuracy):.3f}"""

ax.text(0.02, 0.98, stats_text,
       transform=ax.transAxes,
       fontsize=11,
       verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()

# Save
output_file = 'claude/lazypredict/findings/samples_vs_accuracy_clustered.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")

# Print cluster statistics
print("\n" + "="*70)
print("CLUSTER PERFORMANCE ANALYSIS")
print("="*70)

cluster_stats = df.groupby('cluster').agg({
    'Accuracy': ['mean', 'std', 'min', 'max', 'count']
}).round(3)

cluster_stats.columns = ['Mean_Acc', 'Std_Acc', 'Min_Acc', 'Max_Acc', 'Count']
cluster_stats = cluster_stats.sort_values('Mean_Acc', ascending=False)

print(cluster_stats.to_string())

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
