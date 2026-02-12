#!/usr/bin/env python3
"""
Comparison Plot: MobileNet vs LinearSVC
Sample Size vs Accuracy with 5 Acoustic Clusters
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read both datasets
df_svc = pd.read_csv('claude/lazypredict/findings/error_analysis_linearsvc.csv')
df_mob = pd.read_csv('claude/lazypredict/findings/error_analysis_mobilenet.csv')

# Merge on class name
df = df_svc[['Class', 'Total_Samples', 'Accuracy']].merge(
    df_mob[['Class', 'Accuracy']],
    on='Class',
    suffixes=('_svc', '_mob')
)

# Convert to percentage
df['Accuracy_svc'] = df['Accuracy_svc'] * 100
df['Accuracy_mob'] = df['Accuracy_mob'] * 100

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

# Assign cluster
def get_cluster_5(class_name):
    for cluster_name, classes in clusters_5.items():
        if class_name in classes:
            return cluster_name
    return 'Other'

df['cluster'] = df['Class'].apply(get_cluster_5)

# Color map for 5 clusters
cluster_colors_5 = {
    'Percussion/Transients': '#FF6B6B',
    'Instruments/Music': '#4ECDC4',
    'Human/Vocal': '#FFE66D',
    'Mechanical/Ambient': '#95E1D3',
    'Nature/Animals': '#A8E6CF'
}

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# Plot 1: LinearSVC
for cluster_name, color in cluster_colors_5.items():
    cluster_df = df[df['cluster'] == cluster_name]
    if len(cluster_df) > 0:
        ax1.scatter(cluster_df['Total_Samples'],
                   cluster_df['Accuracy_svc'],
                   s=120,
                   alpha=0.7,
                   c=color,
                   edgecolors='black',
                   linewidth=1,
                   label=f'{cluster_name} ({len(cluster_df)})',
                   zorder=3)

# Trend line for SVC
z1 = np.polyfit(df['Total_Samples'], df['Accuracy_svc'], 2)
p1 = np.poly1d(z1)
x_trend = np.linspace(df['Total_Samples'].min(), df['Total_Samples'].max(), 100)
ax1.plot(x_trend, p1(x_trend), "r--", alpha=0.6, linewidth=2.5, label='Trend', zorder=1)
ax1.axvline(x=15, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='15-sample threshold', zorder=2)

ax1.set_xlabel('Number of Test Samples per Class', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('LinearSVC Performance\n(Classical ML - STFT/Mel Features)', fontsize=15, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax1.legend(loc='lower right', fontsize=9, framealpha=0.95)
ax1.set_xlim(0, df['Total_Samples'].max() + 2)
ax1.set_ylim(-5, 105)

# Stats box for SVC
stats_svc = f"""LinearSVC:
Mean: {df['Accuracy_svc'].mean():.1f}%
Median: {df['Accuracy_svc'].median():.1f}%
Correlation: {df['Total_Samples'].corr(df['Accuracy_svc']):.3f}"""
ax1.text(0.02, 0.98, stats_svc, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))

# Plot 2: MobileNet
for cluster_name, color in cluster_colors_5.items():
    cluster_df = df[df['cluster'] == cluster_name]
    if len(cluster_df) > 0:
        ax2.scatter(cluster_df['Total_Samples'],
                   cluster_df['Accuracy_mob'],
                   s=120,
                   alpha=0.7,
                   c=color,
                   edgecolors='black',
                   linewidth=1,
                   label=f'{cluster_name} ({len(cluster_df)})',
                   zorder=3)

# Trend line for MobileNet
z2 = np.polyfit(df['Total_Samples'], df['Accuracy_mob'], 2)
p2 = np.poly1d(z2)
ax2.plot(x_trend, p2(x_trend), "r--", alpha=0.6, linewidth=2.5, label='Trend', zorder=1)
ax2.axvline(x=15, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='15-sample threshold', zorder=2)

ax2.set_xlabel('Number of Test Samples per Class', fontsize=13, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_title('MobileNetV4 Performance\n(Deep Learning - Mel-Spectrogram CNN)', fontsize=15, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
ax2.set_xlim(0, df['Total_Samples'].max() + 2)
ax2.set_ylim(-5, 105)

# Stats box for MobileNet
stats_mob = f"""MobileNet:
Mean: {df['Accuracy_mob'].mean():.1f}%
Median: {df['Accuracy_mob'].median():.1f}%
Correlation: {df['Total_Samples'].corr(df['Accuracy_mob']):.3f}"""
ax2.text(0.02, 0.98, stats_mob, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=2))

# Overall title
fig.suptitle('Model Comparison: Sample Size vs Accuracy (5 Acoustic Clusters)\nFreeSound Audio Classification (74 Classes)',
            fontsize=17, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_file = 'claude/lazypredict/findings/mobilenet_vs_linearsvc_scatter.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved to: {output_file}")

# Print comparative statistics
print("\n" + "="*80)
print("MODEL COMPARISON ANALYSIS")
print("="*80)

print("\nOverall Performance:")
print(f"  LinearSVC    Mean: {df['Accuracy_svc'].mean():.2f}%  Median: {df['Accuracy_svc'].median():.2f}%")
print(f"  MobileNet    Mean: {df['Accuracy_mob'].mean():.2f}%  Median: {df['Accuracy_mob'].median():.2f}%")
print(f"  Improvement: +{df['Accuracy_mob'].mean() - df['Accuracy_svc'].mean():.2f}%")

print("\nCorrelation with Sample Size:")
print(f"  LinearSVC:  {df['Total_Samples'].corr(df['Accuracy_svc']):.3f}")
print(f"  MobileNet:  {df['Total_Samples'].corr(df['Accuracy_mob']):.3f}")

print("\nPer-Cluster Performance:")
print("-" * 80)
for cluster_name in cluster_colors_5.keys():
    cluster_df = df[df['cluster'] == cluster_name]
    if len(cluster_df) > 0:
        svc_mean = cluster_df['Accuracy_svc'].mean()
        mob_mean = cluster_df['Accuracy_mob'].mean()
        improvement = mob_mean - svc_mean
        print(f"{cluster_name:25s}: SVC {svc_mean:5.1f}% | MobileNet {mob_mean:5.1f}% | Diff: {improvement:+6.1f}%")

print("\n" + "="*80)
print("BIGGEST IMPROVEMENTS (MobileNet vs LinearSVC)")
print("="*80)

df['improvement'] = df['Accuracy_mob'] - df['Accuracy_svc']
top_improvements = df.nlargest(10, 'improvement')

for _, row in top_improvements.iterrows():
    print(f"{row['Class']:40s} {row['improvement']:+6.1f}% "
          f"({row['Accuracy_mob']:5.1f}% vs {row['Accuracy_svc']:5.1f}%)")

print("\n" + "="*80)
print("BIGGEST REGRESSIONS (MobileNet worse than LinearSVC)")
print("="*80)

bottom_improvements = df.nsmallest(10, 'improvement')

for _, row in bottom_improvements.iterrows():
    print(f"{row['Class']:40s} {row['improvement']:+6.1f}% "
          f"({row['Accuracy_mob']:5.1f}% vs {row['Accuracy_svc']:5.1f}%)")

print("\n✓ Analysis complete!")
