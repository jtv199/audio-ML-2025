#!/usr/bin/env python3
"""
Generate publication-quality figures from model comparison tables.
Creates compact, professional visualizations suitable for journals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.titlesize'] = 10
plt.rcParams['figure.dpi'] = 300

# Color palette
COLORS = {
    'VGGish_Tabular': '#2E7D32',  # Dark green
    'MobileNetV4': '#1565C0',      # Dark blue
    'VGGish_Solo': '#F57C00',      # Orange
    'LinearSVC': '#6A1B9A',        # Purple
    'DeiT_Tiny': '#C62828'         # Red
}

def create_model_performance_comparison():
    """Figure 1: Model Performance Overview with Per-Class Statistics"""

    # Data
    models = ['VGGish +\nTabular', 'MobileNetV4', 'LinearSVC', 'DeiT Tiny']
    accuracy = [89.24, 75.55, 57.44, 19.62]
    std_dev = [14.53, 20.44, 23.70, 17.47]
    perfect = [37, 14, 2, 0]
    failed = [5, 2, 6, 38]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    fig.suptitle('Model Performance Comparison', fontweight='bold', y=1.02)

    # Panel A: Accuracy with error bars (std dev)
    ax1 = axes[0]
    colors_list = ['#2E7D32', '#1565C0', '#6A1B9A', '#C62828']
    bars = ax1.bar(range(len(models)), accuracy, yerr=std_dev,
                   color=colors_list, alpha=0.8, capsize=4,
                   edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=0, ha='center')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top')

    # Add accuracy values on bars
    for i, (v, s) in enumerate(zip(accuracy, std_dev)):
        ax1.text(i, v + s + 2, f'{v:.1f}%', ha='center', va='bottom',
                fontsize=7, fontweight='bold')

    # Panel B: Perfect vs Failed Classes
    ax2 = axes[1]
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax2.bar(x - width/2, perfect, width, label='Perfect (100%)',
                    color='#4CAF50', alpha=0.9, edgecolor='black', linewidth=0.8)
    bars2 = ax2.bar(x + width/2, failed, width, label='Failed (0%)',
                    color='#F44336', alpha=0.9, edgecolor='black', linewidth=0.8)

    ax2.set_ylabel('Number of Classes', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=0, ha='center')
    ax2.legend(loc='upper right', frameon=True, fancybox=False,
               edgecolor='black', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=7)

    # Panel C: Consistency (Std Dev)
    ax3 = axes[2]
    bars = ax3.barh(range(len(models)), std_dev, color=colors_list,
                    alpha=0.8, edgecolor='black', linewidth=0.8)
    ax3.set_xlabel('Standard Deviation (%)', fontweight='bold')
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels(models)
    ax3.set_xlim(0, max(std_dev) * 1.15)
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top')

    # Add values
    for i, v in enumerate(std_dev):
        ax3.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=7, fontweight='bold')

    plt.tight_layout()
    plt.savefig('claude/lazypredict/findings/fig1_model_performance_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('claude/lazypredict/findings/fig1_model_performance_comparison.pdf',
                bbox_inches='tight', facecolor='white')
    print("✓ Saved Figure 1: Model Performance Comparison")
    plt.close()


def create_ttest_results_figure():
    """Figure 2: Statistical Significance Testing Results"""

    fig = plt.figure(figsize=(7.0, 3.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.35)

    # Panel A: Mean Difference vs DeiT Tiny
    ax1 = fig.add_subplot(gs[0, :])

    models = ['VGGish +\nTabular', 'MobileNetV4', 'VGGish\nSolo', 'LinearSVC']
    mean_diff = [73.66, 63.97, 48.49, 43.24]
    cohens_d = [2.684, 2.564, 1.634, 1.648]
    colors_list = ['#2E7D32', '#1565C0', '#F57C00', '#6A1B9A']

    bars = ax1.bar(range(len(models)), mean_diff, color=colors_list,
                   alpha=0.8, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Mean Difference vs DeiT Tiny\n(percentage points)',
                   fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=0, ha='center')
    ax1.set_ylim(0, max(mean_diff) * 1.15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add values and significance stars
    for i, v in enumerate(mean_diff):
        ax1.text(i, v + 1.5, f'+{v:.1f}%\n***', ha='center', va='bottom',
                fontsize=7, fontweight='bold')

    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax1.set_title('Improvement over DeiT Tiny Baseline (all p < 0.001)',
                  fontsize=9, pad=8)

    # Panel B: P-values (log scale)
    ax2 = fig.add_subplot(gs[1, 0])

    all_models = models + ['DeiT Tiny']
    p_values = [2.79e-35, 5.24e-34, 1.87e-22, 1.18e-22, 1.0]
    log_p_values = [-np.log10(p) for p in p_values]
    all_colors = colors_list + ['#C62828']

    bars = ax2.barh(range(len(all_models)), log_p_values, color=all_colors,
                    alpha=0.8, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel("-log₁₀(p-value)", fontweight='bold')
    ax2.set_yticks(range(len(all_models)))
    ax2.set_yticklabels(all_models)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add significance threshold line
    ax2.axvline(x=-np.log10(0.001), color='red', linestyle='--',
                linewidth=1, label='p=0.001', alpha=0.7)

    for i, (v, p) in enumerate(zip(log_p_values, p_values)):
        if p < 1:
            ax2.text(v + 0.5, i, f'p={p:.0e}', va='center', fontsize=6, fontweight='bold')

    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax2.set_xlim(0, max(log_p_values) * 1.15)

    # Panel C: VGGish Solo vs Tabular
    ax3 = fig.add_subplot(gs[1, 1])

    categories = ['VGGish\nSolo', 'VGGish +\nTabular']
    accuracies = [58.03, 83.21]
    colors_vgg = ['#F57C00', '#2E7D32']

    bars = ax3.bar(range(len(categories)), accuracies, color=colors_vgg,
                   alpha=0.8, edgecolor='black', linewidth=0.8)

    # Add arrow showing improvement
    ax3.annotate('', xy=(1, 83.21), xytext=(0, 58.03),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax3.text(0.5, 70, '+25.18%***', ha='center', va='center',
            fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7,
                     edgecolor='black', linewidth=1))

    ax3.set_ylabel('Accuracy (%)', fontweight='bold')
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, rotation=0, ha='center')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    for i, v in enumerate(accuracies):
        ax3.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom',
                fontsize=7, fontweight='bold')

    ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax3.set_title('Impact of Tabular Features', fontsize=9, pad=8)

    # Add legend for significance
    fig.text(0.5, 0.02, '*** p < 0.001 (highly significant) | Paired t-tests, n=74 classes',
             ha='center', fontsize=7, style='italic')

    plt.savefig('claude/lazypredict/findings/fig2_statistical_significance.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('claude/lazypredict/findings/fig2_statistical_significance.pdf',
                bbox_inches='tight', facecolor='white')
    print("✓ Saved Figure 2: Statistical Significance Testing")
    plt.close()


def create_vggish_improvement_heatmap():
    """Figure 3: Per-Class Improvement with Tabular Features"""

    # Read data
    df = pd.read_csv('all_4_models_per_class_comparison.csv')

    # Calculate improvement
    df['Improvement'] = df['Accuracy_vggish_tab'] - df['Accuracy_vggish_solo']
    df = df.sort_values('Improvement', ascending=False)

    # Take top 15 improved and bottom 5
    top_improved = df.head(15)
    bottom_changed = df.tail(5)
    combined = pd.concat([top_improved, bottom_changed])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 4.0),
                                     gridspec_kw={'width_ratios': [2, 1]})

    # Panel A: Top/Bottom Classes
    y_pos = np.arange(len(combined))
    colors = ['#2E7D32' if x > 0 else '#C62828' if x < 0 else '#9E9E9E'
              for x in combined['Improvement']]

    bars = ax1.barh(y_pos, combined['Improvement'], color=colors,
                    alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_yticks(y_pos)
    labels = [label.replace('_', ' ') for label in combined['Class']]
    ax1.set_yticklabels(labels, fontsize=6)
    ax1.set_xlabel('Improvement (percentage points)', fontweight='bold')
    ax1.set_title('Per-Class Impact of Adding Tabular Features',
                  fontsize=9, fontweight='bold', pad=8)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.invert_yaxis()

    # Add values
    for i, (idx, row) in enumerate(combined.iterrows()):
        val = row['Improvement']
        if abs(val) > 5:
            ax1.text(val - 2 if val > 0 else val + 2, i, f'{val:+.0f}',
                    va='center', ha='right' if val > 0 else 'left',
                    fontsize=5, fontweight='bold', color='white')

    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top')

    # Panel B: Summary Statistics
    improved = (df['Improvement'] > 0).sum()
    worsened = (df['Improvement'] < 0).sum()
    unchanged = (df['Improvement'] == 0).sum()

    categories = ['Improved', 'Worsened', 'Unchanged']
    values = [improved, worsened, unchanged]
    colors_pie = ['#2E7D32', '#C62828', '#9E9E9E']

    wedges, texts, autotexts = ax2.pie(values, labels=categories, autopct='%1.0f%%',
                                        colors=colors_pie, startangle=90,
                                        textprops={'fontsize': 8, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 1})

    ax2.set_title('Distribution of\nClass-Level Changes',
                  fontsize=9, fontweight='bold', pad=8)

    # Add count labels
    for i, (cat, val) in enumerate(zip(categories, values)):
        angle = (wedges[i].theta2 + wedges[i].theta1) / 2
        x = 0.7 * np.cos(np.deg2rad(angle))
        y = 0.7 * np.sin(np.deg2rad(angle))
        ax2.text(x, y, f'n={val}', ha='center', va='center',
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white',
                         alpha=0.8, edgecolor='black', linewidth=0.5))

    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig('claude/lazypredict/findings/fig3_vggish_improvement_analysis.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('claude/lazypredict/findings/fig3_vggish_improvement_analysis.pdf',
                bbox_inches='tight', facecolor='white')
    print("✓ Saved Figure 3: VGGish Improvement Analysis")
    plt.close()


def create_combined_summary_table():
    """Figure 4: Combined Summary Table as Image"""

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.axis('tight')
    ax.axis('off')

    # Table data
    data = [
        ['Model', 'Accuracy', 'Std Dev', 'Perfect', 'Failed', 'vs DeiT Δ'],
        ['VGGish + Tabular', '89.24%', '14.53%', '37', '5', '+73.66%***'],
        ['MobileNetV4', '75.55%', '20.44%', '14', '2', '+63.97%***'],
        ['VGGish Solo', '58.03%', '29.09%', '—', '—', '+48.49%***'],
        ['LinearSVC', '57.44%', '23.70%', '2', '6', '+43.24%***'],
        ['DeiT Tiny', '19.62%', '17.47%', '0', '38', 'baseline'],
    ]

    # Create table
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     colWidths=[0.28, 0.14, 0.14, 0.12, 0.12, 0.20])

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2)

    # Style header row
    for i in range(len(data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white')
        cell.set_edgecolor('white')
        cell.set_linewidth(1)

    # Style data rows with alternating colors
    colors_model = ['#E8F5E9', '#E3F2FD', '#FFF3E0', '#F3E5F5', '#FFEBEE']
    for i in range(1, len(data)):
        for j in range(len(data[0])):
            cell = table[(i, j)]
            cell.set_facecolor(colors_model[i-1])
            cell.set_edgecolor('gray')
            cell.set_linewidth(0.5)

            # Bold the model names
            if j == 0:
                cell.set_text_props(weight='bold', ha='left')

            # Highlight best values
            if i == 1 and j in [1, 2, 3, 5]:  # VGGish best metrics
                cell.set_text_props(weight='bold', color='#1B5E20')

    # Add title
    ax.text(0.5, 0.95, 'Comprehensive Model Performance Summary',
            ha='center', va='top', transform=ax.transAxes,
            fontsize=10, fontweight='bold')

    # Add footnote
    footnote = ('*** p < 0.001 (highly significant) | Perfect: classes with 100% accuracy | '
                'Failed: classes with 0% accuracy\n'
                'Δ: Mean difference vs DeiT Tiny baseline | All comparisons use paired t-tests (n=74 classes)')
    ax.text(0.5, 0.02, footnote, ha='center', va='bottom',
            transform=ax.transAxes, fontsize=6, style='italic', wrap=True)

    plt.savefig('claude/lazypredict/findings/fig4_summary_table.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('claude/lazypredict/findings/fig4_summary_table.pdf',
                bbox_inches='tight', facecolor='white')
    print("✓ Saved Figure 4: Summary Table")
    plt.close()


# Generate all figures
if __name__ == '__main__':
    print("\nGenerating publication-quality figures...")
    print("=" * 60)

    create_model_performance_comparison()
    create_ttest_results_figure()
    create_vggish_improvement_heatmap()
    create_combined_summary_table()

    print("=" * 60)
    print("\n✓ All figures generated successfully!")
    print("\nGenerated files:")
    print("  • fig1_model_performance_comparison.png/.pdf")
    print("  • fig2_statistical_significance.png/.pdf")
    print("  • fig3_vggish_improvement_analysis.png/.pdf")
    print("  • fig4_summary_table.png/.pdf")
    print("\nFigures saved in: claude/lazypredict/findings/")
    print("\nFormat: PNG (300 DPI) + PDF (vector)")
