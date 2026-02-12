---

## Model Performance Comparison with Per-Class Statistics

| Model | Accuracy | Std Dev | Perfect Classes | Failed Classes | Correlation |
|-------|----------|---------|-----------------|----------------|-------------|
| **VGGish + Tabular** | **89.24%** | 14.53% | 37 | 5 | 0.212 |
| **MobileNetV4** | 75.55% | 20.44% | 14 | 2 | -0.026 |
| **LinearSVC** | 57.44% | 23.70% | 2 | 6 | 0.083 |
| **DeiT Tiny** | 19.62% | 17.47% | 0 | 38 | 0.053 |

### Key Insights

- **VGGish + Tabular** is the clear winner with **89.24% accuracy** and the most perfect classes (37)
- **Highest consistency**: VGGish has the lowest standard deviation (14.53%), showing more stable performance across classes
- **MobileNetV4** achieves second place with 75.55% accuracy and only 2 failed classes
- **Best generalization**: VGGish shows positive correlation (0.212) with sample count, indicating it scales better with more data
- **DeiT Tiny struggles significantly** with 19.62% accuracy and 38 failed classes - likely needs more training data or fine-tuning

---

## Statistical Significance Testing (T-Tests)

### Comparison vs DeiT Tiny (Baseline)

All models were compared against **DeiT Tiny** as the baseline using paired t-tests (n=74 classes):

| Model | Mean Diff vs DeiT | t-statistic | p-value | Cohen's d | Significance | Result |
|-------|-------------------|-------------|---------|-----------|--------------|--------|
| **VGGish + Tabular** | **+73.66%** | 23.088 | 2.79e-35 | 2.684 | *** | Highly significant improvement |
| **MobileNetV4** | +63.97% | 22.053 | 5.24e-34 | 2.564 | *** | Highly significant improvement |
| **VGGish Solo** | +48.49% | 14.054 | 1.87e-22 | 1.634 | *** | Highly significant improvement |
| **LinearSVC** | +43.24% | 14.175 | 1.18e-22 | 1.648 | *** | Highly significant improvement |

**Interpretation:**
- All models show **highly significant improvements** over DeiT Tiny (p < 0.001)
- Effect sizes are **very large** for all comparisons (Cohen's d > 0.8)
- VGGish + Tabular shows the **largest improvement** with 73.66 percentage points gain
- MobileNetV4 has the second-largest improvement (63.97 percentage points)

### VGGish Solo vs VGGish + Tabular Features

Comparing the impact of adding tabular features to VGGish embeddings:

| Comparison | Mean Diff | t-statistic | p-value | Cohen's d | Significance | Classes Improved | Classes Worsened |
|------------|-----------|-------------|---------|-----------|--------------|------------------|------------------|
| **VGGish + Tabular vs Solo** | **+25.18%** | -7.954 | 1.72e-11 | 0.925 | *** | 61 (82%) | 4 (5%) |

**Interpretation:**
- Adding tabular features provides a **highly significant improvement** (p = 1.72e-11)
- **Very large effect size** (Cohen's d = 0.925)
- Improvement of **25.18 percentage points** on average
- **82% of classes** (61 out of 74) showed improvement with tabular features
- Only **5% of classes** (4 out of 74) performed worse
- The remaining 9 classes (12%) showed no change

**Statistical Notes:**
- `***` = p < 0.001 (highly significant)
- Cohen's d interpretation: |d| < 0.2 (small), 0.2-0.5 (medium), 0.5-0.8 (large), > 0.8 (very large)
- Paired t-tests were used (same 74 classes compared across all models)

---

## Publication-Quality Figures

The following publication-ready figures have been generated for journal submission:

### Figure 1: Model Performance Comparison
![Model Performance Comparison](fig1_model_performance_comparison.png)
*Three-panel comparison showing (A) accuracy with standard deviation, (B) perfect vs failed classes, and (C) consistency metrics.*

### Figure 2: Statistical Significance Testing
![Statistical Significance](fig2_statistical_significance.png)
*Statistical analysis showing (A) mean differences vs DeiT Tiny baseline, (B) p-value significance levels, and (C) impact of tabular features on VGGish.*

### Figure 3: VGGish Improvement Analysis
![VGGish Improvement](fig3_vggish_improvement_analysis.png)
*Per-class improvement analysis showing (A) top/bottom classes affected by tabular features and (B) distribution of class-level changes.*

### Figure 4: Comprehensive Summary Table
![Summary Table](fig4_summary_table.png)
*Publication-ready table summarizing all key metrics across models.*

**Figure Format:**
- **PNG**: 300 DPI (for presentations and documents)
- **PDF**: Vector format (for journal submissions)
- **Size**: Compact 7" width (standard single-column journal format)
