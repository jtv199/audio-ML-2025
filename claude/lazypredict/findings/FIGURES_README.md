# Publication-Quality Figures - README

## Overview

This directory contains publication-ready figures comparing model performance for the FreeSound Audio Tagging 2019 project.

## Generated Figures

### Figure 1: Model Performance Comparison
**Files:** `fig1_model_performance_comparison.png` / `.pdf`
**Size:** 7.0" × 2.2"
**Panels:**
- **A**: Accuracy with standard deviation error bars
- **B**: Perfect classes (100% accuracy) vs Failed classes (0% accuracy)
- **C**: Consistency metrics (standard deviation across classes)

**Key Finding:** VGGish + Tabular achieves 89.24% accuracy with highest consistency (14.53% std dev)

---

### Figure 2: Statistical Significance Testing
**Files:** `fig2_statistical_significance.png` / `.pdf`
**Size:** 7.0" × 3.5"
**Panels:**
- **A**: Mean performance difference vs DeiT Tiny baseline (all p < 0.001)
- **B**: P-value significance levels (log scale)
- **C**: VGGish Solo vs VGGish + Tabular comparison (+25.18% improvement)

**Key Finding:** All models show highly significant improvements over DeiT Tiny, with VGGish + Tabular showing +73.66 percentage points improvement

---

### Figure 3: VGGish Improvement Analysis
**Files:** `fig3_vggish_improvement_analysis.png` / `.pdf`
**Size:** 7.0" × 4.0"
**Panels:**
- **A**: Per-class improvement when adding tabular features (top 15 + bottom 5 classes)
- **B**: Distribution pie chart showing 82% classes improved, 5% worsened, 12% unchanged

**Key Finding:** 61 out of 74 classes (82%) showed improvement with tabular features

---

### Figure 4: Comprehensive Summary Table
**Files:** `fig4_summary_table.png` / `.pdf`
**Size:** 7.0" × 2.5"
**Content:** Publication-ready table with all key metrics:
- Accuracy and standard deviation
- Perfect and failed class counts
- Statistical comparison vs DeiT Tiny baseline
- All models ranked by performance

**Key Finding:** Clear visual summary suitable for journal publication

---

## Technical Specifications

### Format
- **PNG**: 300 DPI resolution (suitable for presentations, posters, and online publication)
- **PDF**: Vector format (suitable for journal submission and print)

### Dimensions
- Width: 7.0 inches (standard single-column journal format)
- Height: Varies by figure (2.2" - 4.0")
- Can be scaled for double-column layouts

### Style
- **Font**: Serif (Times New Roman / DejaVu Serif)
- **Font sizes**:
  - Body text: 8pt
  - Axis labels: 9pt
  - Titles: 10pt
  - Panel labels (A, B, C): 12pt bold
- **Colors**: Color-blind friendly palette
  - VGGish + Tabular: Dark green (#2E7D32)
  - MobileNetV4: Dark blue (#1565C0)
  - VGGish Solo: Orange (#F57C00)
  - LinearSVC: Purple (#6A1B9A)
  - DeiT Tiny: Red (#C62828)

### Design Principles
- Clean, professional appearance suitable for peer-reviewed journals
- High information density without clutter
- Clear visual hierarchy
- Consistent styling across all figures
- Panel labels (A, B, C) for multi-panel figures
- Grid lines for readability
- Black borders for clarity in print

---

## Usage Guidelines

### For Journal Submission
1. Use **PDF format** for manuscript submission
2. Cite as "Figure 1", "Figure 2", etc.
3. Include figure captions from the main report
4. Figures are designed for single-column width but scale to double-column

### For Presentations
1. Use **PNG format** at 300 DPI
2. High resolution ensures clarity on projectors and screens
3. White background works well on slides

### For Online Publication
1. Use **PNG format** for web compatibility
2. 300 DPI ensures quality on high-DPI displays
3. Suitable for blogs, GitHub READMEs, and documentation

---

## Regenerating Figures

To regenerate all figures:

```bash
~/miniconda3/envs/freesound/bin/python generate_publication_figures.py
```

**Dependencies:**
- pandas
- numpy
- matplotlib
- seaborn
- scipy

**Input files required:**
- `all_4_models_per_class_comparison.csv`
- `claude/lazypredict/findings/error_analysis_deit_tiny.csv`

---

## Figure Captions (for manuscript)

### Figure 1
**Model Performance Comparison across Four Architectures.** (A) Overall accuracy with standard deviation error bars showing consistency across 74 audio classes. (B) Distribution of perfect (100% accuracy) and failed (0% accuracy) classes for each model. (C) Standard deviation values indicating model consistency, with lower values representing more stable performance across classes. VGGish + Tabular achieves the highest accuracy (89.24%) with best consistency (14.53% std dev).

### Figure 2
**Statistical Significance Analysis of Model Performance.** (A) Mean accuracy difference compared to DeiT Tiny baseline, showing all models significantly outperform the baseline (all p < 0.001, indicated by ***). (B) P-value significance levels on log scale, demonstrating highly significant improvements (all p < 10⁻²²). (C) Direct comparison of VGGish Solo vs VGGish + Tabular features, showing 25.18 percentage point improvement (p = 1.72e-11). All comparisons use paired t-tests across 74 classes.

### Figure 3
**Impact of Tabular Features on VGGish Performance.** (A) Per-class improvement distribution showing top 15 most improved and bottom 5 classes when adding tabular features to VGGish embeddings. Green bars indicate improvement, red bars indicate performance decrease. (B) Overall distribution of class-level changes: 82% of classes improved (n=61), 5% worsened (n=4), and 12% remained unchanged (n=9).

### Figure 4
**Comprehensive Model Performance Summary.** Publication-ready table presenting key metrics for all five models: overall accuracy, standard deviation (consistency), number of perfect and failed classes, and statistical comparison against DeiT Tiny baseline. *** indicates p < 0.001. VGGish + Tabular (highlighted in green) achieves best performance across all metrics.

---

## Citation

If you use these figures in your work, please cite:

```
FreeSound Audio Tagging 2019: Comparative Analysis of Deep Learning
and Traditional Machine Learning Approaches
Models: VGGish, MobileNetV4, DeiT Tiny, LinearSVC
Dataset: 74 audio classes, n=XX samples
Analysis: Paired t-tests with Bonferroni correction
```

---

**Generated:** October 17, 2025
**Author:** Claude Code Analysis
**Project:** Kaggle FreeSound Audio Tagging 2019
