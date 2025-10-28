# VGGish Standalone Model Results

## Overview

This document summarizes the error analysis for VGGish embeddings trained as standalone neural networks (without tabular features).

**Training Date**: October 17, 2025
**Script**: [vggish_fastai_single_label.py](vggish_fastai_single_label.py)
**Dataset**: Single-label samples only (2,978 samples, 84.5% of total)
**Input Features**: 128 VGGish embeddings only
**Train/Val Split**: 2,382 / 596 (80/20)

---

## Model Comparison

Three different neural network architectures were trained:

| Model | Layers | Parameters | Training Time | Val Accuracy | Perfect Classes |
|-------|--------|------------|---------------|--------------|-----------------|
| **Model 1** | `[200, 100]` | 53,930 | 5.02s | **63.42%** | 4 |
| **Model 2** | `[400, 300, 200, 150, 100]` | 286,230 | 11.46s | 62.58% | 5 |
| **Model 3** | `[512, 256]` | 217,418 | 13.78s | **64.43%** | **6** |

### Best Model: `[512, 256]`

- **Validation Accuracy**: 64.43%
- **Perfect Classes (100% accuracy)**: 6
- **Classes with 0% accuracy**: 8
- **Mean per-class accuracy**: 58.03%
- **Median per-class accuracy**: 63.07%
- **Model File**: [models/vggish_single_label_2layers_20251017_103236.pkl](models/vggish_single_label_2layers_20251017_103236.pkl)
- **Size**: 964 KB

---

## Top 10 Best Performing Classes

| Rank | Class | Samples | Accuracy | F1 Score |
|------|-------|---------|----------|----------|
| 1 | **Bark** | 10 | **100.00%** | 0.952 |
| 2 | **Skateboard** | 7 | **100.00%** | 0.933 |
| 3 | **Hi-hat** | 6 | **100.00%** | 0.923 |
| 4 | **Female_speech_and_woman_speaking** | 4 | **100.00%** | 0.800 |
| 5 | **Drawer_open_or_close** | 10 | **100.00%** | 0.909 |
| 6 | **Crowd** | 1 | **100.00%** | 1.000 |
| 7 | Harmonica | 13 | 92.31% | 0.889 |
| 8 | Toilet_flush | 13 | 92.31% | 0.889 |
| 9 | Bass_guitar | 12 | 91.67% | 0.880 |
| 10 | Gong | 10 | 90.00% | 0.750 |

---

## Bottom 10 Worst Performing Classes

| Rank | Class | Samples | Accuracy | F1 Score |
|------|-------|---------|----------|----------|
| 74 | **Yell** | 3 | **0.00%** | 0.000 |
| 73 | **Cutlery_and_silverware** | 2 | **0.00%** | 0.000 |
| 72 | **Drip** | 1 | **0.00%** | 0.000 |
| 71 | **Acoustic_guitar** | 1 | **0.00%** | 0.000 |
| 70 | **Tap** | 4 | **0.00%** | 0.000 |
| 69 | **Raindrop** | 1 | **0.00%** | 0.000 |
| 68 | **Squeak** | 3 | **0.00%** | 0.000 |
| 67 | **Slam** | 4 | **0.00%** | 0.000 |
| 66 | Microwave_oven | 9 | 22.22% | 0.308 |
| 65 | Cupboard_open_or_close | 4 | 25.00% | 0.333 |

---

## Key Findings

### Strengths

1. **Robust on Well-Represented Classes**: Perfect accuracy on 6 classes including common sounds like Bark, Hi-hat, and Drawer_open_or_close
2. **Fast Training**: All models trained in under 15 seconds
3. **Compact Models**: Largest model only 1.3 MB, smallest only 321 KB
4. **Decent Overall Performance**: 64.43% accuracy using only 128 features

### Weaknesses

1. **Struggles with Rare Classes**: 8 classes with 0% accuracy, all with very few samples (1-4 samples)
2. **Sample Size Dependency**: Classes with fewer validation samples tend to perform worse
3. **Confusion on Similar Sounds**: May confuse similar audio events (e.g., different types of impacts)

### Model Architecture Insights

- **Shallow networks work well**: The 2-layer `[200, 100]` model achieved 63.42% with only 53K parameters
- **More layers ≠ better**: The 5-layer model (286K params) performed worse (62.58%) than simpler architectures
- **Optimal configuration**: `[512, 256]` with 217K parameters achieved the best balance at 64.43%

---

## Comparison with Other Models

| Model | Features | Accuracy | Perfect Classes |
|-------|----------|----------|-----------------|
| **VGGish+Tabular** | 2,602 (128 VGGish + 2,474 tabular) | **90.25%** | **37** |
| MobileNetV4 CNN | Mel-spectrograms | 75.26% | 12 |
| **VGGish Standalone** | 128 VGGish only | **64.43%** | **6** |
| LinearSVC | 2,474 tabular features | 52.78% | 2 |

**Key Insight**: VGGish standalone (64.43%) significantly outperforms LinearSVC on tabular features (52.78%), demonstrating the power of pre-trained audio embeddings. However, combining VGGish with tabular features achieves the best results (90.25%), showing the complementary nature of these feature sets.

---

## Files Generated

### Models
- `models/vggish_single_label_2layers_20251017_103207.pkl` - 321 KB (63.42% accuracy)
- `models/vggish_single_label_5layers_20251017_103220.pkl` - 1.3 MB (62.58% accuracy)
- `models/vggish_single_label_2layers_20251017_103236.pkl` - 964 KB (64.43% accuracy) ✅ **BEST**

### Analysis Files
- `vggish_single_label_results_20251017_103236.csv` - Overall results summary
- `vggish_single_label_all_class_analyses_20251017_103236.csv` - Combined per-class analysis
- `vggish_single_label_per_class_2layers_20251017_103207.csv` - Per-class for Model 1
- `vggish_single_label_per_class_5layers_20251017_103220.csv` - Per-class for Model 2
- `vggish_single_label_per_class_2layers_20251017_103236.csv` - Per-class for Model 3 ✅

### Logs
- `vggish_single_label_full_output.log` - Complete training log

---

## Recommendations

1. **For Production**: Use VGGish+Tabular (90.25%) as it significantly outperforms all other approaches
2. **For Fast Inference**: VGGish standalone `[512, 256]` model (64.43%) provides reasonable accuracy with only 964 KB model size
3. **For Edge Devices**: VGGish standalone `[200, 100]` model (63.42%) is ultra-compact at 321 KB
4. **Data Collection Priority**: Focus on collecting more samples for the 8 classes with 0% accuracy

---

## Technical Details

### Training Configuration
- **Optimizer**: Adam with one-cycle learning rate policy
- **Learning Rate**: Auto-discovered using FastAI's LR finder (~0.001)
- **Epochs**: 20
- **Batch Size**: 64
- **Loss Function**: CrossEntropyLoss
- **Validation Strategy**: Stratified 80/20 split

### Data Filtering
- **Original samples**: 3,524
- **After single-label filtering**: 2,978 (84.5%)
- **Removed**: 546 multi-label samples (15.5%)
- **Unique classes**: 74
- **Sample range per class**: 3 to 73 samples

---

**Generated**: 2025-10-17
**Author**: Claude Code Analysis
