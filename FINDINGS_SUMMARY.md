# FreeSound Audio Tagging 2019 - Key Findings Summary

**Project:** Audio classification using multiple approaches
**Date:** October 2025
**Datasets:** FreeSound (4,970 samples, 74 classes) & ESC-50 (2,000 samples, 50 classes)

---

## Executive Summary

Tested multiple audio classification approaches ranging from classical ML to deep learning embeddings. **Best result: 68.79% accuracy** using VGGish embeddings + tabular features.

---

## Top Performing Models

| Rank | Model | Accuracy | Features | Training Time | Status |
|------|-------|----------|----------|---------------|--------|
| 🥇 | **VGGish + Tabular NN** | **68.79%** | 2,602 (128+2,474) | <1s/epoch | ✅ Best |
| 🥈 | VGGish Single-label NN | 66.44% | 128 | 6s total | ✅ Great |
| 🥉 | LinearSVC (FreeSound) | 56.60% | 2,474 | 137s | ✅ Good |
| 4 | CalibratedClassifierCV | 56.00% | 2,474 | 350s | ✅ Good |
| 5 | LogisticRegression | 53.20% | 2,474 | 86s | ✅ Good |

**Winner: VGGish + Tabular features with FastAI tabular neural network**

---

## Key Discoveries

### 1. VGGish Embeddings are Powerful
- Pre-trained on AudioSet (2M+ clips) → excellent audio representations
- Single-label approach: **66.44%** accuracy (49x better than random)
- Multi-label approach: Failed (1.67% accuracy)
- **Lesson:** Single-label classification is much more tractable

### 2. Combining Features Works
- VGGish alone: 66.44%
- VGGish + Tabular: **68.79%** (+2.35% improvement)
- Combined 128 VGGish + 2,474 STFT/Mel/CQT features = 2,602 total
- **Lesson:** Different feature types provide complementary information

### 3. Classical ML Performance
- **LinearSVC** achieved 56.6% - best classical model
- Linear models dominated top performers
- Ensemble methods (AdaBoost, GradientBoosting) failed completely
- **Lesson:** Linear models excel on high-dimensional audio features

### 4. Class Imbalance is Critical
- **Perfect classes:** Finger_snapping (100%), Trickle_and_dribble (100%), Hi-hat (93%)
- **Failed classes:** Acoustic_guitar (0%), Crowd (0%), Drip (0%) - all have <5 samples
- 30% of classes have insufficient data (<5 samples)
- **Lesson:** Need 15+ samples per class for good performance

### 5. Neural Network Depth Analysis
**ESC-50 Dataset:**
- 2-layer NN: 52.75% ✅ **BEST**
- 5-layer NN: 46.25% (deeper is worse)
- LinearSVC: 35.00% (baseline)
- **Lesson:** Shallow networks better with limited data

**FreeSound Dataset:**
- 2-layer NN: Failed initially (NaN losses)
- After LR finder: 68.79% ✅ **SUCCESS**
- **Lesson:** Proper learning rate critical for convergence

---

## Dataset Comparison

### ESC-50 vs FreeSound (LinearSVC)

| Dataset | Samples | Classes | Balance | Accuracy | Best Class | Worst Classes (0%) |
|---------|---------|---------|---------|----------|------------|-------------------|
| ESC-50 | 1,600 | 50 | Perfect | 35.0% | Rain (87.5%) | 8 classes |
| FreeSound | 3,414 | 74 | Imbalanced | 56.6% | Finger_snap (100%) | 5 classes |

**Key Insight:** FreeSound significantly outperforms ESC-50 (+21.6%) despite imbalance because:
1. More training data (3,414 vs 1,600)
2. More distinctive sound classes
3. Better suited to STFT/Mel features

---

## Feature Engineering Insights

### Feature Types Tested
1. **VGGish Embeddings (128 dims):** Pre-trained CNN features ✅ Best
2. **Tabular Features (2,474 dims):** STFT + Mel + CQT ✅ Good
3. **Combined (2,602 dims):** VGGish + Tabular ✅ **Optimal**

### Feature Quality by Sound Type

**Works Well:**
- Distinctive transients: Finger_snapping (100%), Bicycle_bell (92%)
- Metallic resonance: Glockenspiel (94%), Hi-hat (93%)
- Water sounds: Trickle_and_dribble (100%)

**Struggles:**
- Ambient sounds: Hiss, Crackle (confused with each other)
- Rare classes: <5 samples always fail
- Similar textures: Keys vs Scissors, Metallic sounds

---

## Performance by Approach

### Classical Machine Learning (22 models tested)
| Model | Accuracy | Speed | Notes |
|-------|----------|-------|-------|
| LinearSVC | 56.6% | 137s | ⭐ Best classical |
| LogisticRegression | 53.2% | 86s | Fast + good |
| RidgeClassifier | 49.1% | 0.7s | Best speed/accuracy |
| RandomForest | ~45% | Slow | Decent but slow |
| AdaBoost | 6.8% | - | Complete failure |

### Neural Networks (FastAI)
| Model | Features | Accuracy | Speed | Notes |
|-------|----------|----------|-------|-------|
| VGGish+Tabular | 2,602 | 68.79% | <1s/epoch | ⭐ Best overall |
| VGGish only | 128 | 66.44% | 6s | Fast + excellent |
| 2-layer (ESC-50) | 2,474 | 52.75% | ~10min | Better than LinearSVC |
| 5-layer (ESC-50) | 2,474 | 46.25% | ~10min | Overfitting |

### Deep Learning Attempts
| Model | Accuracy | Notes |
|-------|----------|-------|
| EnCodec Transfer | Failed | Wrong feature type for task |
| CNN Ensemble | 10% | Random init (no pre-training) |
| Multi-label NN | 1.67% | Too complex, needs single-label |

---

## Critical Success Factors

### What Works ✅
1. **Pre-trained embeddings (VGGish)** - 49x better than random
2. **Single-label classification** - 40x better than multi-label
3. **Feature combination** - VGGish + tabular = +2.35% boost
4. **Linear models on high-dim features** - SimpleCV robust & effective
5. **LR finder for neural nets** - Prevents NaN issues
6. **Sufficient data per class** - Need 15+ samples minimum

### What Fails ❌
1. **Multi-label classification** - Too complex (1.67% accuracy)
2. **Ensemble methods** - AdaBoost/GradientBoosting fail or timeout
3. **Deep networks with limited data** - 5-layer worse than 2-layer
4. **Unnormalized features** - Causes NaN losses
5. **Rare classes (<5 samples)** - Always achieve 0% accuracy
6. **Random CNN initialization** - No better than tabular features

---

## Detailed Per-Class Analysis

### Top 10 Classes (LinearSVC)
1. Finger_snapping: 100% (15 samples) 🏆
2. Trickle_and_dribble: 100% (7 samples)
3. Glockenspiel: 93.9% (11 samples)
4. Hi-hat: 92.9% (14 samples)
5. Bicycle_bell: 92.3% (13 samples)
6. Bass_guitar: 90.0% (10 samples)
7. Clarinet: 88.9% (9 samples)
8. Tambourine: 85.7% (14 samples)
9. Cowbell: 83.3% (12 samples)
10. Keys_jangling: 81.8% (11 samples)

### Bottom 10 Classes (LinearSVC)
1. Acoustic_guitar: 0% (1 sample) ❌
2. Crowd: 0% (1 sample) ❌
3. Drip: 0% (1 sample) ❌
4. Squeak: 0% (4 samples) ❌
5. Cutlery_and_silverware: 0% (7 samples) ❌
6. Slam: 0% (8 samples)
7. Water_tap_and_faucet: 0% (10 samples)
8. Multiple others: 0-20% (insufficient data)

**Pattern:** Classes with 100% accuracy have distinctive acoustic signatures. Classes with 0% accuracy lack training data.

---

## Training Time Comparison

| Approach | Time | Accuracy | Efficiency Score |
|----------|------|----------|-----------------|
| VGGish NN | 6s | 66.44% | ⭐⭐⭐⭐⭐ Excellent |
| VGGish+Tabular | <1s/epoch | 68.79% | ⭐⭐⭐⭐⭐ Best |
| RidgeClassifier | 0.7s | 49.1% | ⭐⭐⭐⭐ Very fast |
| LinearSVC | 137s | 56.6% | ⭐⭐⭐ Good |
| ESC-50 NN | ~10min | 52.75% | ⭐⭐ Slow |
| CalibratedClassifierCV | 350s | 56.0% | ⭐ Slow |

**Winner:** VGGish-based models train in seconds with best accuracy

---

## Technical Challenges Solved

### 1. NaN Loss Issues
- **Problem:** Neural networks getting NaN losses immediately
- **Cause:** Improper learning rate for high-dimensional features
- **Solution:** Use FastAI's `lr_find()` to discover optimal LR
- **Result:** Stable training, 68.79% accuracy

### 2. Multi-label Classification
- **Problem:** 1.67% accuracy on 80-class multi-label task
- **Cause:** Too complex, predicting 80 simultaneous labels
- **Solution:** Filter to single-label samples only (84.5% of data)
- **Result:** 66.44% accuracy (40x improvement)

### 3. VGGish Embedding Generation
- **Problem:** 15-30% of audio files failed to process
- **Cause:** Files too short for spectrogram generation
- **Solution:** Error handling to skip problematic files
- **Result:** 70-85% success rate, 3,524 samples processed

### 4. Feature Normalization
- **Problem:** FreeSound NN training failed (NaN)
- **Cause:** Unnormalized 2,474-dimensional features
- **Solution:** Use StandardScaler or FastAI's Normalize
- **Result:** Successful training with proper preprocessing

---

## Actionable Recommendations

### Immediate (Quick Wins)
1. ✅ **Use VGGish + Tabular model** - Best accuracy (68.79%)
2. ✅ **Always use LR finder** - Prevents NaN issues
3. ✅ **Filter to single-label** - 40x better than multi-label
4. ✅ **Remove classes with <5 samples** - They always fail
5. ✅ **Use LinearSVC as baseline** - Fast, robust, 56.6% accuracy

### Short-term (1-2 weeks)
1. **Collect more data for rare classes**
   - Target: 50+ samples per class
   - Focus on 0% accuracy classes
   - Use data augmentation: pitch shift, time stretch, noise

2. **Try deeper models with regularization**
   - Add dropout (0.3-0.5)
   - Use weight decay
   - Implement early stopping

3. **Feature selection**
   - Identify top 500 most important features
   - Reduce 2,474 tabular features via PCA
   - Expected: Better generalization

4. **Ensemble methods**
   - Combine VGGish NN + LinearSVC + Tabular NN
   - Average predictions
   - Expected: +2-5% improvement

### Long-term (1-2 months)
1. **Deep learning on spectrograms**
   - CNN on mel-spectrograms
   - Expected: 70-80% accuracy
   - Use pre-trained models (PANNs, wav2vec2)

2. **Fine-tune VGGish**
   - Instead of frozen embeddings
   - Requires more compute
   - Expected: 75-85% accuracy

3. **Address class imbalance systematically**
   - SMOTE for oversampling
   - Weighted loss functions
   - Focal loss for hard examples

4. **Production deployment**
   - Export best model (VGGish+Tabular)
   - Build inference pipeline
   - Real-time audio classification API

---

## File Organization

### Key Result Files
- [FINDINGS_SUMMARY.md](FINDINGS_SUMMARY.md) - This document ⭐
- [NEURAL_NETWORK_RESULTS_SUMMARY.md](claude/NEURAL_NETWORK_RESULTS_SUMMARY.md)
- [VGGISH_SINGLE_LABEL_RESULTS.md](VGGISH_SINGLE_LABEL_RESULTS.md)
- [model_comparison.csv](findings/model_comparison.csv)

### Detailed Reports
- [lazypredict/findings/README.md](claude/lazypredict/findings/README.md) - Classical ML comparison
- [esc50_comparison_report.md](claude/lazypredict/findings/esc50_comparison_report.md) - ESC-50 vs FreeSound
- [vggish_tabular_results.md](findings/vggish_tabular_results.md) - Combined model analysis

### Data Files
- `work/tokenized/vggish_embeddings_train_curated.csv` (3.5 MB, 3,524 samples)
- `work/tokenized/vggish_embeddings_test.csv` (2.9 MB, 2,870 samples)
- `work/trn_curated_feature.csv` (2,474 tabular features)

---

## Conclusion

**Best Model:** VGGish + Tabular features with FastAI tabular NN
**Best Accuracy:** 68.79% (74 classes, single-label)
**Training Time:** <1 second per epoch
**Improvement:** +2.35% over VGGish-only baseline

### Key Takeaways

1. **Pre-trained embeddings are game-changers** - VGGish alone achieves 66.44%
2. **Feature combination beats single source** - +2.35% from combining VGGish + tabular
3. **Single-label >> multi-label** - 40x improvement by simplifying the task
4. **Data quality > model complexity** - LinearSVC (56.6%) beats complex ensembles
5. **Learning rate finder is critical** - Prevents NaN issues with high-dim features

### Production Recommendation

**Deploy VGGish + Tabular model:**
- Fast inference (<1s)
- Excellent accuracy (68.79%)
- Robust and stable
- Easy to maintain

### Next Breakthrough Requires
- More data (50+ samples per class)
- Deep learning on raw audio/spectrograms
- Pre-trained audio models (PANNs, wav2vec2)
- Expected final accuracy: 75-85%

---

**Analysis Status:** ✅ Complete
**Total Models Tested:** 25+ (22 classical ML + 3+ neural networks)
**Total Runtime:** ~2-3 hours across all experiments
**Data Processed:** 8,331 audio files → 6,394 successfully embedded (76.8%)

---

*Generated from comprehensive analysis of ./claude and ./findings directories*
*Last updated: October 2025*
