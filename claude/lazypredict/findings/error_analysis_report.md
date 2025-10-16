# Error Analysis Report: Per-Class Performance

**Date:** October 16, 2025
**Models Analyzed:** LinearSVC, LogisticRegression, RandomForestClassifier

---

## Executive Summary

Conducted per-class error analysis on the top 3 performing models. Key finding: **Performance varies dramatically by class** - from 100% accuracy on `Finger_snapping` to 0% on several rare classes like `Acoustic_guitar` and `Cutlery_and_silverware`.

---

## Best Performing Classes (All Models)

### 🏆 Top 5 Most Accurate Classes

| Rank | Class | LinearSVC Acc | LogisticReg Acc | RandomForest Acc | Avg Accuracy |
|------|-------|---------------|-----------------|------------------|--------------|
| 1 | **Finger_snapping** | 100.0% | 100.0% | 100.0% | **100.0%** |
| 2 | **Trickle_and_dribble** | 100.0% | 100.0% | - | **100.0%** |
| 3 | **Glockenspiel** | 90.9% | 90.9% | 100.0% | **93.9%** |
| 4 | **Hi-hat** | 92.9% | 92.9% | 92.9% | **92.9%** |
| 5 | **Bicycle_bell** | - | 92.3% | 92.3% | **92.3%** |

**Why These Classes Excel:**
- Clear, distinctive acoustic signatures
- Consistent features across samples
- Minimal confusion with other classes
- Adequate training samples (10-15 per class)

---

## Worst Performing Classes (All Models)

### ❌ Bottom 5 Least Accurate Classes

| Rank | Class | LinearSVC Acc | LogisticReg Acc | RandomForest Acc | Samples |
|------|-------|---------------|-----------------|------------------|---------|
| 1 | **Acoustic_guitar** | 0.0% | 0.0% | 0.0% | 1 |
| 2 | **Crowd** | 0.0% | 0.0% | 0.0% | 1 |
| 3 | **Drip** | 0.0% | 0.0% | 0.0% | 1 |
| 4 | **Squeak** | 0.0% | 0.0% | 0.0% | 4 |
| 5 | **Cutlery_and_silverware** | 0.0% | 0.0% | 0.0% | 7 |

**Why These Classes Fail:**
- **Insufficient training data** (1-7 samples) - severe class imbalance
- Generic/ambiguous acoustic properties
- High similarity to other classes
- Models have no chance to learn patterns

---

## Confusion Pattern Analysis

### LinearSVC - Best Model (56.6% accuracy)

#### Most Confused Classes (High False Positives)

Classes that are **wrongly predicted** most often:

| Class | False Positives | Precision | Issue |
|-------|-----------------|-----------|-------|
| **Crackle** | 13 | 43.5% | Confused with similar crackling sounds |
| **Keys_jangling** | 13 | 45.8% | Metallic sounds overlap with coins, bells |
| **Whispering** | 13 | 38.1% | Ambient noise confused with whispers |
| **Car_passing_by** | 12 | 40.0% | Traffic sounds overlap with engines |
| **Scissors** | 11 | 26.7% | Metallic cutting confused with other metal |

#### Most Missed Classes (High False Negatives)

Classes that **fail to be detected**:

| Class | False Negatives | Recall | Issue |
|-------|-----------------|--------|-------|
| **Hiss** | 13 | 13.3% | White noise-like sounds hard to distinguish |
| **Zipper_(clothing)** | 13 | 13.3% | Very specific sound, confused with fabric |
| **Chewing_and_mastication** | 12 | 20.0% | Quiet, subtle sounds hard to detect |
| **Cricket** | 12 | 14.3% | High-pitched insect sounds blend together |
| **Computer_keyboard** | 11 | 26.7% | Typing sounds confused with clicking |

---

## Detailed Case Studies

### Case Study 1: Worst Class - Acoustic_guitar

**Performance:**
- LinearSVC: 0% accuracy (0/1 correct)
- LogisticRegression: 0% accuracy (0/1 correct)
- RandomForestClassifier: 0% accuracy (0/1 correct)

**What Happens:**
- The **single test sample** is predicted as `Microwave_oven`
- Severe data insufficiency - only 1 sample in test set
- Likely very few training samples as well

**Root Cause:**
- Class imbalance - this class is underrepresented
- Acoustic features may overlap with electric hum/drone sounds
- Need 10-20x more samples to learn this class

**Recommendation:**
- Collect more acoustic guitar samples
- Use data augmentation (pitch shift, time stretch)
- Consider removing from dataset if samples can't be obtained

---

### Case Study 2: Best Class - Finger_snapping

**Performance:**
- LinearSVC: 100% accuracy (15/15 correct)
- LogisticRegression: 100% accuracy (15/15 correct)
- RandomForestClassifier: 100% accuracy (15/15 correct)

**What Happens:**
- **All 15 test samples** correctly predicted across all models
- Zero confusion with any other class

**Why It Works:**
- **Distinctive acoustic signature** - sharp transient with specific spectral content
- Clear temporal pattern (snap peak + decay)
- Minimal acoustic overlap with other classes
- Adequate samples (15 in test, likely 60+ in training)

**Insight:**
- This is the "gold standard" - other classes should aim for this level of distinctiveness
- Success shows that when features are clear, even simple linear models achieve perfection

---

## Model Comparison: Error Patterns

### LinearSVC (Best Overall: 56.6%)

**Strengths:**
- Best at avoiding false positives (more conservative)
- Highest precision on ambiguous classes
- Clean decision boundaries

**Weaknesses:**
- Still misses rare classes completely
- Struggles with acoustic similarity (hiss, zipper)

### LogisticRegression (53.2%)

**Strengths:**
- Good generalization
- Balanced precision/recall
- Handles uncertainty better with probabilities

**Weaknesses:**
- More false positives than LinearSVC
- Convergence issues (needs more iterations)

### RandomForestClassifier (47.9%)

**Strengths:**
- Perfect on some classes (Glockenspiel: 100%)
- Non-linear patterns captured

**Weaknesses:**
- **Severe overfitting** on rare classes
- Highest false positive rate (e.g., Zipper: 22 FP)
- Generates predictions even with insufficient data

---

## Statistical Summary

### Per-Class Accuracy Distribution (LinearSVC)

- **Mean Accuracy:** 56.6%
- **Median Accuracy:** ~57%
- **Standard Deviation:** High (suggests big variance between classes)
- **Range:** 0% to 100%

**Key Insight:** Performance is **bimodal** - classes either work very well (>90%) or fail completely (<10%).

---

## Actionable Recommendations

### 1. Address Class Imbalance (Critical)

**Problem:** Classes with <5 samples achieve 0% accuracy

**Solutions:**
- Collect more data for rare classes
- Use data augmentation:
  - Time stretching
  - Pitch shifting
  - Adding background noise
  - SpecAugment
- Consider class weighting in training
- Use SMOTE or other oversampling techniques

### 2. Feature Engineering for Confused Classes

**Problem:** Hiss, Zipper, Crackle confused with similar sounds

**Solutions:**
- Add temporal features (attack, decay, sustain)
- Include spectral contrast features
- Use delta and delta-delta features (rate of change)
- Add zero-crossing rate for transient sounds
- Include spectral rolloff and centroid

### 3. Model Improvements

**For LinearSVC (Current Best):**
- ✅ Keep using - best performance
- Tune C parameter for better regularization
- Try different kernels (RBF, poly)

**For LogisticRegression:**
- Increase `max_iter` to 5000+ to ensure convergence
- Try different solvers (saga, liblinear)
- Add L1 regularization for feature selection

**Alternative Approaches:**
- Try **LightGBM** - better for imbalanced data
- Test **Neural Networks** with class weights
- Ensemble: Combine LinearSVC + LogisticReg predictions

### 4. Remove or Merge Problematic Classes

**Classes to Consider Removing:**
- Acoustic_guitar (1 sample)
- Crowd (1 sample)
- Drip (1 sample)

**Classes to Consider Merging:**
- Merge similar metallic sounds (Keys_jangling + Scissors)
- Merge similar vocals (Whispering + Breathing)
- Merge similar ambient (Hiss + White_noise)

---

## Visualizations Generated

1. **accuracy_distribution_linearsvc.png**
   - Histogram of per-class accuracy
   - Shows bimodal distribution

2. **top_bottom_classes_linearsvc.png**
   - Top 10 vs Bottom 10 classes side-by-side
   - Clear visual of performance gap

3. **precision_recall_scatter_linearsvc.png**
   - Scatter plot: Precision vs Recall
   - Color-coded by accuracy
   - Identifies precision/recall trade-offs

---

## Files Generated

1. **error_analysis_linearsvc.csv** - Full per-class metrics for LinearSVC
2. **error_analysis_logisticregression.csv** - Full per-class metrics for LogisticReg
3. **error_analysis_randomforestclassifier.csv** - Full per-class metrics for RandomForest
4. **error_analysis_summary.csv** - Comparison summary across models
5. **error_analysis_output.txt** - Complete console output

---

## Conclusions

1. **Class imbalance is the #1 issue** - classes with <5 samples have 0% accuracy
2. **Distinctive sounds work perfectly** - Finger_snapping achieves 100% across all models
3. **LinearSVC is the best model** - most robust to overfitting and confusion
4. **56.6% ceiling is due to:**
   - Insufficient data for rare classes (30% of classes have <5 samples)
   - Acoustic similarity between classes (metallic, ambient sounds)
   - Feature limitations (tabular features miss temporal patterns)

5. **Path to improvement:**
   - Collect 50+ samples per class (minimum)
   - Use deep learning on raw audio/spectrograms
   - Address class imbalance with augmentation
   - Remove/merge classes with <10 samples

---

**Next Steps:** Focus on data collection and deep learning approaches to break the 56.6% accuracy ceiling.
