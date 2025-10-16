# Model Comparison Report: Audio Classification
**Date:** October 16, 2025
**Dataset:** FreeSound Audio Tagging 2019
**Task:** Multi-class audio classification (74 classes)

---

## Executive Summary

Evaluated 22 sklearn classifiers on tabular audio features (2,474 features) extracted from audio samples. **Linear models significantly outperformed tree-based and ensemble methods**, with LinearSVC achieving the best accuracy of **56.6%**.

---

## Dataset Characteristics

- **Total Samples:** 4,970 audio clips
- **Single-Label Samples:** 4,269 (85.9%)
- **Features:** 2,474 tabular audio features per sample
- **Classes:** 74 unique sound categories
- **Train/Test Split:** 80/20 (3,414 train / 854 test)
- **Data Quality:** 1 missing value per feature (removed)

---

## Model Performance Rankings

### Top 10 Models

| Rank | Model | Accuracy | Balanced Acc | F1 Score | Time (s) |
|------|-------|----------|--------------|----------|----------|
| 1 | **LinearSVC** | **56.6%** | 52.8% | 55.1% | 137 |
| 2 | **CalibratedClassifierCV** | **56.0%** | 51.9% | 53.9% | 350 |
| 3 | **LogisticRegression** | **53.2%** | 49.2% | 51.6% | 86 |
| 4 | RidgeClassifier | 49.1% | 44.1% | 45.8% | 0.7 |
| 5 | RandomForestClassifier | 47.9% | 43.7% | 46.4% | 4.3 |
| 6 | ExtraTreesClassifier | 47.8% | 44.1% | 46.3% | 1.1 |
| 7 | RidgeClassifierCV | 46.0% | 41.2% | 42.0% | 21 |
| 8 | KNeighborsClassifier | 39.2% | 35.6% | 38.5% | 0.3 |
| 9 | BaggingClassifier | 39.0% | 35.9% | 37.7% | 30 |
| 10 | PassiveAggressiveClassifier | 36.1% | 32.4% | 34.8% | 25 |

---

## Key Findings

### 1. Linear Models Dominate
- **Top 3 models are all linear classifiers**
- LinearSVC and LogisticRegression excel on high-dimensional tabular audio features
- Linear models are 10-15% more accurate than tree-based methods

### 2. Ensemble Methods Underperform
- **AdaBoostClassifier:** Only 6.8% accuracy (worst performer)
- **GradientBoostingClassifier:** Timed out after 15 minutes
- **BaggingClassifier:** 39.0% accuracy (8th place)
- Ensemble methods likely overfit due to high dimensionality

### 3. Speed vs Accuracy Trade-offs

**Fast & Accurate (Best Value):**
- RidgeClassifier: 49.1% accuracy in 0.7s
- LogisticRegression: 53.2% accuracy in 86s
- RandomForestClassifier: 47.9% accuracy in 4.3s

**Slow but Top Performers:**
- LinearSVC: 56.6% accuracy in 137s
- CalibratedClassifierCV: 56.0% accuracy in 350s

### 4. Model Failures
- **GradientBoostingClassifier:** Timeout (>15 minutes)
- **QuadraticDiscriminantAnalysis:** 3.9% accuracy (too complex for data)
- **AdaBoostClassifier:** 6.8% accuracy (weak learners fail on high-dim data)

---

## Performance by Model Category

### Segment 1: Fast Models (5 models)
- **Best:** Perceptron (31.6%)
- **Fastest:** DummyClassifier (0.01s baseline)
- **Finding:** Simple models struggle with 74-class problem

### Segment 2: Linear Models (8 models)
- **Best:** LinearSVC (56.6%)
- **Runner-up:** LogisticRegression (53.2%)
- **Finding:** Linear models excel on high-dimensional features

### Segment 3: Tree & Neighbor Models (5 models)
- **Best:** RandomForestClassifier (47.9%)
- **Finding:** Trees overfit; ensembles of 100 trees don't help

### Segment 4: Ensemble Models (4 models)
- **Best:** CalibratedClassifierCV (56.0%)
- **Worst:** AdaBoostClassifier (6.8%)
- **Finding:** Complex ensembles fail on this feature space

---

## Recommendations

### For Production Use
1. **LinearSVC** - Best accuracy (56.6%), reasonable speed (137s)
2. **LogisticRegression** - Excellent accuracy (53.2%), faster training (86s)
3. **RidgeClassifier** - Fast baseline (0.7s), good accuracy (49.1%)

### For Further Improvement
1. **Feature Engineering:**
   - Current features may not capture temporal audio patterns
   - Consider deep learning on raw audio (CNNs, RNNs)
   - Explore mel-spectrograms, MFCCs with neural networks

2. **Hyperparameter Tuning:**
   - LinearSVC: Tune C parameter (regularization)
   - LogisticRegression: Increase max_iter to avoid convergence warnings
   - RandomForest: Reduce n_estimators to prevent overfitting

3. **Advanced Models to Test:**
   - **LightGBM:** Faster gradient boosting optimized for high-dimensional data
   - **XGBoost:** Alternative gradient boosting with better regularization
   - **Neural Networks:** Multi-layer perceptrons may capture non-linear patterns

4. **Multi-Label Classification:**
   - Current analysis uses only 85.9% of data (single-label samples)
   - 14.1% of samples have multiple labels
   - Use `OneVsRestClassifier` or `MultiOutputClassifier` wrappers

---

## Technical Notes

### Training Environment
- **Runtime:** 29 minutes 15 seconds
- **Models Tested:** 22 classifiers
- **Success Rate:** 95.5% (21/22 completed)
- **Timeout:** 1 model (GradientBoostingClassifier)

### Data Preprocessing
- Removed 1 row with missing values per feature
- No feature scaling applied (linear models may benefit from StandardScaler)
- No dimensionality reduction (PCA/UMAP could help)

### Evaluation Metrics
- **Accuracy:** Overall correctness
- **Balanced Accuracy:** Accounts for class imbalance
- **F1 Score:** Weighted average (harmonic mean of precision/recall)
- **No ROC AUC:** Not computed for multiclass

---

## Conclusions

1. **Linear models are the clear winners** for tabular audio features in this dataset
2. **High dimensionality (2,474 features)** favors simple linear classifiers over complex ensembles
3. **56.6% accuracy ceiling** suggests either:
   - Features are insufficient for this task
   - 74-class problem is inherently difficult
   - Deep learning on raw audio may perform better

4. **Next steps should focus on:**
   - Feature engineering and selection
   - Deep learning approaches (CNNs on spectrograms)
   - Multi-label classification for complete dataset

---

## Appendix: Detailed Results

### All Model Results (21 Successful Models)

| Model | Accuracy | Balanced Acc | F1 Score | Time (s) | Status |
|-------|----------|--------------|----------|----------|--------|
| LinearSVC | 56.6% | 52.8% | 55.1% | 137.0 | Success |
| CalibratedClassifierCV | 56.0% | 51.9% | 53.9% | 349.8 | Success |
| LogisticRegression | 53.2% | 49.2% | 51.6% | 85.7 | Success |
| RidgeClassifier | 49.1% | 44.1% | 45.8% | 0.7 | Success |
| RandomForestClassifier | 47.9% | 43.7% | 46.4% | 4.3 | Success |
| ExtraTreesClassifier | 47.8% | 44.1% | 46.3% | 1.1 | Success |
| RidgeClassifierCV | 46.0% | 41.2% | 42.0% | 20.7 | Success |
| KNeighborsClassifier | 39.2% | 35.6% | 38.5% | 0.3 | Success |
| BaggingClassifier | 39.0% | 35.9% | 37.7% | 30.3 | Success |
| PassiveAggressiveClassifier | 36.1% | 32.4% | 34.8% | 24.9 | Success |
| SGDClassifier | 36.1% | 32.9% | 33.8% | 39.2 | Success |
| Perceptron | 31.6% | 27.6% | 28.9% | 14.3 | Success |
| LinearDiscriminantAnalysis | 24.8% | 22.8% | 23.5% | 22.6 | Success |
| DecisionTreeClassifier | 24.8% | 22.6% | 24.3% | 30.8 | Success |
| ExtraTreeClassifier | 23.0% | 20.9% | 21.6% | 0.1 | Success |
| GaussianNB | 19.8% | 18.5% | 16.7% | 1.0 | Success |
| NearestCentroid | 19.7% | 18.6% | 18.1% | 0.2 | Success |
| AdaBoostClassifier | 6.8% | 5.8% | 4.5% | 84.6 | Success |
| BernoulliNB | 4.7% | 3.8% | 2.1% | 0.3 | Success |
| QuadraticDiscriminantAnalysis | 3.9% | 4.2% | 2.1% | 2.7 | Success |
| DummyClassifier | 1.8% | 1.4% | 0.1% | 0.01 | Success |
| GradientBoostingClassifier | - | - | - | 900.0 | **Timeout** |

---

**Report Generated:** 2025-10-16 09:36
**Script Location:** `/claude/model_comparison.py`
**CSV Results:** `/claude/model_comparison_results_20251016_093240.csv`
