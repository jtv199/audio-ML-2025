# Model Comparison Findings

**Analysis Date:** October 16, 2025

## Quick Summary

Tested 22 sklearn classifiers on audio classification task with 74 classes and 2,474 features.

### Winner: LinearSVC (56.6% accuracy)

## Files in This Directory

1. **model_comparison_report.md** - Comprehensive analysis report
2. **model_comparison_results_20251016_093240.csv** - All 22 model results
3. **model_comparison_top10_20251016_093240.csv** - Top 10 performers

## Key Takeaways

- **Linear models dominate:** Top 3 are all linear classifiers
- **Ensemble methods failed:** AdaBoost (6.8%), GradientBoosting (timeout)
- **Best speed/accuracy:** RidgeClassifier (49.1% in 0.7s)
- **56.6% ceiling** suggests need for deep learning or better features

## Top 3 Recommended Models

1. **LinearSVC** - 56.6% accuracy, 137s training
2. **LogisticRegression** - 53.2% accuracy, 86s training
3. **RidgeClassifier** - 49.1% accuracy, 0.7s training (fastest)

## Next Steps

- Try LightGBM/XGBoost for gradient boosting
- Test neural networks on raw audio/spectrograms
- Implement multi-label classification for full dataset
- Apply feature scaling and dimensionality reduction

---

See **model_comparison_report.md** for detailed analysis.
