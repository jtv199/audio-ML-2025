# VGGish + Tabular Features Training Notebook

## Overview

Created a comprehensive Jupyter notebook for training a FastAI tabular model that combines:
- **128 VGGish embeddings** (pre-trained audio features from Google's VGGish)
- **2,474 tabular features** (STFT means/stds, Mel spectrogram means/stds)
- **Total: 2,602 input features**

Training on **2,978 single-label samples** across **74 audio classes**.

## Notebook: vggish_tabular_training.ipynb

### Features

1. **Complete End-to-End Workflow**
   - Data loading and merging
   - Single-label filtering
   - Train/validation split (stratified)
   - Model creation
   - LR finder (critical for high-dimensional data!)
   - Training for 20 epochs
   - Evaluation and visualization
   - Model saving

2. **LR Finder Integration**
   - Automatically finds optimal learning rate
   - Critical for 2,602-dimensional input
   - Prevents NaN issues from bad LR selection
   - Plots LR curve for visual inspection

3. **Model Architecture**
   - 2-layer neural network: [200, 100]
   - Input: 2,602 features
   - Output: 74 classes (softmax)
   - Total parameters: ~553K

4. **Comprehensive Results**
   - Training history plots
   - Validation accuracy
   - Error rate
   - Comparison to random baseline
   - Sample misclassifications
   - Results saved to CSV

## How to Use

### Option 1: Run in Jupyter

```bash
# Activate conda environment
source ~/miniconda3/bin/activate freesound

# Start Jupyter
jupyter notebook vggish_tabular_training.ipynb
```

### Option 2: Convert to Python Script

```bash
# Convert notebook to Python
jupyter nbconvert --to script vggish_tabular_training.ipynb

# Run the script
python vggish_tabular_training.py
```

### Option 3: Run with papermill (automated)

```bash
papermill vggish_tabular_training.ipynb \
    vggish_tabular_output.ipynb \
    --log-output
```

## Key Improvements Over Previous Attempt

### Previous Script Issues:
- ❌ Used default/auto LR (3.63e-05) - too low!
- ❌ Hit NaN values during training
- ❌ Model never learned (1% accuracy)
- ❌ No visualization of training progress

### Notebook Solutions:
- ✅ **LR Finder** finds optimal learning rate
- ✅ Interactive - can manually adjust LR if needed
- ✅ Plots training curves to detect issues
- ✅ Comprehensive evaluation and analysis
- ✅ Better error handling
- ✅ Step-by-step execution with checkpoints

## Expected Results

Based on the VGGish-only model (66.44% accuracy), adding 2,474 more features should:

**Optimistic Scenario** (features help):
- Validation Accuracy: **70-75%**
- The tabular features provide complementary information
- Model captures patterns VGGish missed

**Realistic Scenario** (features neutral):
- Validation Accuracy: **60-70%**  
- Some features help, some add noise
- Still better than random (1.35%)

**Pessimistic Scenario** (features hurt):
- Validation Accuracy: **50-60%**
- Too many features cause overfitting
- Model struggles with dimensionality

## Important Notes

### 1. Learning Rate is Critical
With 2,602 features, the learning rate must be carefully chosen:
- Too low (< 1e-4): Model won't learn, may hit NaN
- Too high (> 1e-2): Gradients explode, definite NaN
- **Sweet spot**: Usually 1e-3 to 5e-3

The LR finder will suggest a value, but you can override it manually in cell 7:
```python
# If LR finder suggests something too low/high
optimal_lr = 1e-3  # or try 5e-3, 1e-2, etc.
```

### 2. Training Time
- Expected: **30-60 seconds** for 20 epochs
- Faster than CNN training
- Much faster than VGGish embedding extraction

### 3. Memory Requirements
- Features: 2,602 dimensions
- Model: ~553K parameters
- Should fit easily in CPU memory
- No GPU required

### 4. Model Interpretability
With 2,602 input features, it's hard to interpret what the model learned. Consider:
- Feature importance analysis
- Ablation studies (try VGGish-only vs Tabular-only)
- Dimensionality reduction (PCA) on tabular features

## Files Generated

After running the notebook:

1. **models/vggish_tabular_2layers_TIMESTAMP.pkl** - Trained model
2. **vggish_tabular_results_TIMESTAMP.csv** - Results summary
3. **Training plots** - Loss curves, accuracy progression

## Next Steps

If the combined model works well (>65% accuracy):

1. **Try Deeper Models**
   - 3 layers: [400, 200, 100]
   - 4 layers: [512, 256, 128, 64]

2. **Ensemble Approaches**
   - Train VGGish-only model
   - Train tabular-only model  
   - Ensemble their predictions

3. **Feature Selection**
   - Use only the most important tabular features
   - Reduce from 2,474 to top 500

4. **Hyperparameter Tuning**
   - Try different batch sizes
   - Experiment with dropout rates
   - Test various learning rates

## Troubleshooting

### Issue: Model hits NaN
**Solution**: Lower the learning rate
```python
optimal_lr = 1e-4  # or even 1e-5
```

### Issue: Accuracy stuck at ~1%
**Solution**: Increase the learning rate
```python
optimal_lr = 5e-3  # or 1e-2
```

### Issue: Slow training
**Solution**: Increase batch size
```python
# In cell 4, change bs=64 to:
bs=128  # or 256
```

### Issue: Overfitting (train acc >> val acc)
**Solutions**:
- Add dropout: `layers=[200, 100], ps=0.5`
- Use fewer epochs
- Add weight decay: `wd=0.01` in fit_one_cycle

## Comparison to Previous Approaches

| Approach | Features | Accuracy | Status |
|----------|----------|----------|--------|
| VGGish Multi-label | 128 | 1.67% | ❌ Too hard |
| VGGish Single-label | 128 | 66.44% | ✅ Good! |
| VGGish + Tabular (script) | 2,602 | 1% (NaN) | ❌ Bad LR |
| **VGGish + Tabular (notebook)** | **2,602** | **TBD** | ⏳ **To run** |

---

**Ready to train!** Open the notebook and execute cells sequentially.
