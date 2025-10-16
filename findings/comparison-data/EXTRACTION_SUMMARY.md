
# Training Metrics Extraction Summary

## Files Generated

1. **training_metrics_extracted.csv** (41KB)
   - Complete dataset with 528 epoch records across all 5 notebooks
   - Columns: accuracy_multi, epoch, learning_rate, lwlrap, model, pretrained, time, train_loss, training_phase, valid_loss

2. **training_metrics_summary.md** (20KB)
   - Detailed markdown report with training progression tables for each model
   - Shows first 10 epochs and final epoch of each training phase

## Models Analyzed

### 1. ResNet18 (Baseline)
- **File**: cnn-2d-basic-solution-powered-by-fast-ai.ipynb
- **Architecture**: resnet18
- **Pretrained**: No
- **Total Epochs**: 115
- **Training Phases**: 5
  - Phase 1: 15 epochs @ lr=1e-1
  - Phase 2: 20 epochs @ lr=3e-3
  - Phase 3: 20 epochs @ lr=1e-3
  - Phase 4: 50 epochs @ lr=slice(1e-3, 3e-3)
  - Phase 5: 10 epochs @ lr=slice(1e-4, 1e-3)
- **Best LWLRAP**: 0.722973
- **Final LWLRAP**: 0.711425
- **Final Valid Loss**: 0.034925

### 2. MobileNetV4 Conv Small
- **File**: mobilnet-test.ipynb
- **Architecture**: mobilenetv4_conv_small
- **Pretrained**: No
- **Total Epochs**: 90
- **Training Phases**: 5
  - Phase 1: 10 epochs @ lr=1e-1
  - Phase 2: 5 epochs @ lr=3e-3
  - Phase 3: 20 epochs @ lr=1e-3
  - Phase 4: 50 epochs @ lr=slice(1e-3, 3e-3)
  - Phase 5: 5 epochs @ lr=slice(1e-4, 1e-3)
- **Best LWLRAP**: 0.654780
- **Best Accuracy**: 0.989499
- **Final LWLRAP**: 0.650506
- **Final Accuracy**: 0.989311
- **Final Valid Loss**: 0.043148

### 3. DeiT Tiny (Transfer Learning)
- **File**: vision-transformer-transfer.ipynb
- **Architecture**: deit_tiny_patch16_224
- **Pretrained**: Yes (ImageNet)
- **Total Epochs**: 106
- **Training Phases**: 5
- **Best LWLRAP**: 0.235876
- **Best Accuracy**: 0.985488
- **Final LWLRAP**: 0.235876
- **Final Accuracy**: 0.985463
- **Final Valid Loss**: 0.066228

### 4. DeiT Tiny (No Transfer)
- **File**: vision-transformer-no-transfer.ipynb
- **Architecture**: deit_tiny_patch16_224
- **Pretrained**: No
- **Total Epochs**: 115
- **Training Phases**: 5
- **Best LWLRAP**: 0.259623
- **Best Accuracy**: 0.985501
- **Final LWLRAP**: 0.253036
- **Final Accuracy**: 0.985475
- **Final Valid Loss**: 0.064540

### 5. ConViT Tiny
- **File**: vision-transformer-cvit-smaller.ipynb
- **Architecture**: convit_tiny
- **Pretrained**: No
- **Total Epochs**: 102
- **Training Phases**: 5
- **Best LWLRAP**: 0.259623
- **Best Accuracy**: 0.985488
- **Final LWLRAP**: 0.253036
- **Final Accuracy**: 0.985475
- **Final Valid Loss**: 0.064540

## Key Findings

### Performance Ranking (by LWLRAP)
1. **ResNet18**: 0.711 - WINNER
2. **MobileNetV4**: 0.651
3. **DeiT Tiny (No Transfer)**: 0.253
4. **ConViT Tiny**: 0.253
5. **DeiT Tiny (Transfer)**: 0.236 - WORST

### Key Insights

1. **CNN architectures significantly outperform Vision Transformers** for this audio classification task
   - ResNet18: 0.711 LWLRAP
   - MobileNetV4: 0.651 LWLRAP
   - Best ViT: 0.259 LWLRAP (2.8x worse!)

2. **Transfer learning from ImageNet HURT performance**
   - DeiT with pretrained weights: 0.236 LWLRAP
   - DeiT without pretrained weights: 0.253 LWLRAP
   - 7% performance degradation

3. **LWLRAP vs Accuracy disconnect**
   - MobileNetV4 has highest accuracy (98.93%) but lower LWLRAP (0.651)
   - Vision transformers have high accuracy (~98.5%) but terrible LWLRAP (~0.25)
   - This suggests the models are predicting many classes with similar confidence

4. **Training efficiency**
   - ResNet18: 115 epochs, ~12 minutes total
   - MobileNetV4: 90 epochs, faster per epoch
   - Vision Transformers: More epochs but didn't converge well

5. **All models improved dramatically from first to final epoch**
   - ResNet18: +0.631 LWLRAP improvement
   - MobileNetV4: +0.396 LWLRAP improvement
   - DeiT (no transfer): +0.182 LWLRAP improvement

## Metrics Explanation

- **LWLRAP** (Label Weighted Label Ranking Average Precision): Primary competition metric for multi-label audio classification
- **Accuracy Multi**: Multi-label accuracy - percentage of exactly correct label predictions
- **Train Loss**: Binary cross-entropy loss on training set
- **Valid Loss**: Binary cross-entropy loss on validation set
- **Time**: Training time per epoch (format: MM:SS)

## Data Format

### CSV Structure
```
accuracy_multi,epoch,learning_rate,lwlrap,model,pretrained,time,train_loss,training_phase,valid_loss
,0,1e-1,0.080807,resnet18,No,00:07,0.257204,1,0.092621
0.985501,1,1e-1,0.113652,resnet18,No,00:06,0.125856,1,0.098057
...
```

### Training Phase Numbering
Each notebook has multiple fit_one_cycle calls, numbered sequentially:
- Phase 1: Initial training with high learning rate
- Phase 2-5: Fine-tuning with progressively lower learning rates

## Usage

### Loading the CSV in Python
```python
import pandas as pd

df = pd.read_csv('training_metrics_extracted.csv')

# Filter by model
resnet_data = df[df['model'] == 'resnet18']

# Plot training progression
import matplotlib.pyplot as plt
plt.plot(resnet_data['epoch'], resnet_data['lwlrap'])
plt.xlabel('Epoch')
plt.ylabel('LWLRAP')
plt.title('ResNet18 Training Progression')
plt.show()
```

### Key Queries
```python
# Best performing epoch for each model
best_epochs = df.loc[df.groupby('model')['lwlrap'].idxmax()]

# Training phase comparison
phase_avg = df.groupby(['model', 'training_phase'])['lwlrap'].mean()

# Final performance
final_perf = df.groupby('model').last()[['lwlrap', 'valid_loss']]
```

## Recommendations

1. **Use ResNet18 for this task** - significantly better LWLRAP performance
2. **Train from scratch** - don't use ImageNet pretrained weights
3. **Use progressive learning rate scheduling** - all successful models used 5 training phases
4. **Monitor LWLRAP, not just accuracy** - accuracy can be misleading for multi-label tasks
5. **Vision Transformers are not suitable** for this audio spectrogram classification task

