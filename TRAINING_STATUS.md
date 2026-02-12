# Training Status

## Setup Complete ✓

**Training Started:** Successfully running in background

**Process ID (PID):** 23800

**GPU:** NVIDIA GeForce RTX 2070 SUPER (CUDA 12.1) - **ACTIVE**

**PyTorch:** 2.4.1+cu121 (with CUDA support)

## Current Status

Training is currently in the **data preprocessing phase**, converting 4269 audio files to mel-spectrograms.

Progress: Converting at ~10-15 images/second
- Training audio: 4269 files
- Test audio: 3361 files

## Models to Train

1. **ResNet18** - 115 total epochs (6 phases)
2. **DeiT Tiny** - 115 total epochs (6 phases)

## Training Configuration

Each model will train for:
- Phase 1: 5 epochs @ lr=0.1
- Phase 2: 10 epochs @ lr=0.01
- Phase 3: 20 epochs @ lr=0.003
- Phase 4: 20 epochs @ lr=0.001
- Phase 5: 50 epochs @ lr=0.001-0.003 (discriminative)
- Phase 6: 10 epochs @ lr=0.0001-0.001 (fine-tuning)

**Total:** 115 epochs per model × 2 models = 230 epochs

## Files Created

- **train_models.py** - Main training script (running in background)
- **training.log** - Real-time training output
- **training_results.csv** - Epoch-by-epoch metrics (will be created after first epoch)
- **monitor_training.sh** - Monitoring script (checks every 10-30 minutes)
- **models/** - Saved models directory

## Monitoring

### Check Training Progress

```bash
# View last 50 lines of log
tail -50 training.log

# Check if training is running
ps aux | grep train_models.py | grep -v grep

# Check GPU usage
nvidia-smi

# View training results
cat training_results.csv
```

### Run Automated Monitor

```bash
# Start the monitoring script (checks every 10-30 min)
./monitor_training.sh
```

### Manual Progress Checks

```bash
# Quick status
tail -30 training.log

# GPU utilization
watch -n 5 nvidia-smi

# Training results summary
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('training_results.csv')
for model in df['model_name'].unique():
    model_df = df[df['model_name'] == model]
    print(f"\n{model}:")
    print(f"  Epochs: {len(model_df)}/115")
    print(f"  Best accuracy: {model_df['accuracy'].max():.4f}")
    print(f"  Latest loss: {model_df['valid_loss'].iloc[-1]:.4f}")
EOF
```

## Estimated Time

- Data preprocessing: ~5-10 minutes (currently running)
- Training per epoch: ~2-5 minutes (depends on GPU)
- Total estimated time: **8-10 hours** (for both models)

## What's Next

The script will automatically:
1. ✓ Convert all audio to mel-spectrograms
2. Create DataLoaders
3. Train ResNet18 (115 epochs)
4. Save ResNet18 model and results
5. Train DeiT Tiny (115 epochs)
6. Save DeiT Tiny model and results
7. Complete and exit

All results are being saved to `training_results.csv` and models to the `models/` directory.

## Notes

- Training is fully automated - no intervention needed
- Results are saved after each epoch
- Models are saved after training completes
- GPU usage should be high (80-95%) during training
- If training stops unexpectedly, check training.log for errors
