#!/usr/bin/env python3
"""
Quick GPU Test with FastAI Tabular Learner
Tests if training actually uses GPU
"""

import torch
import time
from fastai.tabular.all import *

print("="*60)
print("GPU Training Test")
print("="*60)

# Check CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    print("WARNING: CUDA not available!")

print("\n" + "="*60)
print("Creating Simple Tabular Dataset")
print("="*60)

# Create a simple synthetic dataset
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

# Generate synthetic data
n_samples = 10000
X, y = make_classification(n_samples=n_samples, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['target'] = y

print(f"Dataset shape: {df.shape}")
print(f"Features: 20")
print(f"Samples: {n_samples}")

# Split data
splits = RandomSplitter(valid_pct=0.2)(range_of(df))

# Create TabularPandas
cont_names = [f'feature_{i}' for i in range(20)]
to = TabularPandas(df,
                   procs=[Normalize],
                   cont_names=cont_names,
                   y_names='target',
                   splits=splits)

# Create DataLoaders
dls = to.dataloaders(bs=256)

print("\n" + "="*60)
print("Creating Tabular Learner")
print("="*60)

# Create learner
learn = tabular_learner(dls, metrics=accuracy, layers=[200, 100])

# Check device
model_device = next(learn.model.parameters()).device
print(f"Model device: {model_device}")

if torch.cuda.is_available():
    print(f"GPU memory allocated after model creation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

print("\n" + "="*60)
print("Training for 5 epochs (should be FAST on GPU)")
print("="*60)

start_time = time.time()

# Train
learn.fit_one_cycle(5, 1e-2)

end_time = time.time()
training_time = end_time - start_time

print("\n" + "="*60)
print("Training Complete")
print("="*60)

print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Time per epoch: {training_time/5:.2f} seconds")

if torch.cuda.is_available():
    print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    # Check GPU utilization
    print("\nGPU Utilization Test:")
    if training_time / 5 < 2:  # Less than 2 seconds per epoch
        print("✓ Training is FAST - GPU is being used!")
    else:
        print("⚠ Training is slow - might be using CPU")

# Get predictions to verify model works
preds, _ = learn.get_preds()
print(f"\nPredictions shape: {preds.shape}")
print(f"Model accuracy: {accuracy(preds, dls.valid.items.target.values).item():.4f}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)

# Final check
if torch.cuda.is_available() and model_device.type == 'cuda':
    print("\n✓✓✓ SUCCESS: Model trained on GPU!")
else:
    print("\n✗✗✗ WARNING: Model trained on CPU!")
