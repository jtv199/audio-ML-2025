#!/bin/bash
# Script to run ResNet18 single-label training
# This may take longer than 10 minutes, limited to 20 epochs

echo "Starting ResNet18 single-label audio classification training..."
echo "This will train for 20 epochs total on single-label data only"
echo "Using accuracy as the metric (instead of lwlrap)"
echo ""

# Run with conda environment
~/miniconda3/envs/freesound/bin/python train_resnet18_single_label.py

echo ""
echo "Training complete! Check the output above for results."
