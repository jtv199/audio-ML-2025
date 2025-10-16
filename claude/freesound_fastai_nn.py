#!/usr/bin/env python3
"""
FastAI Neural Network Training - FreeSound Dataset
Train 2-layer NN on FreeSound single-label features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from fastai.tabular.all import *
from fastai.metrics import accuracy as fastai_accuracy
import warnings
warnings.filterwarnings('ignore')

def train_nn(df, feature_cols, layers, epochs=10, lr=1e-3):
    """Train a neural network using FastAI"""

    # Create train/valid split
    df_combined = df.copy().reset_index(drop=True)
    train_idx = df_combined[df_combined['split'] == 'train'].index.tolist()
    valid_idx = df_combined[df_combined['split'] == 'valid'].index.tolist()
    splits = (train_idx, valid_idx)

    # Create DataLoaders
    dls = TabularDataLoaders.from_df(
        df_combined,
        path='.',
        cont_names=feature_cols,
        y_names='target',
        splits=splits,
        bs=64
    )

    # Create learner
    learn = tabular_learner(dls, layers=layers, metrics=fastai_accuracy)

    # Train
    learn.fit_one_cycle(epochs, lr)

    # Get predictions
    preds, targets = learn.get_preds()
    preds_class = preds.argmax(dim=1).numpy()
    targets_np = targets.numpy()

    # Calculate metrics
    acc = accuracy_score(targets_np, preds_class)
    bal_acc = balanced_accuracy_score(targets_np, preds_class)
    f1 = f1_score(targets_np, preds_class, average='weighted')

    return {
        'layers': layers,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_score': f1,
        'train_samples': len(train_idx),
        'valid_samples': len(valid_idx),
        'epochs': epochs,
        'learning_rate': lr
    }

def main():
    print("=" * 80)
    print("FASTAI NEURAL NETWORK - FREESOUND DATASET")
    print("=" * 80)
    print()

    # Load FreeSound features
    print("Loading FreeSound dataset...")
    features_path = Path('work/trn_curated_feature.csv')
    labels_path = Path('input/train_curated.csv')

    if not features_path.exists():
        print(f"ERROR: {features_path} not found!")
        return

    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found!")
        return

    # Load features and labels
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    # Merge features and labels
    df = features_df.merge(labels_df, left_on='file', right_on='fname', how='inner')

    # Filter to single-class samples only
    df['num_labels'] = df['labels'].str.count(',') + 1
    single_class_df = df[df['num_labels'] == 1].copy()

    print(f"  Original samples: {len(df)}")
    print(f"  Single-class samples: {len(single_class_df)} ({100 * len(single_class_df) / len(df):.1f}%)")

    # Encode labels to integer targets
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    single_class_df['target'] = le.fit_transform(single_class_df['labels'])

    # Get feature columns (all except file, fname, labels, num_labels, target)
    feature_cols = [col for col in single_class_df.columns
                    if col not in ['file', 'fname', 'labels', 'num_labels', 'target', 'split']]

    # Create train/valid split (80/20)
    train_df, valid_df = train_test_split(single_class_df, test_size=0.2,
                                          random_state=42, stratify=single_class_df['target'])
    train_df['split'] = 'train'
    valid_df['split'] = 'valid'
    df_combined = pd.concat([train_df, valid_df], ignore_index=True)

    print(f"  Classes: {df_combined['target'].nunique()}")
    print(f"  Features: {len(feature_cols)}")
    print()

    # Train 2-layer NN
    print("=" * 80)
    print("Training 2-Layer NN")
    print("Architecture: [200, 100]")
    print("=" * 80)
    print()
    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    print()
    print("Training for 10 epochs...")

    results_2layer = train_nn(df_combined, feature_cols, layers=[200, 100], epochs=10, lr=1e-3)

    print()
    print("=" * 80)
    print("RESULTS - 2-Layer NN on FreeSound")
    print("=" * 80)
    print(f"Overall Accuracy:      {results_2layer['accuracy']:.4f} ({results_2layer['accuracy']*100:.2f}%)")
    print(f"Balanced Accuracy:     {results_2layer['balanced_accuracy']:.4f} ({results_2layer['balanced_accuracy']*100:.2f}%)")
    print(f"F1 Score (weighted):   {results_2layer['f1_score']:.4f}")
    print()

    # Save results
    results_df = pd.DataFrame([results_2layer])
    results_df['num_layers'] = 2
    results_df['layers'] = str(results_2layer['layers'])

    output_file = 'claude/lazypredict/findings/freesound_fastai_nn_results.csv'
    results_df.to_csv(output_file, index=False)

    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print("FreeSound LinearSVC: 56.60%")
    print(f"FreeSound 2-layer NN: {results_2layer['accuracy']*100:.2f}%")
    diff = results_2layer['accuracy']*100 - 56.60
    print(f"Difference: {diff:+.2f}%")
    print()

    if diff > 0:
        print(f"✓ Neural network is BETTER by {abs(diff):.2f}%")
    else:
        print(f"✗ Neural network is WORSE by {abs(diff):.2f}%")

    print()
    print(f"Results saved to: {output_file}")
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
