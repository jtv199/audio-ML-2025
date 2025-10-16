#!/usr/bin/env python3
"""
FastAI Tabular Neural Network Training
Trains 2-layer and 5-layer neural networks on FreeSound and ESC-50 datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from fastai.tabular.all import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FASTAI TABULAR NEURAL NETWORK TRAINING")
print("=" * 80)
print()


def prepare_freesound_data():
    """Load and prepare FreeSound dataset"""
    print("Loading FreeSound dataset...")

    # Load features and labels
    features_df = pd.read_csv('work/trn_curated_feature.csv')
    labels_df = pd.read_csv('input/train_curated.csv')

    # Merge
    df = features_df.merge(labels_df, left_on='file', right_on='fname', how='inner')

    # Filter to single-class samples
    df['num_labels'] = df['labels'].str.count(',') + 1
    single_class_df = df[df['num_labels'] == 1].copy()

    # Remove rows with missing values
    feature_cols = [col for col in single_class_df.columns
                    if col not in ['file', 'fname', 'labels', 'num_labels']]

    mask = ~single_class_df[feature_cols].isnull().any(axis=1)
    single_class_df = single_class_df[mask]

    # Add target column
    single_class_df['target'] = single_class_df['labels']

    print(f"  Samples: {len(single_class_df)}")
    print(f"  Classes: {single_class_df['target'].nunique()}")
    print(f"  Features: {len(feature_cols)}")

    return single_class_df, feature_cols


def prepare_esc50_data():
    """Load and prepare ESC-50 dataset"""
    print("Loading ESC-50 dataset...")

    df = pd.read_csv('data/esc50/esc50_features.csv')

    # Use category as target
    df['target'] = df['category']

    feature_cols = [col for col in df.columns
                    if col not in ['filename', 'fold', 'target', 'category', 'esc10']]

    print(f"  Samples: {len(df)}")
    print(f"  Classes: {df['target'].nunique()}")
    print(f"  Features: {len(feature_cols)}")

    return df, feature_cols


def train_nn(df, feature_cols, layers, dataset_name, epochs=10, lr=1e-3):
    """
    Train neural network using fastai tabular learner

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        layers: List of hidden layer sizes (e.g., [200, 100] for 2-layer)
        dataset_name: Name for logging
        epochs: Number of training epochs
        lr: Learning rate
    """
    print(f"\n{'=' * 80}")
    print(f"Training {len(layers)}-Layer NN on {dataset_name}")
    print(f"Architecture: {layers}")
    print(f"{'=' * 80}\n")

    # Create combined dataframe with split indicator
    df_combined = df.copy().reset_index(drop=True)

    if dataset_name == "ESC-50":
        # Use fold-based split for ESC-50
        train_idx = df_combined[df_combined['fold'] != 5].index.tolist()
        valid_idx = df_combined[df_combined['fold'] == 5].index.tolist()
    else:
        # Random split for FreeSound
        from sklearn.model_selection import train_test_split
        train_idx, valid_idx = train_test_split(
            range(len(df_combined)),
            test_size=0.2,
            random_state=42,
            stratify=df_combined['target']
        )

    print(f"Train samples: {len(train_idx)}")
    print(f"Valid samples: {len(valid_idx)}")

    # Create TabularDataLoaders with proper splits
    splits = (train_idx, valid_idx)

    dls = TabularDataLoaders.from_df(
        df_combined,
        path='.',
        cont_names=feature_cols,
        y_names='target',
        splits=splits,
        bs=64
    )

    # Create learner
    from fastai.metrics import accuracy as fastai_accuracy
    learn = tabular_learner(dls, layers=layers, metrics=fastai_accuracy)

    # Train
    print(f"\nTraining for {epochs} epochs...")
    learn.fit_one_cycle(epochs, lr)

    # Get predictions
    preds, targets = learn.get_preds()
    y_pred = preds.argmax(dim=1).numpy()
    y_true = targets.numpy()

    # Decode class names
    vocab = dls.vocab
    y_pred_labels = [vocab[i] for i in y_pred]
    y_true_labels = [vocab[i] for i in y_true]

    # Calculate metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    balanced_acc = balanced_accuracy_score(y_true_labels, y_pred_labels)
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

    print(f"\n{'=' * 80}")
    print(f"RESULTS - {len(layers)}-Layer NN on {dataset_name}")
    print(f"{'=' * 80}")
    print(f"Overall Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Balanced Accuracy:     {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"F1 Score (weighted):   {f1:.4f}")
    print()

    return {
        'dataset': dataset_name,
        'layers': str(layers),
        'num_layers': len(layers),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'train_samples': len(train_idx),
        'valid_samples': len(valid_idx),
        'epochs': epochs,
        'learning_rate': lr
    }


def main():
    """Main execution"""

    # Prepare datasets
    print("=" * 80)
    print("PREPARING DATASETS")
    print("=" * 80)
    print()

    freesound_df, freesound_features = prepare_freesound_data()
    print()
    esc50_df, esc50_features = prepare_esc50_data()
    print()

    # Define architectures
    arch_2layer = [200, 100]  # 2 hidden layers
    arch_5layer = [400, 300, 200, 100, 50]  # 5 hidden layers

    results = []

    # Train on FreeSound
    print("\n" + "=" * 80)
    print("FREESOUND EXPERIMENTS")
    print("=" * 80)

    # 2-layer
    result = train_nn(freesound_df, freesound_features, arch_2layer,
                     "FreeSound", epochs=10, lr=1e-3)
    results.append(result)

    # 5-layer
    result = train_nn(freesound_df, freesound_features, arch_5layer,
                     "FreeSound", epochs=10, lr=1e-3)
    results.append(result)

    # Train on ESC-50
    print("\n" + "=" * 80)
    print("ESC-50 EXPERIMENTS")
    print("=" * 80)

    # 2-layer
    result = train_nn(esc50_df, esc50_features, arch_2layer,
                     "ESC-50", epochs=10, lr=1e-3)
    results.append(result)

    # 5-layer
    result = train_nn(esc50_df, esc50_features, arch_5layer,
                     "ESC-50", epochs=10, lr=1e-3)
    results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - ALL EXPERIMENTS")
    print("=" * 80)
    print()

    df_results = pd.DataFrame(results)
    print(df_results[['dataset', 'num_layers', 'layers', 'accuracy', 'balanced_accuracy', 'f1_score']].to_string(index=False))
    print()

    # Comparisons
    print("=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)
    print()

    # FreeSound: 2-layer vs 5-layer
    fs_2 = df_results[(df_results['dataset'] == 'FreeSound') & (df_results['num_layers'] == 2)]['accuracy'].values[0]
    fs_5 = df_results[(df_results['dataset'] == 'FreeSound') & (df_results['num_layers'] == 5)]['accuracy'].values[0]
    print(f"FreeSound:")
    print(f"  2-layer NN: {fs_2*100:.2f}%")
    print(f"  5-layer NN: {fs_5*100:.2f}%")
    print(f"  Difference: {(fs_5 - fs_2)*100:+.2f}%")
    print()

    # ESC-50: 2-layer vs 5-layer
    esc_2 = df_results[(df_results['dataset'] == 'ESC-50') & (df_results['num_layers'] == 2)]['accuracy'].values[0]
    esc_5 = df_results[(df_results['dataset'] == 'ESC-50') & (df_results['num_layers'] == 5)]['accuracy'].values[0]
    print(f"ESC-50:")
    print(f"  2-layer NN: {esc_2*100:.2f}%")
    print(f"  5-layer NN: {esc_5*100:.2f}%")
    print(f"  Difference: {(esc_5 - esc_2)*100:+.2f}%")
    print()

    # vs LinearSVC baseline
    print("Comparison to LinearSVC Baseline:")
    print(f"  FreeSound LinearSVC: 56.56%")
    print(f"  FreeSound 2-layer NN: {fs_2*100:.2f}% ({(fs_2*100 - 56.56):+.2f}%)")
    print(f"  FreeSound 5-layer NN: {fs_5*100:.2f}% ({(fs_5*100 - 56.56):+.2f}%)")
    print()
    print(f"  ESC-50 LinearSVC: 35.00%")
    print(f"  ESC-50 2-layer NN: {esc_2*100:.2f}% ({(esc_2*100 - 35.00):+.2f}%)")
    print(f"  ESC-50 5-layer NN: {esc_5*100:.2f}% ({(esc_5*100 - 35.00):+.2f}%)")
    print()

    # Save results
    output_dir = Path('claude/lazypredict/findings')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'fastai_nn_results.csv'
    df_results.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    print()

    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
