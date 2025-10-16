#!/usr/bin/env python3
"""
FastAI Tabular Neural Network Training - ESC-50 Only
Trains 2-layer and 5-layer neural networks on ESC-50 dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from fastai.tabular.all import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FASTAI NEURAL NETWORK - ESC-50 ONLY")
print("=" * 80)
print()


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
    print()

    return df, feature_cols


def train_nn(df, feature_cols, layers, epochs=10, lr=1e-3):
    """
    Train neural network using fastai tabular learner

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        layers: List of hidden layer sizes (e.g., [200, 100] for 2-layer)
        epochs: Number of training epochs
        lr: Learning rate
    """
    print(f"{'=' * 80}")
    print(f"Training {len(layers)}-Layer NN")
    print(f"Architecture: {layers}")
    print(f"{'=' * 80}\n")

    # Create combined dataframe with split indicator
    df_combined = df.copy().reset_index(drop=True)

    # Use fold-based split for ESC-50
    train_idx = df_combined[df_combined['fold'] != 5].index.tolist()
    valid_idx = df_combined[df_combined['fold'] == 5].index.tolist()

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
    print(f"RESULTS - {len(layers)}-Layer NN on ESC-50")
    print(f"{'=' * 80}")
    print(f"Overall Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Balanced Accuracy:     {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"F1 Score (weighted):   {f1:.4f}")
    print()

    return {
        'num_layers': len(layers),
        'layers': str(layers),
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

    # Prepare dataset
    esc50_df, esc50_features = prepare_esc50_data()

    # Define architectures
    arch_2layer = [200, 100]  # 2 hidden layers
    arch_5layer = [400, 300, 200, 100, 50]  # 5 hidden layers

    results = []

    # Train 2-layer
    result = train_nn(esc50_df, esc50_features, arch_2layer, epochs=10, lr=1e-3)
    results.append(result)

    # Train 5-layer
    result = train_nn(esc50_df, esc50_features, arch_5layer, epochs=10, lr=1e-3)
    results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - ESC-50 NEURAL NETWORKS")
    print("=" * 80)
    print()

    df_results = pd.DataFrame(results)
    print(df_results[['num_layers', 'layers', 'accuracy', 'balanced_accuracy', 'f1_score']].to_string(index=False))
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()

    esc_2 = results[0]['accuracy']
    esc_5 = results[1]['accuracy']

    print(f"2-layer NN: {esc_2*100:.2f}%")
    print(f"5-layer NN: {esc_5*100:.2f}%")
    print(f"Difference: {(esc_5 - esc_2)*100:+.2f}%")
    print()

    # vs LinearSVC baseline
    print("Comparison to LinearSVC Baseline:")
    print(f"  ESC-50 LinearSVC: 35.00%")
    print(f"  ESC-50 2-layer NN: {esc_2*100:.2f}% ({(esc_2*100 - 35.00):+.2f}%)")
    print(f"  ESC-50 5-layer NN: {esc_5*100:.2f}% ({(esc_5*100 - 35.00):+.2f}%)")
    print()

    if esc_5 > esc_2:
        print(f"✓ Deeper network (5-layer) is BETTER by {(esc_5 - esc_2)*100:.2f}%")
    elif esc_2 > esc_5:
        print(f"✗ Deeper network (5-layer) is WORSE by {(esc_2 - esc_5)*100:.2f}%")
    else:
        print("= Both networks perform equally")

    print()

    # Save results
    output_dir = Path('claude/lazypredict/findings')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'esc50_fastai_nn_results.csv'
    df_results.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    print()

    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
