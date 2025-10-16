#!/usr/bin/env python3
"""
FastAI Tabular Predictor for VGGish Embeddings
Train models with 2 and 5 layers on VGGish embeddings
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FastAI imports
from fastai.tabular.all import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

import torch
torch.set_num_threads(4)  # Limit CPU threads


def prepare_data():
    """Load and prepare the VGGish embeddings data"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    # Load embeddings
    train_df = pd.read_csv('work/tokenized/vggish_embeddings_train_curated.csv')
    print(f"Train data loaded: {train_df.shape}")

    # Process labels - convert multi-label to separate columns
    print("\nProcessing labels...")

    # Split labels and get unique classes
    all_labels = []
    for labels in train_df['labels']:
        all_labels.extend(labels.split(','))
    unique_labels = sorted(list(set(all_labels)))
    print(f"Found {len(unique_labels)} unique labels")
    print(f"Labels: {unique_labels[:10]}..." if len(unique_labels) > 10 else f"Labels: {unique_labels}")

    # Create binary columns for each label
    for label in unique_labels:
        train_df[f'label_{label}'] = train_df['labels'].apply(
            lambda x: 1 if label in x.split(',') else 0
        )

    # Get embedding columns
    emb_cols = [f'emb_{i}' for i in range(128)]
    label_cols = [f'label_{label}' for label in unique_labels]

    print(f"\nEmbedding columns: {len(emb_cols)}")
    print(f"Label columns: {len(label_cols)}")
    print(f"Total features: {len(train_df.columns)}")

    # Split into train and validation
    train_idx, val_idx = train_test_split(
        range(len(train_df)),
        test_size=0.2,
        random_state=42
    )

    train_df['is_valid'] = False
    train_df.loc[val_idx, 'is_valid'] = True

    print(f"\nTrain samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")

    return train_df, emb_cols, label_cols, unique_labels


def create_tabular_learner(df, emb_cols, label_cols, layers, lr=1e-3):
    """Create a FastAI tabular learner"""

    # For multi-label classification, we'll train one model per label
    # and combine predictions later
    # For simplicity, let's predict the most common label or use a single-label approach

    # Alternative: Create one-hot encoded single label (taking first label for each sample)
    df_copy = df.copy()
    df_copy['primary_label'] = df_copy['labels'].apply(lambda x: x.split(',')[0])

    # Create TabularDataLoaders
    dls = TabularDataLoaders.from_df(
        df_copy,
        path='.',
        procs=[Categorify, Normalize],
        cat_names=[],  # No categorical features
        cont_names=emb_cols,  # All embedding columns are continuous
        y_names='primary_label',  # Target is the primary label
        valid_idx=df_copy[df_copy['is_valid']].index.tolist(),
        bs=64
    )

    # Create learner
    learn = tabular_learner(
        dls,
        layers=layers,
        metrics=[accuracy, error_rate],
        loss_func=CrossEntropyLossFlat()
    )

    return learn, dls


def train_and_evaluate(df, emb_cols, label_cols, unique_labels, layers, epochs=10):
    """Train and evaluate a model"""

    print(f"\n{'='*80}")
    print(f"TRAINING MODEL WITH {len(layers)} LAYERS: {layers}")
    print(f"{'='*80}")

    # Create learner
    learn, dls = create_tabular_learner(df, emb_cols, label_cols, layers)

    print(f"\nModel architecture:")
    print(learn.model)

    # Find learning rate
    print("\nFinding optimal learning rate...")
    try:
        suggested_lrs = learn.lr_find(show_plot=False)
        lr_steep = float(suggested_lrs.valley) if hasattr(suggested_lrs, 'valley') else 1e-3
        lr_min = lr_steep / 10
    except:
        lr_steep = 1e-3
        lr_min = 1e-4
    print(f"Using LR: {lr_steep:.2e}")

    # Train
    print(f"\nTraining for {epochs} epochs...")
    start_time = datetime.now()

    learn.fit_one_cycle(epochs, lr_max=lr_steep)

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Evaluate
    print("\nEvaluating on validation set...")
    val_preds, val_targets = learn.get_preds(dl=dls.valid)
    val_preds_class = val_preds.argmax(dim=1)

    val_accuracy = accuracy_score(val_targets.cpu().numpy(), val_preds_class.cpu().numpy())

    print(f"\nValidation Accuracy: {val_accuracy:.4f}")

    # Get class names
    vocab = dls.vocab[1]  # Get label vocabulary

    # Save results
    results = {
        'layers': layers,
        'num_layers': len(layers),
        'epochs': epochs,
        'training_time': training_time,
        'val_accuracy': val_accuracy,
        'lr_min': float(lr_min),
        'lr_steep': float(lr_steep),
        'num_params': sum(p.numel() for p in learn.model.parameters()),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    # Save model
    model_name = f"vggish_fastai_{len(layers)}layers_{results['timestamp']}"
    learn.save(model_name)
    print(f"\nModel saved as: {model_name}")

    return results, learn


def train_multi_label_approach(df, emb_cols, label_cols, unique_labels, layers, epochs=10):
    """Train model with proper multi-label classification"""

    print(f"\n{'='*80}")
    print(f"TRAINING MULTI-LABEL MODEL WITH {len(layers)} LAYERS: {layers}")
    print(f"{'='*80}")

    # Prepare data for multi-label
    df_copy = df.copy()

    # Split data
    train_df = df_copy[~df_copy['is_valid']].copy()
    val_df = df_copy[df_copy['is_valid']].copy()

    X_train = train_df[emb_cols].values
    y_train = train_df[label_cols].values
    X_val = val_df[emb_cols].values
    y_val = val_df[label_cols].values

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Create DataLoaders manually
    from torch.utils.data import TensorDataset, DataLoader

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)

    dls = DataLoaders(train_dl, val_dl)

    # Create custom model
    class MultiLabelModel(nn.Module):
        def __init__(self, input_dim, output_dim, layers):
            super().__init__()

            # Build layers
            layer_list = []
            prev_dim = input_dim

            for layer_dim in layers:
                layer_list.extend([
                    nn.Linear(prev_dim, layer_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(layer_dim),
                    nn.Dropout(0.5)
                ])
                prev_dim = layer_dim

            # Final output layer
            layer_list.append(nn.Linear(prev_dim, output_dim))

            self.layers = nn.Sequential(*layer_list)

        def forward(self, x):
            return self.layers(x)

    # Create model
    model = MultiLabelModel(
        input_dim=len(emb_cols),
        output_dim=len(label_cols),
        layers=layers
    )

    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create learner with BCEWithLogitsLoss for multi-label
    learn = Learner(
        dls,
        model,
        loss_func=BCEWithLogitsLossFlat(),
        metrics=[accuracy_multi]
    )

    # Find learning rate
    print("\nFinding optimal learning rate...")
    try:
        suggested_lrs = learn.lr_find(show_plot=False)
        lr_steep = float(suggested_lrs.valley) if hasattr(suggested_lrs, 'valley') else 1e-3
        lr_min = lr_steep / 10
    except:
        lr_steep = 1e-3
        lr_min = 1e-4
    print(f"Using LR: {lr_steep:.2e}")

    # Train
    print(f"\nTraining for {epochs} epochs...")
    start_time = datetime.now()

    learn.fit_one_cycle(epochs, lr_max=lr_steep)

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Evaluate
    print("\nEvaluating on validation set...")
    val_preds, val_targets = learn.get_preds(dl=dls.valid)

    # Apply sigmoid to get probabilities
    val_probs = torch.sigmoid(val_preds)
    val_preds_binary = (val_probs > 0.5).float()

    # Calculate metrics
    val_hamming = hamming_loss(val_targets.cpu().numpy(), val_preds_binary.cpu().numpy())
    val_accuracy = accuracy_score(val_targets.cpu().numpy(), val_preds_binary.cpu().numpy())

    print(f"\nValidation Hamming Loss: {val_hamming:.4f}")
    print(f"Validation Accuracy (exact match): {val_accuracy:.4f}")

    # Per-label accuracy
    per_label_acc = (val_targets == val_preds_binary).float().mean(dim=0)
    print(f"Mean per-label accuracy: {per_label_acc.mean():.4f}")

    # Save results
    results = {
        'layers': layers,
        'num_layers': len(layers),
        'epochs': epochs,
        'training_time': training_time,
        'val_hamming_loss': float(val_hamming),
        'val_accuracy': float(val_accuracy),
        'val_per_label_acc': float(per_label_acc.mean()),
        'lr_min': float(lr_min),
        'lr_steep': float(lr_steep),
        'num_params': sum(p.numel() for p in model.parameters()),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    # Save model
    model_name = f"vggish_multilabel_{len(layers)}layers_{results['timestamp']}"
    learn.save(model_name)
    print(f"\nModel saved as: {model_name}")

    return results, learn


def main():
    """Main execution"""

    print("="*80)
    print("FASTAI TABULAR PREDICTOR ON VGGISH EMBEDDINGS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Prepare data
    df, emb_cols, label_cols, unique_labels = prepare_data()

    # Train models with different layer configurations
    layer_configs = [
        [200, 100],  # 2 layers
        [400, 300, 200, 150, 100]  # 5 layers
    ]

    all_results = []

    for layers in layer_configs:
        results, learner = train_multi_label_approach(
            df, emb_cols, label_cols, unique_labels,
            layers=layers,
            epochs=20
        )
        all_results.append(results)

    # Save results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(all_results)
    print("\n", results_df.to_string())

    # Save to CSV
    output_file = f"vggish_fastai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    for idx, row in results_df.iterrows():
        print(f"\nModel {idx+1}: {row['num_layers']} layers {row['layers']}")
        print(f"  Parameters: {row['num_params']:,}")
        print(f"  Training time: {row['training_time']:.2f}s ({row['training_time']/60:.2f} min)")
        print(f"  Val Hamming Loss: {row['val_hamming_loss']:.4f}")
        print(f"  Val Accuracy: {row['val_accuracy']:.4f}")
        print(f"  Val Per-Label Acc: {row['val_per_label_acc']:.4f}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
