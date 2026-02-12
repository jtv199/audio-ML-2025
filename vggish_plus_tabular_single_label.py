#!/usr/bin/env python3
"""
FastAI Tabular Predictor combining VGGish Embeddings + Tabular Features
Train 2-layer model on single-label samples for 20 epochs
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
from sklearn.metrics import classification_report, accuracy_score

import torch
torch.set_num_threads(4)  # Limit CPU threads


def prepare_data():
    """Load and merge VGGish embeddings with tabular features"""
    print("="*80)
    print("LOADING AND MERGING DATA")
    print("="*80)

    # Load VGGish embeddings
    vggish_df = pd.read_csv('work/tokenized/vggish_embeddings_train_curated.csv')
    print(f"VGGish data loaded: {vggish_df.shape}")

    # Load tabular features
    tabular_df = pd.read_csv('work/trn_curated_feature.csv')
    print(f"Tabular features loaded: {tabular_df.shape}")

    # Merge on filename
    # Rename 'file' to 'fname' in tabular_df to match vggish_df
    tabular_df = tabular_df.rename(columns={'file': 'fname'})

    # Merge datasets
    merged_df = vggish_df.merge(tabular_df, on='fname', how='inner')
    print(f"Merged data: {merged_df.shape}")

    # Filter to single-label samples only
    merged_df['num_labels'] = merged_df['labels'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

    print(f"\nLabel distribution before filtering:")
    print(merged_df['num_labels'].value_counts().sort_index())

    # Keep only single-label samples
    single_label_df = merged_df[merged_df['num_labels'] == 1].copy()

    print(f"\n{'='*80}")
    print(f"FILTERED TO SINGLE-LABEL SAMPLES")
    print(f"{'='*80}")
    print(f"Original samples: {len(merged_df)}")
    print(f"Single-label samples: {len(single_label_df)} ({len(single_label_df)/len(merged_df)*100:.1f}%)")
    print(f"Removed: {len(merged_df) - len(single_label_df)} multi-label samples")

    # Get unique classes
    unique_labels = sorted(single_label_df['labels'].unique())
    print(f"\nUnique classes: {len(unique_labels)}")

    # Get feature columns
    vggish_cols = [f'emb_{i}' for i in range(128)]
    tabular_cols = [col for col in tabular_df.columns if col != 'fname']

    all_feature_cols = vggish_cols + tabular_cols

    print(f"\nFeature breakdown:")
    print(f"  VGGish embeddings: {len(vggish_cols)}")
    print(f"  Tabular features: {len(tabular_cols)}")
    print(f"  Total features: {len(all_feature_cols)}")

    # Reset index
    single_label_df = single_label_df.reset_index(drop=True)

    # Split into train and validation
    train_idx, val_idx = train_test_split(
        range(len(single_label_df)),
        test_size=0.2,
        random_state=42,
        stratify=single_label_df['labels']
    )

    single_label_df['is_valid'] = False
    single_label_df.loc[val_idx, 'is_valid'] = True

    print(f"\nTrain samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")

    return single_label_df, all_feature_cols, unique_labels


def train_model(df, feature_cols, unique_labels, layers=[200, 100], epochs=20):
    """Train FastAI tabular model"""

    print(f"\n{'='*80}")
    print(f"TRAINING MODEL WITH {len(layers)} LAYERS: {layers}")
    print(f"{'='*80}")

    # Create TabularDataLoaders
    dls = TabularDataLoaders.from_df(
        df,
        path='.',
        procs=[Categorify, Normalize],
        cat_names=[],  # No categorical features
        cont_names=feature_cols,  # All features are continuous
        y_names='labels',  # Target is the label
        valid_idx=df[df['is_valid']].index.tolist(),
        bs=64
    )

    # Create learner
    learn = tabular_learner(
        dls,
        layers=layers,
        metrics=[accuracy, error_rate],
        loss_func=CrossEntropyLossFlat()
    )

    print(f"\nModel architecture:")
    print(learn.model)

    num_params = sum(p.numel() for p in learn.model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    # Find learning rate
    print("\nFinding optimal learning rate...")
    try:
        suggested_lrs = learn.lr_find(show_plot=False)
        lr_steep = float(suggested_lrs.valley) if hasattr(suggested_lrs, 'valley') else 1e-3
    except:
        lr_steep = 1e-3

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

    print(f"\nValidation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

    # Save results
    results = {
        'layers': str(layers),
        'num_layers': len(layers),
        'epochs': epochs,
        'training_time': training_time,
        'val_accuracy': val_accuracy,
        'lr': float(lr_steep),
        'num_params': num_params,
        'num_features': len(feature_cols),
        'num_classes': len(unique_labels),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    # Save model using export
    Path("models").mkdir(exist_ok=True)
    model_name = f"vggish_tabular_single_label_{len(layers)}layers_{results['timestamp']}"
    learn.export(f"models/{model_name}.pkl")
    print(f"\nModel exported as: models/{model_name}.pkl")

    return results, learn


def main():
    """Main execution"""

    print("="*80)
    print("FASTAI TABULAR: VGGISH + TABULAR FEATURES (SINGLE-LABEL)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Prepare data
    df, feature_cols, unique_labels = prepare_data()

    # Train 2-layer model for 20 epochs
    results, learner = train_model(
        df, feature_cols, unique_labels,
        layers=[200, 100],
        epochs=20
    )

    # Save results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    results_df = pd.DataFrame([results])
    print("\n", results_df.to_string())

    # Save to CSV
    output_file = f"vggish_tabular_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal Validation Accuracy: {results['val_accuracy']:.4f} ({results['val_accuracy']*100:.2f}%)")
    print(f"Model saved to: models/vggish_tabular_single_label_2layers_{results['timestamp']}.pkl")


if __name__ == "__main__":
    main()
