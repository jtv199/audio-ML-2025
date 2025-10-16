#!/usr/bin/env python3
"""
FastAI Transfer Learning on EnCodec Token Embeddings
Uses EnCodec tokens as pre-trained features + 2 dense MLP layers
Finds optimal learning rate and trains for 20 epochs
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# FastAI imports
from fastai.tabular.all import *
from fastai.callback.all import *

import warnings
warnings.filterwarnings('ignore')


def load_encodec_data():
    """Load EnCodec tokenized features and prepare for FastAI"""
    print("=" * 80)
    print("LOADING ENCODEC TOKENIZED DATA")
    print("=" * 80)

    # Load tokenized features
    DATA_PATH = Path('./work/tokenized')
    FEATURES_CSV = DATA_PATH / 'tokenized_train_curated.csv'

    print(f"Loading from: {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    print(f"Dataset shape: {df.shape}")

    # Filter to single-class samples
    df['num_labels'] = df['labels'].str.count(',') + 1
    single_class_df = df[df['num_labels'] == 1].copy()

    print(f"Total samples: {len(df)}")
    print(f"Single-class samples: {len(single_class_df)} ({100 * len(single_class_df) / len(df):.1f}%)")
    print(f"Number of unique classes: {single_class_df['labels'].nunique()}")

    # Prepare features
    metadata_cols = ['fname', 'labels', 'num_tokens', 'token_shape', 'num_chunks',
                     'duration_sec', 'num_labels']
    token_cols = [col for col in single_class_df.columns if col not in metadata_cols]

    print(f"Number of token features: {len(token_cols)}")

    # Create clean dataframe with tokens + target
    clean_df = single_class_df[token_cols + ['labels']].copy()

    # Rename target column for FastAI
    clean_df = clean_df.rename(columns={'labels': 'target'})

    print(f"Final dataset shape: {clean_df.shape}")
    print(f"Class distribution (top 10):")
    print(clean_df['target'].value_counts().head(10))
    print()

    return clean_df, token_cols


def create_dataloaders(df, token_cols, valid_pct=0.2, bs=64, seed=42):
    """Create FastAI DataLoaders for tabular data"""
    print("=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)

    # Define categorical and continuous variables
    cat_names = []  # No categorical features (all tokens are continuous)
    cont_names = token_cols  # All token columns are continuous

    print(f"Continuous features: {len(cont_names)}")
    print(f"Categorical features: {len(cat_names)}")
    print(f"Validation split: {valid_pct*100}%")
    print(f"Batch size: {bs}")

    # Create TabularDataLoaders
    dls = TabularDataLoaders.from_df(
        df,
        path='.',
        cat_names=cat_names,
        cont_names=cont_names,
        y_names='target',
        valid_idx=list(range(int(len(df) * (1-valid_pct)), len(df))),  # Last 20% for validation
        bs=bs,
        seed=seed
    )

    print(f"\nTrain samples: {len(dls.train_ds)}")
    print(f"Valid samples: {len(dls.valid_ds)}")
    print(f"Number of classes: {dls.c}")
    print()

    return dls


def create_model(dls, hidden_layers=[1024, 512, 512, 256, 256, 128, 128, 64, 64, 32],
                 ps=[0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]):
    """
    Create FastAI learner with custom MLP layers

    Args:
        dls: DataLoaders
        hidden_layers: List of hidden layer sizes (default: 10 dense layers)
        ps: Dropout probabilities for each layer
    """
    print("=" * 80)
    print("CREATING MODEL")
    print("=" * 80)

    print(f"Architecture:")
    print(f"  Input features: {len(dls.cont_names)}")
    print(f"  Hidden layers ({len(hidden_layers)} layers): {hidden_layers}")
    print(f"  Dropout rates: {ps}")
    print(f"  Output classes: {dls.c}")
    print()

    # Create TabularLearner with custom architecture
    learn = tabular_learner(
        dls,
        layers=hidden_layers,
        metrics=[accuracy, error_rate],
        config=tabular_config(
            ps=ps,
            embed_p=0.0,  # No embeddings since we have no categorical features
            use_bn=True,   # Use batch normalization
        )
    )

    return learn


def find_learning_rate(learn):
    """Use FastAI's learning rate finder"""
    print("=" * 80)
    print("FINDING OPTIMAL LEARNING RATE")
    print("=" * 80)

    # Run learning rate finder
    print("Running lr_find()...")
    lr_find_result = learn.lr_find()

    # Get suggested learning rate
    suggested_lr = lr_find_result.valley
    print(f"\nSuggested learning rate: {suggested_lr:.2e}")

    # Plot learning rate finder
    fig = learn.recorder.plot_lr_find()
    plt.savefig('claude/encodec_lr_finder.png', dpi=150, bbox_inches='tight')
    print(f"Learning rate finder plot saved to: claude/encodec_lr_finder.png")
    plt.close()

    return suggested_lr


def train_model(learn, lr, epochs=20):
    """Train the model for specified epochs"""
    print("=" * 80)
    print(f"TRAINING FOR {epochs} EPOCHS")
    print("=" * 80)

    print(f"Learning rate: {lr:.2e}")
    print(f"Epochs: {epochs}")
    print()

    # Train with fit_one_cycle
    learn.fit_one_cycle(epochs, lr)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    return learn


def evaluate_model(learn):
    """Evaluate model and show results"""
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Get final metrics
    final_loss, final_acc, final_err = learn.validate()

    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Error Rate: {final_err:.4f}")
    print()

    # Show confusion matrix
    print("Generating confusion matrix...")
    interp = ClassificationInterpretation.from_learner(learn)

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(20, 20))
    interp.plot_confusion_matrix(figsize=(20, 20), dpi=60)
    plt.savefig('claude/encodec_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: claude/encodec_confusion_matrix.png")
    plt.close()

    # Top losses
    print("\nTop 10 losses:")
    interp.plot_top_losses(9, figsize=(15, 10))
    plt.savefig('claude/encodec_top_losses.png', dpi=150, bbox_inches='tight')
    print(f"Top losses plot saved to: claude/encodec_top_losses.png")
    plt.close()

    return final_acc, final_loss


def save_training_history(learn, output_dir='./claude'):
    """Save training history and plots"""
    print("=" * 80)
    print("SAVING TRAINING HISTORY")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save training history plot
    learn.recorder.plot_loss()
    plt.savefig(output_path / 'encodec_training_loss.png', dpi=150, bbox_inches='tight')
    print(f"Training loss plot saved to: {output_path}/encodec_training_loss.png")
    plt.close()

    # Save model
    model_path = output_path / 'encodec_fastai_model'
    learn.export(model_path.with_suffix('.pkl'))
    print(f"Model saved to: {model_path}.pkl")

    # Save training history to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_file = output_path / f'encodec_training_history_{timestamp}.csv'

    # Extract training history
    history_data = {
        'epoch': list(range(len(learn.recorder.values))),
        'train_loss': [v[0] for v in learn.recorder.values],
        'valid_loss': [v[1] for v in learn.recorder.values],
        'accuracy': [v[2] if len(v) > 2 else None for v in learn.recorder.values],
        'error_rate': [v[3] if len(v) > 3 else None for v in learn.recorder.values],
    }
    history_df = pd.DataFrame(history_data)
    history_df.to_csv(history_file, index=False)
    print(f"Training history saved to: {history_file}")

    return history_df


def save_summary_report(final_acc, final_loss, lr, epochs, df, history_df, output_dir='./claude'):
    """Save a summary report"""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_path / f'encodec_fastai_report_{timestamp}.txt'

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FASTAI TRANSFER LEARNING - ENCODEC TOKEN EMBEDDINGS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Number of classes: {df['target'].nunique()}\n")
        f.write(f"Number of features: {len(df.columns) - 1}\n\n")

        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Input features: 1200 (EnCodec tokens)\n")
        f.write(f"Hidden layers (10 layers): [1024, 512, 512, 256, 256, 128, 128, 64, 64, 32]\n")
        f.write(f"Dropout rates: [0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]\n")
        f.write(f"Batch normalization: Yes\n\n")

        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Learning rate: {lr:.2e}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Optimizer: Adam (via fit_one_cycle)\n\n")

        f.write("FINAL RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Final Validation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)\n")
        f.write(f"Final Validation Loss: {final_loss:.4f}\n")
        f.write(f"Final Error Rate: {1-final_acc:.4f}\n\n")

        f.write("TRAINING HISTORY (Last 5 Epochs)\n")
        f.write("-" * 80 + "\n")
        f.write(history_df.tail(5).to_string(index=False))
        f.write("\n\n")

        f.write("BEST EPOCH\n")
        f.write("-" * 80 + "\n")
        best_epoch = history_df.loc[history_df['accuracy'].idxmax()]
        f.write(f"Epoch: {int(best_epoch['epoch'])}\n")
        f.write(f"Accuracy: {best_epoch['accuracy']:.4f}\n")
        f.write(f"Validation Loss: {best_epoch['valid_loss']:.4f}\n")

    print(f"Summary report saved to: {report_file}")
    return report_file


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("FASTAI TRANSFER LEARNING - ENCODEC TOKEN EMBEDDINGS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Set random seed
    set_seed(42)

    # Load data
    df, token_cols = load_encodec_data()

    # Create dataloaders
    dls = create_dataloaders(df, token_cols, valid_pct=0.2, bs=64)

    # Create model with 10 dense MLP layers
    learn = create_model(dls)

    # Find optimal learning rate
    lr = find_learning_rate(learn)

    # Train for 20 epochs
    learn = train_model(learn, lr, epochs=20)

    # Evaluate
    final_acc, final_loss = evaluate_model(learn)

    # Save training history
    history_df = save_training_history(learn)

    # Save summary report
    save_summary_report(final_acc, final_loss, lr, 20, df, history_df)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal Validation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
