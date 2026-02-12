#!/usr/bin/env python3
"""
FastAI Tabular Predictor for VGGish Embeddings - SINGLE LABEL ONLY
Train models on single-label samples only (filtering out multi-label samples)
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
torch.set_num_threads(4)  # Limit CPU threads


def prepare_data():
    """Load and prepare the VGGish embeddings data - SINGLE LABEL ONLY"""
    print("="*80)
    print("LOADING DATA (SINGLE-LABEL ONLY)")
    print("="*80)

    # Load embeddings
    train_df = pd.read_csv('work/tokenized/vggish_embeddings_train_curated.csv')
    print(f"Train data loaded: {train_df.shape}")

    # Filter to single-label samples only
    train_df['num_labels'] = train_df['labels'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

    print(f"\nLabel distribution before filtering:")
    print(train_df['num_labels'].value_counts().sort_index())

    # Keep only single-label samples
    single_label_df = train_df[train_df['num_labels'] == 1].copy()

    print(f"\n{'='*80}")
    print(f"FILTERED TO SINGLE-LABEL SAMPLES")
    print(f"{'='*80}")
    print(f"Original samples: {len(train_df)}")
    print(f"Single-label samples: {len(single_label_df)} ({len(single_label_df)/len(train_df)*100:.1f}%)")
    print(f"Removed: {len(train_df) - len(single_label_df)} multi-label samples")

    # Get unique classes
    unique_labels = sorted(single_label_df['labels'].unique())
    print(f"\nUnique classes: {len(unique_labels)}")
    print(f"Sample classes: {unique_labels[:10]}...")

    # Count samples per class
    class_counts = single_label_df['labels'].value_counts()
    print(f"\nClass distribution:")
    print(f"  Min samples per class: {class_counts.min()}")
    print(f"  Max samples per class: {class_counts.max()}")
    print(f"  Mean samples per class: {class_counts.mean():.1f}")
    print(f"  Median samples per class: {class_counts.median():.1f}")

    # Get embedding columns
    emb_cols = [f'emb_{i}' for i in range(128)]

    print(f"\nEmbedding columns: {len(emb_cols)}")
    print(f"Total features: {len(single_label_df.columns)}")

    # Reset index after filtering
    single_label_df = single_label_df.reset_index(drop=True)

    # Split into train and validation
    train_idx, val_idx = train_test_split(
        range(len(single_label_df)),
        test_size=0.2,
        random_state=42,
        stratify=single_label_df['labels']  # Stratified split
    )

    single_label_df['is_valid'] = False
    single_label_df.loc[val_idx, 'is_valid'] = True

    print(f"\nTrain samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")

    return single_label_df, emb_cols, unique_labels


def create_tabular_learner(df, emb_cols, layers, lr=1e-3):
    """Create a FastAI tabular learner for single-label classification"""

    # Create TabularDataLoaders
    dls = TabularDataLoaders.from_df(
        df,
        path='.',
        procs=[Categorify, Normalize],
        cat_names=[],  # No categorical features
        cont_names=emb_cols,  # All embedding columns are continuous
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

    return learn, dls


def train_and_evaluate(df, emb_cols, unique_labels, layers, epochs=20):
    """Train and evaluate a model"""

    print(f"\n{'='*80}")
    print(f"TRAINING MODEL WITH {len(layers)} LAYERS: {layers}")
    print(f"{'='*80}")

    # Create learner
    learn, dls = create_tabular_learner(df, emb_cols, layers)

    print(f"\nModel architecture:")
    print(learn.model)

    num_params = sum(p.numel() for p in learn.model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

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

    print(f"\nValidation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

    # Get class names first (needed for results dict)
    try:
        vocab = dls.vocab  # Get label vocabulary
        if isinstance(vocab, tuple):
            vocab = vocab[1]  # Extract labels from tuple
        vocab = list(vocab)  # Convert to list
    except:
        try:
            vocab = list(dls.train_ds.y.vocab)  # Try alternative way
        except:
            vocab = list(range(74))  # Fallback to indices

    # Perform detailed per-class error analysis
    print("\n" + "="*80)
    print("DETAILED PER-CLASS ERROR ANALYSIS")
    print("="*80)

    val_df = df[df['is_valid']].copy()

    # Get predictions and true labels
    y_pred_indices = val_preds_class.cpu().numpy()
    y_true_indices = val_targets.cpu().numpy()

    # Debug: print indices range and vocab length
    print(f"Debug: Vocab length: {len(vocab)}")
    print(f"Debug: Pred indices range: {y_pred_indices.min()} to {y_pred_indices.max()}")
    print(f"Debug: True indices range: {y_true_indices.min()} to {y_true_indices.max()}")

    # Convert to class names - ensure indices are within bounds
    y_pred = []
    y_true = []
    for i in y_pred_indices:
        idx = int(i)  # Convert to Python int
        if idx < len(vocab):
            y_pred.append(vocab[idx])
        else:
            y_pred.append(f"Unknown_{idx}")

    for i in y_true_indices:
        idx = int(i)  # Convert to Python int
        if idx < len(vocab):
            y_true.append(vocab[idx])
        else:
            y_true.append(f"Unknown_{idx}")

    # Create per-class analysis
    from sklearn.metrics import precision_score, recall_score, f1_score

    class_results = []
    for class_name in sorted(set(y_true)):
        # Get indices for this class
        class_mask = [i for i, label in enumerate(y_true) if label == class_name]

        if len(class_mask) == 0:
            continue

        # Calculate metrics for this class
        y_true_class = [1 if label == class_name else 0 for label in y_true]
        y_pred_class = [1 if label == class_name else 0 for label in y_pred]

        precision = precision_score(y_true_class, y_pred_class, zero_division=0)
        recall = recall_score(y_true_class, y_pred_class, zero_division=0)
        f1 = f1_score(y_true_class, y_pred_class, zero_division=0)

        # Calculate per-class accuracy
        correct = sum([1 for i in class_mask if y_pred[i] == y_true[i]])
        total = len(class_mask)
        accuracy = correct / total if total > 0 else 0

        class_results.append({
            'Class': class_name,
            'Total_Samples': total,
            'Correct': correct,
            'Incorrect': total - correct,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        })

    class_analysis_df = pd.DataFrame(class_results)
    class_analysis_df = class_analysis_df.sort_values('Accuracy', ascending=False)

    # Print summary statistics
    print(f"\nTotal classes analyzed: {len(class_analysis_df)}")
    print(f"Classes with 100% accuracy: {len(class_analysis_df[class_analysis_df['Accuracy'] == 1.0])}")
    print(f"Classes with 0% accuracy: {len(class_analysis_df[class_analysis_df['Accuracy'] == 0.0])}")
    print(f"\nMean accuracy across classes: {class_analysis_df['Accuracy'].mean():.4f}")
    print(f"Median accuracy: {class_analysis_df['Accuracy'].median():.4f}")

    # Top 10 classes
    print("\n" + "="*80)
    print("TOP 10 CLASSES BY ACCURACY")
    print("="*80)
    print(class_analysis_df.head(10).to_string(index=False))

    # Bottom 10 classes
    print("\n" + "="*80)
    print("BOTTOM 10 CLASSES BY ACCURACY")
    print("="*80)
    print(class_analysis_df.tail(10).to_string(index=False))

    # Save per-class analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_file = f"vggish_single_label_per_class_{len(layers)}layers_{timestamp}.csv"
    class_analysis_df.to_csv(analysis_file, index=False)
    print(f"\nPer-class analysis saved to: {analysis_file}")

    # Save results
    results = {
        'layers': str(layers),
        'num_layers': len(layers),
        'epochs': epochs,
        'training_time': training_time,
        'val_accuracy': val_accuracy,
        'lr_min': float(lr_min),
        'lr_steep': float(lr_steep),
        'num_params': num_params,
        'num_classes': len(vocab),
        'perfect_classes': len(class_analysis_df[class_analysis_df['Accuracy'] == 1.0]),
        'timestamp': timestamp
    }

    # Save model to models/ directory
    from pathlib import Path
    Path("models").mkdir(exist_ok=True)

    model_name = f"vggish_single_label_{len(layers)}layers_{timestamp}"
    # FastAI adds "models/" prefix automatically, so just use the name
    learn.export(f"models/{model_name}.pkl")  # Export instead of save
    print(f"\nModel exported as: models/{model_name}.pkl")

    return results, learn, class_analysis_df


def main():
    """Main execution"""

    print("="*80)
    print("FASTAI TABULAR PREDICTOR ON VGGISH EMBEDDINGS")
    print("SINGLE-LABEL CLASSIFICATION ONLY")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Prepare data
    df, emb_cols, unique_labels = prepare_data()

    # Train models with different layer configurations
    layer_configs = [
        [200, 100],  # 2 layers
        [400, 300, 200, 150, 100],  # 5 layers
        [512, 256],  # 2 layers with more neurons
    ]

    all_results = []
    all_class_analyses = []

    for layers in layer_configs:
        results, learner, class_analysis_df = train_and_evaluate(
            df, emb_cols, unique_labels,
            layers=layers,
            epochs=20
        )
        all_results.append(results)
        class_analysis_df['model'] = f"{len(layers)}_layers_{layers}"
        all_class_analyses.append(class_analysis_df)

    # Save results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(all_results)
    print("\n", results_df.to_string())

    # Save to CSV
    output_file = f"vggish_single_label_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save per-class analyses
    analysis_output_file = f"vggish_single_label_all_class_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    all_analyses_df = pd.concat(all_class_analyses, ignore_index=True)
    all_analyses_df.to_csv(analysis_output_file, index=False)
    print(f"All per-class analyses saved to: {analysis_output_file}")

    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    for idx, row in results_df.iterrows():
        print(f"\nModel {idx+1}: {row['num_layers']} layers {row['layers']}")
        print(f"  Parameters: {row['num_params']:,}")
        print(f"  Training time: {row['training_time']:.2f}s ({row['training_time']/60:.2f} min)")
        print(f"  Val Accuracy: {row['val_accuracy']:.4f} ({row['val_accuracy']*100:.2f}%)")
        print(f"  Perfect classes: {row['perfect_classes']}")

    # Find best model
    best_idx = results_df['val_accuracy'].idxmax()
    best_model = results_df.iloc[best_idx]

    print("\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    print(f"Configuration: {best_model['layers']}")
    print(f"Validation Accuracy: {best_model['val_accuracy']:.4f} ({best_model['val_accuracy']*100:.2f}%)")
    print(f"Parameters: {best_model['num_params']:,}")
    print(f"Training time: {best_model['training_time']:.2f}s")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
