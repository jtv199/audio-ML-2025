#!/usr/bin/env python3
"""
Error Analysis Script
Analyzes which classes are most accurate and most mispredicted for the best models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_recall_fscore_support)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ERROR ANALYSIS - AUDIO CLASSIFICATION")
print("=" * 80)
print()


def load_and_prepare_data():
    """Load features and labels, prepare train/test split"""
    print("Loading data...")

    # Define paths
    DATA_PATH = Path('./work')
    FEATURES_CSV = DATA_PATH / 'trn_curated_feature.csv'
    LABELS_CSV = Path('./input/train_curated.csv')

    # Load features
    features_df = pd.read_csv(FEATURES_CSV)
    labels_df = pd.read_csv(LABELS_CSV)

    # Merge features and labels
    df = features_df.merge(labels_df, left_on='file', right_on='fname', how='inner')

    # Filter to single-class samples
    df['num_labels'] = df['labels'].str.count(',') + 1
    single_class_df = df[df['num_labels'] == 1].copy()

    # Prepare features and target
    feature_cols = [col for col in single_class_df.columns
                    if col not in ['file', 'fname', 'labels', 'num_labels']]

    X = single_class_df[feature_cols]
    y = single_class_df['labels']

    # Remove rows with missing values
    if X.isnull().any().any():
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Number of classes: {y.nunique()}")
    print()

    return X_train, X_test, y_train, y_test


def train_model(name, model, X_train, y_train):
    """Train a model and return it"""
    print(f"Training {name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    print("✓")
    return model


def analyze_per_class_performance(y_true, y_pred, model_name):
    """Analyze performance per class"""
    print(f"\n{'=' * 80}")
    print(f"Per-Class Analysis: {model_name}")
    print(f"{'=' * 80}\n")

    # Get all unique classes
    classes = sorted(y_true.unique())

    # Calculate metrics per class
    results = []
    for cls in classes:
        # Get indices for this class
        cls_indices = y_true == cls

        # True positives, false positives, false negatives
        tp = ((y_true == cls) & (y_pred == cls)).sum()
        fp = ((y_true != cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()
        tn = ((y_true != cls) & (y_pred != cls)).sum()

        # Calculate metrics
        total_actual = cls_indices.sum()
        correct = (y_true[cls_indices] == y_pred[cls_indices]).sum()
        accuracy = correct / total_actual if total_actual > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'Class': cls,
            'Total_Samples': total_actual,
            'Correct': correct,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'TP': tp,
            'FP': fp,
            'FN': fn
        })

    df_results = pd.DataFrame(results)
    return df_results


def analyze_confusion_patterns(y_true, y_pred, class_name):
    """Analyze what a specific class gets confused with"""
    # Get indices where true class is class_name
    class_indices = y_true == class_name

    if class_indices.sum() == 0:
        return None

    # Get predictions for this class
    predictions_for_class = y_pred[class_indices]
    true_labels = y_true[class_indices]

    # Count mispredictions
    confusion_counts = pd.Series(predictions_for_class).value_counts()

    return confusion_counts


def generate_report(X_train, X_test, y_train, y_test):
    """Generate comprehensive error analysis report"""

    # Train top 3 models
    models = {
        'LinearSVC': LinearSVC(random_state=42, max_iter=1000, dual=False),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    }

    all_results = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"Analyzing: {model_name}")
        print(f"{'=' * 80}")

        # Train model
        trained_model = train_model(model_name, model, X_train, y_train)

        # Make predictions
        y_pred = trained_model.predict(X_test)

        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Per-class analysis
        df_class_perf = analyze_per_class_performance(y_test, y_pred, model_name)
        all_results[model_name] = {
            'model': trained_model,
            'predictions': y_pred,
            'class_performance': df_class_perf,
            'overall_accuracy': accuracy
        }

        # Display best and worst classes
        print(f"\n{'-' * 80}")
        print(f"TOP 5 MOST ACCURATE CLASSES:")
        print(f"{'-' * 80}")
        top_5 = df_class_perf.nlargest(5, 'Accuracy')[['Class', 'Accuracy', 'Total_Samples', 'Precision', 'Recall', 'F1_Score']]
        print(top_5.to_string(index=False))

        print(f"\n{'-' * 80}")
        print(f"BOTTOM 5 LEAST ACCURATE CLASSES:")
        print(f"{'-' * 80}")
        bottom_5 = df_class_perf.nsmallest(5, 'Accuracy')[['Class', 'Accuracy', 'Total_Samples', 'Precision', 'Recall', 'F1_Score']]
        print(bottom_5.to_string(index=False))

        # Most confused classes
        print(f"\n{'-' * 80}")
        print(f"CLASSES WITH MOST FALSE POSITIVES (wrongly predicted as this class):")
        print(f"{'-' * 80}")
        most_fp = df_class_perf.nlargest(5, 'FP')[['Class', 'FP', 'Precision', 'Recall']]
        print(most_fp.to_string(index=False))

        print(f"\n{'-' * 80}")
        print(f"CLASSES WITH MOST FALSE NEGATIVES (failed to predict this class):")
        print(f"{'-' * 80}")
        most_fn = df_class_perf.nlargest(5, 'FN')[['Class', 'FN', 'Precision', 'Recall']]
        print(most_fn.to_string(index=False))

    return all_results, y_test


def analyze_specific_class_confusions(results, y_test, model_name='LinearSVC'):
    """Deep dive into specific class confusion patterns"""
    print(f"\n{'=' * 80}")
    print(f"DETAILED CONFUSION ANALYSIS - {model_name}")
    print(f"{'=' * 80}\n")

    model_results = results[model_name]
    y_pred = model_results['predictions']
    df_perf = model_results['class_performance']

    # Get worst performing class
    worst_class = df_perf.nsmallest(1, 'Accuracy').iloc[0]['Class']

    print(f"WORST CLASS: {worst_class}")
    print(f"Accuracy: {df_perf[df_perf['Class'] == worst_class]['Accuracy'].values[0]:.4f}")
    print(f"\nWhat does '{worst_class}' get predicted as?")
    print("-" * 60)

    confusion = analyze_confusion_patterns(y_test, y_pred, worst_class)
    if confusion is not None:
        for pred_class, count in confusion.head(10).items():
            percentage = (count / confusion.sum()) * 100
            marker = "✓" if pred_class == worst_class else "✗"
            print(f"  {marker} {pred_class:40s} {count:3d} times ({percentage:5.1f}%)")

    # Get best performing class
    best_class = df_perf.nlargest(1, 'Accuracy').iloc[0]['Class']

    print(f"\n\nBEST CLASS: {best_class}")
    print(f"Accuracy: {df_perf[df_perf['Class'] == best_class]['Accuracy'].values[0]:.4f}")
    print(f"\nWhat does '{best_class}' get predicted as?")
    print("-" * 60)

    confusion = analyze_confusion_patterns(y_test, y_pred, best_class)
    if confusion is not None:
        for pred_class, count in confusion.head(10).items():
            percentage = (count / confusion.sum()) * 100
            marker = "✓" if pred_class == best_class else "✗"
            print(f"  {marker} {pred_class:40s} {count:3d} times ({percentage:5.1f}%)")

    return worst_class, best_class


def save_detailed_results(results, output_dir='./claude/lazypredict/findings'):
    """Save detailed results to CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}\n")

    for model_name, model_data in results.items():
        df = model_data['class_performance']

        # Save full results
        filename = f"error_analysis_{model_name.lower()}.csv"
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {model_name} analysis to: {filepath}")

    # Create summary comparison
    summary_data = []
    for model_name, model_data in results.items():
        df = model_data['class_performance']
        summary_data.append({
            'Model': model_name,
            'Overall_Accuracy': model_data['overall_accuracy'],
            'Mean_Class_Accuracy': df['Accuracy'].mean(),
            'Median_Class_Accuracy': df['Accuracy'].median(),
            'Std_Class_Accuracy': df['Accuracy'].std(),
            'Best_Class': df.nlargest(1, 'Accuracy').iloc[0]['Class'],
            'Best_Class_Accuracy': df.nlargest(1, 'Accuracy').iloc[0]['Accuracy'],
            'Worst_Class': df.nsmallest(1, 'Accuracy').iloc[0]['Class'],
            'Worst_Class_Accuracy': df.nsmallest(1, 'Accuracy').iloc[0]['Accuracy']
        })

    df_summary = pd.DataFrame(summary_data)
    summary_file = output_path / 'error_analysis_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Saved summary to: {summary_file}")

    print()


def create_visualizations(results, y_test, output_dir='./claude/lazypredict/findings'):
    """Create visualization plots"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'=' * 80}\n")

    # For best model (LinearSVC)
    model_name = 'LinearSVC'
    if model_name in results:
        df_perf = results[model_name]['class_performance']

        # 1. Accuracy distribution
        plt.figure(figsize=(12, 6))
        plt.hist(df_perf['Accuracy'], bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(df_perf['Accuracy'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df_perf["Accuracy"].mean():.3f}')
        plt.axvline(df_perf['Accuracy'].median(), color='green', linestyle='--',
                   label=f'Median: {df_perf["Accuracy"].median():.3f}')
        plt.xlabel('Per-Class Accuracy')
        plt.ylabel('Number of Classes')
        plt.title(f'Distribution of Per-Class Accuracies - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_file = output_path / f'accuracy_distribution_{model_name.lower()}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved accuracy distribution plot: {plot_file}")
        plt.close()

        # 2. Top and bottom classes comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Top 10 classes
        top_10 = df_perf.nlargest(10, 'Accuracy').sort_values('Accuracy')
        ax1.barh(range(len(top_10)), top_10['Accuracy'], color='green', alpha=0.7)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels(top_10['Class'], fontsize=8)
        ax1.set_xlabel('Accuracy')
        ax1.set_title(f'Top 10 Most Accurate Classes - {model_name}')
        ax1.grid(True, alpha=0.3, axis='x')

        # Bottom 10 classes
        bottom_10 = df_perf.nsmallest(10, 'Accuracy').sort_values('Accuracy', ascending=False)
        ax2.barh(range(len(bottom_10)), bottom_10['Accuracy'], color='red', alpha=0.7)
        ax2.set_yticks(range(len(bottom_10)))
        ax2.set_yticklabels(bottom_10['Class'], fontsize=8)
        ax2.set_xlabel('Accuracy')
        ax2.set_title(f'Bottom 10 Least Accurate Classes - {model_name}')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plot_file = output_path / f'top_bottom_classes_{model_name.lower()}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved top/bottom classes plot: {plot_file}")
        plt.close()

        # 3. Precision vs Recall scatter
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df_perf['Recall'], df_perf['Precision'],
                            c=df_perf['Accuracy'], cmap='RdYlGn',
                            s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Accuracy')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision vs Recall per Class - {model_name}')
        plt.grid(True, alpha=0.3)
        plot_file = output_path / f'precision_recall_scatter_{model_name.lower()}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved precision-recall scatter: {plot_file}")
        plt.close()

    print()


def main():
    """Main execution function"""

    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Generate error analysis report
    results, y_test = generate_report(X_train, X_test, y_train, y_test)

    # Deep dive into specific confusions
    worst_class, best_class = analyze_specific_class_confusions(results, y_test)

    # Save results
    save_detailed_results(results)

    # Create visualizations
    create_visualizations(results, y_test)

    print("=" * 80)
    print("ERROR ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nKey Findings:")
    print(f"  - Best performing class: {best_class}")
    print(f"  - Worst performing class: {worst_class}")
    print(f"\nResults saved to: claude/lazypredict/findings/")
    print()


if __name__ == "__main__":
    main()
