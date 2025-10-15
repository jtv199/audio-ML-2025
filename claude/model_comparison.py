#!/usr/bin/env python3
"""
Comprehensive Model Comparison Script
Trains and evaluates 24+ sklearn classifiers in 4 segments with timeout protection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import signal
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Import all classifiers
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (LogisticRegression, RidgeClassifier, RidgeClassifierCV,
                                   PassiveAggressiveClassifier, Perceptron, SGDClassifier)
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import NearestCentroid
from sklearn.dummy import DummyClassifier

import warnings
warnings.filterwarnings('ignore')


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError()


def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test, timeout=300):
    """
    Train and evaluate a single model with timeout protection.

    Args:
        name: Model name
        model: sklearn model instance
        X_train, X_test, y_train, y_test: Train/test splits
        timeout: Maximum time in seconds (default 300 = 5 minutes)

    Returns:
        Dictionary with model results
    """
    print(f"  Training {name}...", end=" ", flush=True)

    start_time = time.time()

    try:
        # Set timeout alarm (only works on Unix-like systems)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Cancel alarm
        signal.alarm(0)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        elapsed_time = time.time() - start_time

        print(f"✓ Done in {elapsed_time:.2f}s (Acc: {accuracy:.4f})")

        return {
            'Model': name,
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_acc,
            'F1 Score': f1,
            'Time Taken': elapsed_time,
            'Status': 'Success'
        }

    except TimeoutError:
        signal.alarm(0)
        print(f"✗ TIMEOUT after {timeout}s")
        return {
            'Model': name,
            'Accuracy': None,
            'Balanced Accuracy': None,
            'F1 Score': None,
            'Time Taken': timeout,
            'Status': 'Timeout'
        }
    except Exception as e:
        signal.alarm(0)
        elapsed_time = time.time() - start_time
        error_msg = str(e)[:100]
        print(f"✗ ERROR: {error_msg}")
        return {
            'Model': name,
            'Accuracy': None,
            'Balanced Accuracy': None,
            'F1 Score': None,
            'Time Taken': elapsed_time,
            'Status': f'Error: {error_msg}'
        }


def load_and_prepare_data():
    """Load features and labels, prepare train/test split"""
    print("Loading data...")

    # Define paths
    DATA_PATH = Path('./work')
    FEATURES_CSV = DATA_PATH / 'trn_curated_feature.csv'
    LABELS_CSV = Path('./input/train_curated.csv')

    # Load features
    print(f"  Loading features from {FEATURES_CSV}")
    features_df = pd.read_csv(FEATURES_CSV)
    print(f"  Features shape: {features_df.shape}")

    # Load labels
    print(f"  Loading labels from {LABELS_CSV}")
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"  Labels shape: {labels_df.shape}")

    # Merge features and labels
    df = features_df.merge(labels_df, left_on='file', right_on='fname', how='inner')
    print(f"  Merged dataframe shape: {df.shape}")

    # Filter to single-class samples
    df['num_labels'] = df['labels'].str.count(',') + 1
    single_class_df = df[df['num_labels'] == 1].copy()
    print(f"\n  Original samples: {len(df)}")
    print(f"  Single-class samples: {len(single_class_df)} ({100 * len(single_class_df) / len(df):.1f}%)")
    print(f"  Number of unique classes: {single_class_df['labels'].nunique()}")

    # Prepare features and target
    feature_cols = [col for col in single_class_df.columns
                    if col not in ['file', 'fname', 'labels', 'num_labels']]

    X = single_class_df[feature_cols]
    y = single_class_df['labels']

    # Remove rows with missing values
    if X.isnull().any().any():
        print(f"\n  Removing rows with missing values...")
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        print(f"  After removal - X shape: {X.shape}, y shape: {y.shape}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Training set: {len(X_train)} samples ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({100*len(X_test)/len(X):.1f}%)")
    print(f"  Number of classes: {y.nunique()}")
    print()

    return X_train, X_test, y_train, y_test


def define_model_segments():
    """Define model segments with their respective classifiers"""

    # Segment 1: Fast Naive Bayes and simple models
    segment_1 = [
        ('DummyClassifier', DummyClassifier(strategy='most_frequent', random_state=42)),
        ('BernoulliNB', BernoulliNB()),
        ('GaussianNB', GaussianNB()),
        ('NearestCentroid', NearestCentroid()),
        ('Perceptron', Perceptron(random_state=42, max_iter=1000)),
    ]

    # Segment 2: Linear models
    segment_2 = [
        ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)),
        ('RidgeClassifier', RidgeClassifier(random_state=42)),
        ('RidgeClassifierCV', RidgeClassifierCV()),
        ('PassiveAggressiveClassifier', PassiveAggressiveClassifier(random_state=42, max_iter=1000)),
        ('SGDClassifier', SGDClassifier(random_state=42, max_iter=1000)),
        ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()),
        ('LinearSVC', LinearSVC(random_state=42, max_iter=1000, dual=False)),
        ('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()),
    ]

    # Segment 3: Tree-based and neighbor models
    segment_3 = [
        ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=42)),
        ('ExtraTreeClassifier', ExtraTreeClassifier(random_state=42)),
        ('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ('RandomForestClassifier', RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)),
        ('ExtraTreesClassifier', ExtraTreesClassifier(random_state=42, n_estimators=100, n_jobs=-1)),
    ]

    # Segment 4: Ensemble and complex models
    segment_4 = [
        ('AdaBoostClassifier', AdaBoostClassifier(random_state=42, algorithm='SAMME')),
        ('BaggingClassifier', BaggingClassifier(random_state=42, n_jobs=-1)),
        ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=42)),
        ('CalibratedClassifierCV', CalibratedClassifierCV(n_jobs=-1)),
    ]

    return [
        ("Segment 1: Fast Models", segment_1, 180),
        ("Segment 2: Linear Models", segment_2, 300),
        ("Segment 3: Tree & Neighbor Models", segment_3, 600),
        ("Segment 4: Ensemble Models", segment_4, 900),
    ]


def run_segment(segment_name, models_list, timeout, X_train, X_test, y_train, y_test):
    """Run all models in a segment"""
    print("=" * 80)
    print(f"{segment_name} ({len(models_list)} models)")
    print("=" * 80)

    results = []
    for model_name, model in models_list:
        result = train_and_evaluate_model(
            model_name, model, X_train, X_test, y_train, y_test, timeout=timeout
        )
        results.append(result)

    # Convert to DataFrame and display
    df = pd.DataFrame(results).set_index('Model')
    print("\n" + "-" * 80)
    print(f"{segment_name} Results:")
    print("-" * 80)
    print(df[['Accuracy', 'Balanced Accuracy', 'F1 Score', 'Time Taken', 'Status']])
    print()

    return results


def save_results(all_results, output_dir='./claude'):
    """Save results to CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(all_results).set_index('Model')

    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results_file = output_path / f'model_comparison_results_{timestamp}.csv'
    df.to_csv(all_results_file)
    print(f"All results saved to: {all_results_file}")

    # Save successful models only
    successful = df[df['Status'] == 'Success']
    if len(successful) > 0:
        successful_file = output_path / f'model_comparison_successful_{timestamp}.csv'
        successful.to_csv(successful_file)
        print(f"Successful models saved to: {successful_file}")

        # Save top 10
        top_10 = successful.sort_values('Accuracy', ascending=False).head(10)
        top_10_file = output_path / f'model_comparison_top10_{timestamp}.csv'
        top_10.to_csv(top_10_file)
        print(f"Top 10 models saved to: {top_10_file}")

    return df


def main():
    """Main execution function"""
    print("=" * 80)
    print("SKLEARN MODEL COMPARISON - AUDIO CLASSIFICATION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Set random seed
    np.random.seed(42)

    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Define model segments
    segments = define_model_segments()

    # Display segment overview
    print("=" * 80)
    print("MODEL SEGMENTS OVERVIEW")
    print("=" * 80)
    total_models = 0
    for segment_name, models_list, timeout in segments:
        print(f"\n{segment_name} ({len(models_list)} models, timeout: {timeout}s):")
        for model_name, _ in models_list:
            print(f"  - {model_name}")
        total_models += len(models_list)
    print(f"\nTOTAL: {total_models} models")
    print()

    # Run all segments
    all_results = []
    for segment_name, models_list, timeout in segments:
        segment_results = run_segment(
            segment_name, models_list, timeout, X_train, X_test, y_train, y_test
        )
        all_results.extend(segment_results)

    # Display combined results
    print("=" * 80)
    print("COMBINED RESULTS - ALL MODELS")
    print("=" * 80)

    df = pd.DataFrame(all_results).set_index('Model')
    successful = df[df['Status'] == 'Success']
    failed = df[df['Status'] != 'Success']

    print(f"\nSummary: {len(successful)}/{len(df)} models completed successfully\n")

    if len(successful) > 0:
        print("Top 10 Models by Accuracy:")
        print("-" * 80)
        top_10 = successful.sort_values('Accuracy', ascending=False).head(10)
        print(top_10[['Accuracy', 'Balanced Accuracy', 'F1 Score', 'Time Taken']])
        print()

    if len(failed) > 0:
        print("\nFailed/Timeout Models:")
        print("-" * 80)
        print(failed[['Time Taken', 'Status']])
        print()

    # Save results
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    save_results(all_results)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
