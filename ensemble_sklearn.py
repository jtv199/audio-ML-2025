#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, hamming_loss
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ENSEMBLE MODEL: 2-Layer NN on Tabular Features")
print("(Note: Simplified version without CNN features)")
print("="*60)

# Configuration
DATA = Path('./input')
WORK = Path('work')
SAMPLE_SIZE = 100  # Use 100 samples as requested

# Load data
print("\n1. Loading data...")
df = pd.read_csv(DATA/'train_curated.csv').sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
features_df = pd.read_csv(WORK/'trn_curated_feature.csv')

# Merge features with labels
# Note: features CSV has 'file' column, not 'fname'
features_df = features_df.rename(columns={'file': 'fname'})
features_df = features_df.merge(df[['fname', 'labels']], on='fname')
print(f"   - Loaded {len(features_df)} samples")

# Prepare labels
mlb = MultiLabelBinarizer()
labels_split = features_df['labels'].str.split(',')
y = mlb.fit_transform(labels_split)
n_classes = len(mlb.classes_)

print(f"   - Number of classes: {n_classes}")

# Prepare tabular features (excluding fname and labels columns)
feature_cols = [col for col in features_df.columns if col not in ['fname', 'labels']]
X_tabular = features_df[feature_cols].values
print(f"   - Tabular feature dimension: {X_tabular.shape[1]}")

# Split data
X_train, X_test, y_train, y_test, fname_train, fname_test = train_test_split(
    X_tabular, y, features_df['fname'].values, 
    test_size=0.2, random_state=42
)

# Scale tabular features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n2. Data split:")
print(f"   - Training samples: {len(X_train)}")
print(f"   - Test samples: {len(X_test)}")

# Train 2-layer NN using sklearn's MLPClassifier
print("\n3. Training 2-layer Neural Network...")
print("   (Using scikit-learn's MLPClassifier)")

# Create and train the model
# MLPClassifier with one hidden layer = 2 layer NN (1 hidden + 1 output)
model = MLPClassifier(
    hidden_layer_sizes=(256,),  # Single hidden layer with 256 units
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    verbose=True
)

model.fit(X_train_scaled, y_train)

# Evaluate
print("\n4. Evaluating model...")
train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)

# Calculate metrics
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
train_hamming = 1 - hamming_loss(y_train, train_preds)
test_hamming = 1 - hamming_loss(y_test, test_preds)

print(f"\n{'='*60}")
print("RESULTS ON 100 SAMPLES")
print(f"{'='*60}")
print(f"\nTraining Set:")
print(f"  - Subset Accuracy (exact match): {train_acc:.4f}")
print(f"  - Hamming Accuracy (per-label):  {train_hamming:.4f}")

print(f"\nTest Set:")
print(f"  - Subset Accuracy (exact match): {test_acc:.4f}")
print(f"  - Hamming Accuracy (per-label):  {test_hamming:.4f}")

# Calculate per-class metrics
print(f"\nPer-sample statistics:")
print(f"  - Avg predictions per sample (test): {test_preds.sum(axis=1).mean():.2f}")
print(f"  - Avg true labels per sample (test): {y_test.sum(axis=1).mean():.2f}")

# Show some predictions
print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS (first 5 test samples):")
print(f"{'='*60}")
for i in range(min(5, len(test_preds))):
    pred_classes = [mlb.classes_[j] for j in range(n_classes) if test_preds[i, j] == 1]
    true_classes = [mlb.classes_[j] for j in range(n_classes) if y_test[i, j] == 1]
    print(f"\nSample {i+1} ({fname_test[i]}):")
    print(f"  True: {', '.join(true_classes) if true_classes else 'No labels'}")
    print(f"  Pred: {', '.join(pred_classes) if pred_classes else 'No predictions'}")
    if true_classes and pred_classes:
        match = set(pred_classes) & set(true_classes)
        print(f"  Match: {len(match)}/{len(true_classes)} correct")

print(f"\n{'='*60}")
print("MODEL ARCHITECTURE")
print(f"{'='*60}")
print(f"2-Layer Neural Network (MLPClassifier):")
print(f"  - Input:  {X_train.shape[1]} tabular features")
print(f"  - Hidden: 256 units with ReLU")
print(f"  - Output: {n_classes} classes")
print(f"  - Solver: Adam optimizer")
print(f"\nNote: This simplified version uses only tabular features")
print(f"      from the pre-extracted audio features CSV.")
print(f"      CNN feature extraction requires additional libraries")
print(f"      (librosa, PyTorch) that have compatibility issues.")
print(f"{'='*60}")

