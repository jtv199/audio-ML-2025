#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ENSEMBLE MODEL: 2-Layer NN on Single-Label Samples")
print("="*60)

# Configuration
DATA = Path('./input')
WORK = Path('work')
SAMPLE_SIZE = 100  # Use 100 samples as requested

# Load data
print("\n1. Loading data...")
df = pd.read_csv(DATA/'train_curated.csv')

# Filter to only single-label samples
df['label_count'] = df['labels'].str.count(',') + 1
single_label_df = df[df['label_count'] == 1].copy()
print(f"   - Total samples in dataset: {len(df)}")
print(f"   - Single-label samples: {len(single_label_df)}")

# Sample 100 from single-label data
single_label_df = single_label_df.sample(n=min(SAMPLE_SIZE, len(single_label_df)), random_state=42).reset_index(drop=True)
print(f"   - Using {len(single_label_df)} single-label samples for training")

# Load features
features_df = pd.read_csv(WORK/'trn_curated_feature.csv')
features_df = features_df.rename(columns={'file': 'fname'})

# Merge features with labels
features_df = features_df.merge(single_label_df[['fname', 'labels']], on='fname')

# Prepare labels (single-label encoding)
le = LabelEncoder()
y = le.fit_transform(features_df['labels'])
n_classes = len(le.classes_)

print(f"   - Number of unique classes: {n_classes}")

# Prepare tabular features
feature_cols = [col for col in features_df.columns if col not in ['fname', 'labels']]
X = features_df[feature_cols].values
print(f"   - Tabular feature dimension: {X.shape[1]}")

# Split data
X_train, X_test, y_train, y_test, fname_train, fname_test = train_test_split(
    X, y, features_df['fname'].values, 
    test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n2. Data split:")
print(f"   - Training samples: {len(X_train)}")
print(f"   - Test samples: {len(X_test)}")
print(f"   - Samples per class (avg): {len(X_train) / n_classes:.2f}")

# Train 2-layer NN
print("\n3. Training 2-layer Neural Network...")
print("   (Using scikit-learn's MLPClassifier)")

model = MLPClassifier(
    hidden_layer_sizes=(256,),  # 2-layer: 1 hidden + 1 output
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    verbose=False
)

model.fit(X_train_scaled, y_train)
print(f"   - Training completed in {model.n_iter_} iterations")
print(f"   - Final loss: {model.loss_:.4f}")

# Evaluate
print("\n4. Evaluating model...")
train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)

# Calculate metrics
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

# Get probabilities for confidence analysis
train_proba = model.predict_proba(X_train_scaled)
test_proba = model.predict_proba(X_test_scaled)

train_confidence = np.max(train_proba, axis=1).mean()
test_confidence = np.max(test_proba, axis=1).mean()

print(f"\n{'='*60}")
print("RESULTS ON SINGLE-LABEL SAMPLES")
print(f"{'='*60}")
print(f"\nTraining Set:")
print(f"  - Accuracy: {train_acc:.4f} ({int(train_acc * len(y_train))}/{len(y_train)} correct)")
print(f"  - Average confidence: {train_confidence:.4f}")

print(f"\nTest Set:")
print(f"  - Accuracy: {test_acc:.4f} ({int(test_acc * len(y_test))}/{len(y_test)} correct)")
print(f"  - Average confidence: {test_confidence:.4f}")

# Show confusion matrix summary
print(f"\n{'='*60}")
print("CONFUSION MATRIX SUMMARY")
print(f"{'='*60}")
cm = confusion_matrix(y_test, test_preds)
print(f"  - Diagonal (correct): {np.diag(cm).sum()}")
print(f"  - Off-diagonal (errors): {cm.sum() - np.diag(cm).sum()}")

# Show detailed predictions for all test samples
print(f"\n{'='*60}")
print(f"DETAILED PREDICTIONS (All {len(y_test)} test samples):")
print(f"{'='*60}")
correct = 0
for i in range(len(y_test)):
    true_label = le.inverse_transform([y_test[i]])[0]
    pred_label = le.inverse_transform([test_preds[i]])[0]
    is_correct = y_test[i] == test_preds[i]
    confidence = test_proba[i, test_preds[i]]
    
    if is_correct:
        correct += 1
        status = "✓"
    else:
        status = "✗"
    
    print(f"{i+1:2d}. {status} {fname_test[i]}")
    print(f"    True: {true_label}")
    print(f"    Pred: {pred_label} (confidence: {confidence:.4f})")
    if not is_correct:
        print(f"    >>> INCORRECT")

print(f"\n{'='*60}")
print("MODEL ARCHITECTURE")
print(f"{'='*60}")
print(f"2-Layer Neural Network (MLPClassifier):")
print(f"  - Input:  {X.shape[1]} tabular features")
print(f"  - Hidden: 256 units with ReLU")
print(f"  - Output: {n_classes} classes (single-label)")
print(f"  - Solver: Adam optimizer")
print(f"  - Max iterations: 200")
print(f"\nFeature Source:")
print(f"  - Tabular features from pre-computed audio analysis")
print(f"  - Features include: MFCC, spectral features, etc.")
print(f"\nSimplification:")
print(f"  - Used only single-label samples for easier classification")
print(f"  - This reduces the problem complexity significantly")
print(f"  - {len(single_label_df)}/{len(df)} samples were single-label")
print(f"{'='*60}")

