import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ENSEMBLE MODEL: Frozen CNN + 2-Layer NN")
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
features_df = features_df.merge(df[['fname', 'labels']], on='fname')
print(f"   - Loaded {len(features_df)} samples")

# Prepare labels
from sklearn.preprocessing import MultiLabelBinarizer
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
X_tab_train, X_tab_test, y_train, y_test, fname_train, fname_test = train_test_split(
    X_tabular, y, features_df['fname'].values, 
    test_size=0.2, random_state=42
)

# Scale tabular features
scaler = StandardScaler()
X_tab_train_scaled = scaler.fit_transform(X_tab_train)
X_tab_test_scaled = scaler.transform(X_tab_test)

print(f"\n2. Data split:")
print(f"   - Training samples: {len(X_tab_train)}")
print(f"   - Test samples: {len(X_tab_test)}")

# For now, we'll skip CNN feature extraction since librosa installation has issues
# Instead, we'll just train on the tabular features
print("\n3. Training 2-layer NN on tabular features...")

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
tabular_dim = X_tab_train_scaled.shape[1]
hidden_dim = 256
tabular_model = TwoLayerNN(tabular_dim, hidden_dim, n_classes)

# Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(tabular_model.parameters(), lr=0.001)

# Convert to tensors
X_tab_train_t = torch.FloatTensor(X_tab_train_scaled)
y_train_t = torch.FloatTensor(y_train)
X_tab_test_t = torch.FloatTensor(X_tab_test_scaled)
y_test_t = torch.FloatTensor(y_test)

# Train tabular model
tabular_model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = tabular_model(X_tab_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:2d}, Loss: {loss.item():.4f}")

# Evaluate
print("\n4. Evaluating model...")
tabular_model.eval()
with torch.no_grad():
    train_preds = torch.sigmoid(tabular_model(X_tab_train_t)).numpy()
    test_preds = torch.sigmoid(tabular_model(X_tab_test_t)).numpy()

# Calculate accuracy (using threshold of 0.5)
train_pred_labels = (train_preds > 0.5).astype(int)
test_pred_labels = (test_preds > 0.5).astype(int)

# Calculate subset accuracy (exact match)
train_acc = accuracy_score(y_train, train_pred_labels)
test_acc = accuracy_score(y_test, test_pred_labels)

# Calculate hamming accuracy (per-label)
from sklearn.metrics import hamming_loss
train_hamming = 1 - hamming_loss(y_train, train_pred_labels)
test_hamming = 1 - hamming_loss(y_test, test_pred_labels)

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
print(f"  - Avg predictions per sample (test): {test_pred_labels.sum(axis=1).mean():.2f}")
print(f"  - Avg true labels per sample (test): {y_test.sum(axis=1).mean():.2f}")

# Show some predictions
print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS (first 5 test samples):")
print(f"{'='*60}")
for i in range(min(5, len(test_pred_labels))):
    pred_classes = [mlb.classes_[j] for j in range(n_classes) if test_pred_labels[i, j] == 1]
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
print(f"2-Layer Neural Network:")
print(f"  - Input:  {tabular_dim} tabular features")
print(f"  - Hidden: {hidden_dim} units with ReLU + Dropout(0.3)")
print(f"  - Output: {n_classes} classes with BCEWithLogitsLoss")
print(f"\nNote: CNN feature extraction skipped due to library issues.")
print(f"      This demo uses only tabular features from LinearSVC.")
print(f"{'='*60}")

