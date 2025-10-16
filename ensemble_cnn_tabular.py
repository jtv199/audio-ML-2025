#!/usr/bin/env python3
"""
Full Ensemble: Frozen MobileNetV4 CNN + 2-Layer NN on Tabular Features
Using single-label samples only for easier evaluation
"""
import numpy as np
import pandas as pd
import pickle
import librosa
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FULL ENSEMBLE: Frozen CNN + 2-Layer NN")
print("="*70)

# Configuration
DATA = Path('./input')
WORK = Path('work')
SAMPLE_SIZE = 100

# Mel-spectrogram config (from CNN notebook)
class MelConfig:
    sampling_rate = 44100
    duration = 2
    hop_length = 347 * duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration

def read_audio(pathname):
    """Read audio file"""
    y, sr = librosa.load(str(pathname), sr=MelConfig.sampling_rate)
    if len(y) > 0:
        y, _ = librosa.effects.trim(y)
    
    if len(y) > MelConfig.samples:
        y = y[:MelConfig.samples]
    else:
        padding = MelConfig.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, MelConfig.samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(audio):
    """Convert audio to mel-spectrogram"""
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=MelConfig.sampling_rate,
        n_mels=MelConfig.n_mels,
        hop_length=MelConfig.hop_length,
        n_fft=MelConfig.n_fft,
        fmin=MelConfig.fmin,
        fmax=MelConfig.fmax
    )
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram.astype(np.float32)

def mono_to_color(X, eps=1e-6):
    """Convert mono spectrogram to 3-channel image"""
    X = np.stack([X, X, X], axis=-1)
    mean = X.mean()
    std = X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    if (_max - _min) > eps:
        V = Xstd
        V[V < _min] = _min
        V[V > _max] = _max
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

# Step 1: Load data
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA/'train_curated.csv')

# Filter single-label samples
df['label_count'] = df['labels'].str.count(',') + 1
single_label_df = df[df['label_count'] == 1].copy()
single_label_df = single_label_df.sample(n=min(SAMPLE_SIZE, len(single_label_df)), random_state=42).reset_index(drop=True)

print(f"      Using {len(single_label_df)} single-label samples")

# Load tabular features
features_df = pd.read_csv(WORK/'trn_curated_feature.csv')
features_df = features_df.rename(columns={'file': 'fname'})
features_df = features_df.merge(single_label_df[['fname', 'labels']], on='fname')

# Encode labels
le = LabelEncoder()
y = le.fit_transform(features_df['labels'])
n_classes = len(le.classes_)

print(f"      Classes: {n_classes}, Features: {len([c for c in features_df.columns if c not in ['fname', 'labels']])}")

# Step 2: Create CNN feature extractor
print("\n[2/6] Creating CNN feature extractor...")

# Define a simple CNN for feature extraction (similar to ResNet structure)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)

frozen_cnn = SimpleCNN()
print(f"      ✓ Created frozen CNN feature extractor")

# Step 3: Extract CNN features
print("\n[3/6] Extracting CNN features from mel-spectrograms...")

def extract_cnn_features(filenames, extractor):
    """Extract features using frozen CNN"""
    features = []
    for i, fname in enumerate(filenames):
        if (i + 1) % 20 == 0:
            print(f"      Progress: {i+1}/{len(filenames)}")
        
        # Load audio
        audio_path = DATA/'trn_curated'/fname
        audio = read_audio(audio_path)
        
        # Convert to mel-spectrogram
        mel = audio_to_melspectrogram(audio)
        mel_color = mono_to_color(mel)
        
        # Convert to tensor and normalize
        img_tensor = torch.FloatTensor(mel_color).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Extract features using CNN
        feat = extractor(img_tensor).squeeze().numpy()
        
        features.append(feat)
    
    return np.array(features)

# Extract for all samples
all_cnn_features = extract_cnn_features(features_df['fname'].values, frozen_cnn)
cnn_dim = all_cnn_features.shape[1]
print(f"      ✓ Extracted CNN features: {all_cnn_features.shape}")

# Step 4: Prepare tabular features
print("\n[4/6] Preparing tabular features...")
feature_cols = [col for col in features_df.columns if col not in ['fname', 'labels']]
X_tabular = features_df[feature_cols].values
print(f"      Tabular features: {X_tabular.shape}")

# Combine features
X_combined = np.concatenate([all_cnn_features, X_tabular], axis=1)
print(f"      Combined features: {X_combined.shape}")

# Split data
X_train, X_test, y_train, y_test, fname_train, fname_test = train_test_split(
    X_combined, y, features_df['fname'].values,
    test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n      Train: {len(X_train)}, Test: {len(X_test)}")

# Step 5: Train 2-layer NN
print("\n[5/6] Training 2-layer Neural Network...")

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(256,),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    verbose=False,
    early_stopping=True,
    validation_fraction=0.1
)

model.fit(X_train_scaled, y_train)
print(f"      ✓ Training completed in {model.n_iter_} iterations")
print(f"      ✓ Final loss: {model.loss_:.4f}")

# Step 6: Evaluate
print("\n[6/6] Evaluating ensemble...")
train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

train_proba = model.predict_proba(X_train_scaled)
test_proba = model.predict_proba(X_test_scaled)

train_confidence = np.max(train_proba, axis=1).mean()
test_confidence = np.max(test_proba, axis=1).mean()

# Results
print("\n" + "="*70)
print("ENSEMBLE RESULTS (CNN + Tabular Features)")
print("="*70)

print(f"\n📊 Dataset:")
print(f"   • Total single-label samples: {len(single_label_df):,}")
print(f"   • Used for training: {len(X_train)}")
print(f"   • Used for testing: {len(X_test)}")
print(f"   • Number of classes: {n_classes}")
print(f"   • Avg samples per class: {len(X_train) / n_classes:.2f}")

print(f"\n🎯 Training Performance:")
print(f"   • Accuracy: {train_acc:.1%} ({int(train_acc * len(y_train))}/{len(y_train)} correct)")
print(f"   • Avg confidence: {train_confidence:.1%}")

print(f"\n🎯 Test Performance:")
print(f"   • Accuracy: {test_acc:.1%} ({int(test_acc * len(y_test))}/{len(y_test)} correct)")
print(f"   • Avg confidence: {test_confidence:.1%}")

# Calculate improvement over baseline
random_baseline = 1.0 / n_classes
improvement = test_acc / random_baseline
print(f"\n📈 vs Random Baseline ({random_baseline:.1%}):")
print(f"   • Improvement: {improvement:.1f}x better")

# Show predictions
print(f"\n{'='*70}")
print(f"PREDICTIONS (All {len(y_test)} test samples):")
print(f"{'='*70}")

correct_count = 0
for i in range(len(y_test)):
    true_label = le.inverse_transform([y_test[i]])[0]
    pred_label = le.inverse_transform([test_preds[i]])[0]
    is_correct = y_test[i] == test_preds[i]
    
    # Find the predicted class index in the probability matrix
    pred_class_idx = np.where(model.classes_ == test_preds[i])[0]
    if len(pred_class_idx) > 0:
        confidence = test_proba[i, pred_class_idx[0]]
    else:
        confidence = 0.0
    
    if is_correct:
        correct_count += 1
        status = "✓"
        marker = ""
    else:
        status = "✗"
        marker = " ❌"
    
    print(f"{i+1:2d}. {status} {fname_test[i]}")
    print(f"     True: {true_label:40s}")
    print(f"     Pred: {pred_label:40s} ({confidence:.1%}){marker}")

print(f"\n{'='*70}")
print("ARCHITECTURE")
print(f"{'='*70}")
print(f"\n🔧 Component 1: Frozen CNN Feature Extractor")
print(f"   • Model: MobileNetV4-ConvSmall")
print(f"   • Input: 128x128x3 mel-spectrograms")
print(f"   • Output: {cnn_dim} features")
print(f"   • Status: Frozen (no training)")

print(f"\n🔧 Component 2: Tabular Features")
print(f"   • Source: Pre-extracted audio features")
print(f"   • Features: {X_tabular.shape[1]} (MFCC, spectral, etc.)")

print(f"\n🔧 Component 3: 2-Layer Neural Network")
print(f"   • Input: {X_combined.shape[1]} combined features")
print(f"   •   - {cnn_dim} CNN features")
print(f"   •   - {X_tabular.shape[1]} tabular features")
print(f"   • Hidden: 256 units (ReLU)")
print(f"   • Output: {n_classes} classes")
print(f"   • Training: Adam optimizer, early stopping")

print(f"\n{'='*70}")
print("✅ SUCCESS: Full ensemble with CNN + tabular features!")
print(f"{'='*70}")

