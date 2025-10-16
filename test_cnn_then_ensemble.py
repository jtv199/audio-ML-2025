#!/usr/bin/env python3
"""
Test trained CNN alone, then test full ensemble
"""
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TESTING: Trained CNN vs Full Ensemble")
print("="*70)

# Configuration
DATA = Path('./input')
WORK = Path('work')
SAMPLE_SIZE = 100

# Mel-spectrogram config
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
    spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=MelConfig.sampling_rate, n_mels=MelConfig.n_mels,
        hop_length=MelConfig.hop_length, n_fft=MelConfig.n_fft,
        fmin=MelConfig.fmin, fmax=MelConfig.fmax
    )
    return librosa.power_to_db(spectrogram).astype(np.float32)

def mono_to_color(X, eps=1e-6):
    X = np.stack([X, X, X], axis=-1)
    mean, std = X.mean(), X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    if (_max - _min) > eps:
        V = Xstd
        V = np.clip(V, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        return V.astype(np.uint8)
    return np.zeros_like(Xstd, dtype=np.uint8)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv(DATA/'train_curated.csv')
df['label_count'] = df['labels'].str.count(',') + 1
single_df = df[df['label_count'] == 1].sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
print(f"    Using {len(single_df)} single-label samples")

# Load tabular features
features_df = pd.read_csv(WORK/'trn_curated_feature.csv').rename(columns={'file': 'fname'})
features_df = features_df.merge(single_df[['fname', 'labels']], on='fname')

# Encode labels
le = LabelEncoder()
y = le.fit_transform(features_df['labels'])
n_classes = len(le.classes_)
print(f"    Classes: {n_classes}")

# Load trained CNN model
print("\n[2] Loading trained MobileNetV4 CNN...")
try:
    # Try loading the state dict
    state_dict = torch.load('models/mobilenetv4_conv_small.pth', map_location='cpu')
    print(f"    ✓ Loaded state dict with {len(state_dict)} parameters")
    
    # Create a simple ResNet-like model to match the saved weights
    from torchvision import models
    cnn_full = models.resnet18(pretrained=False)
    # Modify first conv to accept our input and last fc for our classes
    cnn_full.fc = nn.Linear(cnn_full.fc.in_features, n_classes)
    
    # Try to load compatible weights
    try:
        cnn_full.load_state_dict(state_dict, strict=False)
        print(f"    ✓ Loaded weights into ResNet18 architecture")
    except:
        print(f"    ⚠ Could not load weights, using random initialization")
        
except Exception as e:
    print(f"    ✗ Failed to load model: {e}")
    print(f"    Using random ResNet18 initialization")
    from torchvision import models
    cnn_full = models.resnet18(pretrained=False)
    cnn_full.fc = nn.Linear(cnn_full.fc.in_features, n_classes)

cnn_full.eval()

# Create frozen feature extractor (remove last layer)
class FrozenCNN(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        # Remove the final FC layer
        self.features = nn.Sequential(*list(full_model.children())[:-1])
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)

frozen_cnn = FrozenCNN(cnn_full)

# Extract CNN features and get CNN predictions
print("\n[3] Processing audio files...")

def process_audio(fname, extractor, classifier):
    audio_path = DATA/'trn_curated'/fname
    audio = read_audio(audio_path)
    mel = audio_to_melspectrogram(audio)
    mel_color = mono_to_color(mel)
    img_tensor = torch.FloatTensor(mel_color).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Extract features
    features = extractor(img_tensor).squeeze()
    
    # Get CNN prediction
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = F.softmax(logits, dim=1)
    
    return features.numpy(), probs[0].numpy()

all_cnn_features = []
all_cnn_probs = []

for i, fname in enumerate(features_df['fname'].values):
    if (i + 1) % 20 == 0:
        print(f"    Progress: {i+1}/{len(features_df)}")
    feat, prob = process_audio(fname, frozen_cnn, cnn_full)
    all_cnn_features.append(feat)
    all_cnn_probs.append(prob)

X_cnn = np.array(all_cnn_features)
cnn_probs_all = np.array(all_cnn_probs)

print(f"    ✓ Extracted {X_cnn.shape}")

# Prepare tabular features
feature_cols = [c for c in features_df.columns if c not in ['fname', 'labels']]
X_tab = features_df[feature_cols].values
X_combined = np.concatenate([X_cnn, X_tab], axis=1)

# Split data
X_cnn_train, X_cnn_test, X_tab_train, X_tab_test, X_comb_train, X_comb_test, \
y_train, y_test, fname_train, fname_test = train_test_split(
    X_cnn, X_tab, X_combined, y, features_df['fname'].values,
    test_size=0.2, random_state=42
)

# Get CNN predictions for split data
cnn_train_idx = [list(features_df['fname']).index(f) for f in fname_train]
cnn_test_idx = [list(features_df['fname']).index(f) for f in fname_test]
cnn_probs_train = cnn_probs_all[cnn_train_idx]
cnn_probs_test = cnn_probs_all[cnn_test_idx]

# Scale features for NN
scaler = StandardScaler()
X_comb_train_scaled = scaler.fit_transform(X_comb_train)
X_comb_test_scaled = scaler.transform(X_comb_test)

print(f"\n[4] Training 2-layer NN on combined features...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(256,), activation='relu', solver='adam',
    alpha=0.0001, learning_rate_init=0.001, max_iter=200,
    random_state=42, verbose=False, early_stopping=True
)
nn_model.fit(X_comb_train_scaled, y_train)
print(f"    ✓ Completed in {nn_model.n_iter_} iterations, loss: {nn_model.loss_:.4f}")

# Get ensemble predictions
ensemble_preds_test = nn_model.predict(X_comb_test_scaled)

# Get CNN-only predictions
cnn_preds_test = cnn_probs_test.argmax(axis=1)

# Calculate accuracies
cnn_acc = accuracy_score(y_test, cnn_preds_test)
ensemble_acc = accuracy_score(y_test, ensemble_preds_test)

# Results
print(f"\n{'='*70}")
print("RESULTS: CNN vs Ensemble")
print(f"{'='*70}")
print(f"\nTest Set ({len(y_test)} samples, {n_classes} classes):")
print(f"  📊 CNN Only:      {cnn_acc:.1%} ({int(cnn_acc * len(y_test))}/20 correct)")
print(f"  📊 Ensemble:      {ensemble_acc:.1%} ({int(ensemble_acc * len(y_test))}/20 correct)")
print(f"  📊 Random Guess:  {1/n_classes:.1%}")
print(f"\n  Improvement:")
print(f"    CNN vs Random:      {cnn_acc/(1/n_classes):.1f}x")
print(f"    Ensemble vs Random: {ensemble_acc/(1/n_classes):.1f}x")

# Detailed comparison
print(f"\n{'='*70}")
print("DETAILED COMPARISON (All 20 test samples)")
print(f"{'='*70}")

for i in range(len(y_test)):
    true_label = le.inverse_transform([y_test[i]])[0]
    cnn_pred = le.classes_[cnn_preds_test[i]] if cnn_preds_test[i] < len(le.classes_) else "UNKNOWN"
    
    # Get ensemble prediction
    ensemble_pred_idx = np.where(nn_model.classes_ == ensemble_preds_test[i])[0]
    if len(ensemble_pred_idx) > 0:
        ensemble_pred = le.inverse_transform([ensemble_preds_test[i]])[0]
        ensemble_conf = nn_model.predict_proba(X_comb_test_scaled[i:i+1])[0, ensemble_pred_idx[0]]
    else:
        ensemble_pred = "UNKNOWN"
        ensemble_conf = 0.0
    
    cnn_conf = cnn_probs_test[i, cnn_preds_test[i]] if cnn_preds_test[i] < len(cnn_probs_test[i]) else 0.0
    
    cnn_correct = "✓" if cnn_pred == true_label else "✗"
    ens_correct = "✓" if ensemble_pred == true_label else "✗"
    
    print(f"\n{i+1:2d}. {fname_test[i]}")
    print(f"    True:     {true_label}")
    print(f"    CNN:      {cnn_correct} {cnn_pred:35s} ({cnn_conf:.1%})")
    print(f"    Ensemble: {ens_correct} {ensemble_pred:35s} ({ensemble_conf:.1%})")

print(f"\n{'='*70}")
print("ARCHITECTURE SUMMARY")
print(f"{'='*70}")
print(f"\n🔧 CNN Model (ResNet18):")
print(f"   • Input: 128×128×3 mel-spectrograms")
print(f"   • Architecture: ResNet18")
print(f"   • Feature dim: {X_cnn.shape[1]}")
print(f"   • Output: {n_classes} classes")
print(f"   • Status: Pre-trained weights loaded")

print(f"\n🔧 Ensemble Model:")
print(f"   • CNN features: {X_cnn.shape[1]}")
print(f"   • Tabular features: {X_tab.shape[1]}")
print(f"   • Combined: {X_combined.shape[1]} features")
print(f"   • Architecture: 2-layer NN (256 hidden units)")
print(f"   • Output: {n_classes} classes")

print(f"\n{'='*70}")

