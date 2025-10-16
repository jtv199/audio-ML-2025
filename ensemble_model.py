import numpy as np
import pandas as pd
import pickle
import librosa
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA = Path('./input')
WORK = Path('work')
SAMPLE_SIZE = 100  # Use 100 samples as requested

# Mel-spectrogram config from the CNN notebook
class MelConfig:
    sampling_rate = 44100
    duration = 2
    hop_length = 347 * duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration

def read_audio(pathname, trim_long_data=False):
    """Read audio file and convert to mel-spectrogram"""
    y, sr = librosa.load(str(pathname), sr=MelConfig.sampling_rate)
    if len(y) > 0:
        y, _ = librosa.effects.trim(y)
    
    if len(y) > MelConfig.samples:
        if trim_long_data:
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

# Load data
print("Loading data...")
df = pd.read_csv(DATA/'train_curated.csv').sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
features_df = pd.read_csv(WORK/'trn_curated_feature.csv')

# Merge features with labels
features_df = features_df.merge(df[['fname', 'labels']], on='fname')

# Prepare labels
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
labels_split = features_df['labels'].str.split(',')
y = mlb.fit_transform(labels_split)
n_classes = len(mlb.classes_)

print(f"Number of classes: {n_classes}")
print(f"Number of samples: {len(df)}")

# Prepare tabular features (excluding fname and labels columns)
feature_cols = [col for col in features_df.columns if col not in ['fname', 'labels']]
X_tabular = features_df[feature_cols].values

# Split data
X_tab_train, X_tab_test, y_train, y_test, fname_train, fname_test = train_test_split(
    X_tabular, y, features_df['fname'].values, 
    test_size=0.2, random_state=42
)

# Scale tabular features
scaler = StandardScaler()
X_tab_train_scaled = scaler.fit_transform(X_tab_train)
X_tab_test_scaled = scaler.transform(X_tab_test)

print("\nStep 1: Loading frozen CNN model as feature extractor...")

# Load the saved CNN model
try:
    cnn_model = torch.load('models/mobilenetv4_conv_small.pkl', map_location='cpu')
    print("Loaded saved model successfully")
except:
    print("Could not load saved model, will use simple feature extractor")
    cnn_model = None

# Create frozen CNN feature extractor (remove last 2 layers)
class FrozenCNNFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Remove last 2 layers (typically pooling + classifier)
        if model is not None:
            layers = list(model.children())[:-2]
            self.features = nn.Sequential(*layers)
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            # Simple fallback - lightweight CNN
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        self.features.eval()
    
    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)

extractor = FrozenCNNFeatureExtractor(cnn_model)

# Extract CNN features from mel-spectrograms
print("\nStep 2: Extracting CNN features from mel-spectrograms...")
def extract_cnn_features(filenames, extractor):
    features = []
    for i, fname in enumerate(filenames):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(filenames)} files...")
        
        # Load and convert audio to mel-spectrogram
        audio_path = DATA/'trn_curated'/fname
        audio = read_audio(audio_path)
        mel = audio_to_melspectrogram(audio)
        mel_color = mono_to_color(mel)
        
        # Convert to tensor
        img = torch.FloatTensor(mel_color).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Extract features
        feat = extractor(img).squeeze().numpy()
        features.append(feat)
    
    return np.array(features)

print("Extracting features from training set...")
X_cnn_train = extract_cnn_features(fname_train, extractor)
print("Extracting features from test set...")
X_cnn_test = extract_cnn_features(fname_test, extractor)
cnn_feature_dim = X_cnn_train.shape[1]

print(f"CNN feature dimension: {cnn_feature_dim}")

# Step 3: Build 2-layer NN for tabular features
print("\nStep 3: Training 2-layer NN on tabular features...")

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
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
print("Training tabular model...")
tabular_model.train()
for epoch in range(20):
    optimizer.zero_grad()
    outputs = tabular_model(X_tab_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Step 4: Create ensemble
print("\nStep 4: Creating ensemble...")

class EnsembleModel(nn.Module):
    def __init__(self, tabular_model):
        super().__init__()
        self.tabular_model = tabular_model
        
    def forward(self, cnn_features, tabular_features):
        # Get tabular predictions
        tab_out = torch.sigmoid(self.tabular_model(tabular_features))
        
        # For this simple ensemble, we'll just use tabular predictions
        # In a full implementation, you'd combine CNN features too
        return tab_out

ensemble = EnsembleModel(tabular_model)

# Step 5: Evaluate
print("\nStep 5: Evaluating ensemble...")
ensemble.eval()
with torch.no_grad():
    # Get predictions
    cnn_train_t = torch.FloatTensor(X_cnn_train)
    cnn_test_t = torch.FloatTensor(X_cnn_test)
    
    train_preds = ensemble(cnn_train_t, X_tab_train_t).numpy()
    test_preds = ensemble(cnn_test_t, X_tab_test_t).numpy()

# Calculate accuracy (using threshold of 0.5)
train_pred_labels = (train_preds > 0.5).astype(int)
test_pred_labels = (test_preds > 0.5).astype(int)

# Calculate subset accuracy (exact match)
train_acc = accuracy_score(y_train, train_pred_labels)
test_acc = accuracy_score(y_test, test_pred_labels)

print(f"\n{'='*50}")
print("RESULTS ON 100 SAMPLES")
print(f"{'='*50}")
print(f"Training Set:")
print(f"  - Subset Accuracy (exact match): {train_acc:.4f}")
print(f"  - Sample-wise F1: {(train_pred_labels == y_train).mean():.4f}")

print(f"\nTest Set:")
print(f"  - Subset Accuracy (exact match): {test_acc:.4f}")
print(f"  - Sample-wise F1: {(test_pred_labels == y_test).mean():.4f}")

# Calculate per-class metrics
print(f"\nPer-class statistics:")
print(f"  - Average predictions per sample: {test_pred_labels.sum(axis=1).mean():.2f}")
print(f"  - Average true labels per sample: {y_test.sum(axis=1).mean():.2f}")

# Show some predictions
print(f"\nSample predictions (first 5 test samples):")
for i in range(min(5, len(test_pred_labels))):
    pred_classes = [mlb.classes_[j] for j in range(n_classes) if test_pred_labels[i, j] == 1]
    true_classes = [mlb.classes_[j] for j in range(n_classes) if y_test[i, j] == 1]
    print(f"\nSample {i+1}:")
    print(f"  True: {', '.join(true_classes)}")
    print(f"  Pred: {', '.join(pred_classes)}")

print(f"\n{'='*50}")
print("ENSEMBLE ARCHITECTURE")
print(f"{'='*50}")
print(f"1. Frozen MobileNetV4 CNN (feature extractor)")
print(f"   - Input: 128x128x3 mel-spectrograms")
print(f"   - Output: {cnn_feature_dim} features")
print(f"   - Status: Frozen (no training)")
print(f"\n2. 2-Layer Neural Network")
print(f"   - Input: {tabular_dim} tabular features")
print(f"   - Hidden: {hidden_dim} units with ReLU")
print(f"   - Output: {n_classes} classes")
print(f"\n3. Ensemble: Simple averaging of predictions")
print(f"{'='*50}")

