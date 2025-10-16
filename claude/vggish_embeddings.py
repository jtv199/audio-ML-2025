#!/usr/bin/env python3
"""
VGGish Embedding Extraction Script
Extracts VGGish embeddings from audio files
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import librosa
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import torch only when needed
import torch


def load_vggish_model():
    """Load VGGish model"""
    print("Loading VGGish model...")
    try:
        from torchvggish import vggish
        model = vggish()
        model.eval()
        print("  Model loaded successfully (CPU mode)")
        return model
    except Exception as e:
        print(f"  Error loading from torchvggish package: {e}")
        print("  Trying torch hub...")
        model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        model.eval()
        print("  Model loaded successfully from torch hub (CPU mode)")
        return model


def extract_vggish_embedding(model, audio_path):
    """Extract VGGish embedding from audio file"""
    # Load audio at 16kHz (VGGish requirement)
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

    # VGGish expects audio input, not raw waveform - it handles preprocessing internally
    # The model expects the audio path as input
    with torch.no_grad():
        embeddings = model.forward(str(audio_path))

    # Average over time dimension to get fixed-size embedding (128-dim)
    embedding = embeddings.mean(dim=0).cpu().numpy()

    return embedding


def process_dataset(df, audio_dir, output_dir, max_samples=None):
    """Process dataset and extract VGGish embeddings"""

    if max_samples:
        df = df.head(max_samples)
        print(f"Processing {len(df)} samples (subset)")
    else:
        print(f"Processing {len(df)} samples (full dataset)")

    # Load model
    model = load_vggish_model()

    results = []
    times = []

    print(f"\n{'='*80}")
    print(f"Extracting VGGish embeddings")
    print(f"{'='*80}\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="VGGish"):
        fname = row['fname']
        labels = row['labels']
        audio_path = audio_dir / fname

        if not audio_path.exists():
            print(f"Warning: File not found: {audio_path}")
            continue

        try:
            start_time = time.time()
            embedding = extract_vggish_embedding(model, audio_path)
            elapsed = time.time() - start_time
            times.append(elapsed)

            result = {
                'fname': fname,
                'labels': labels
            }

            # Add embedding dimensions (128-dim)
            for i, val in enumerate(embedding):
                result[f'emb_{i}'] = val

            results.append(result)

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / 'VGGish_embeddings.csv'
    results_df.to_csv(output_file, index=False)

    # Print statistics
    avg_time = np.mean(times) if times else 0
    total_time = np.sum(times) if times else 0
    embedding_dim = len(results[0]) - 2 if results else 0  # Subtract fname and labels columns

    print(f"\n{'='*80}")
    print(f"VGGish Embedding Extraction Complete")
    print(f"{'='*80}")
    print(f"  Samples processed: {len(results_df)}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Average time per sample: {avg_time:.4f}s")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Output file: {output_file}")
    print(f"  CSV shape: {results_df.shape}")
    print(f"{'='*80}\n")

    return results_df, avg_time, total_time


def main():
    """Main execution"""

    print("="*80)
    print("VGGISH AUDIO EMBEDDING EXTRACTION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setup paths
    base_dir = Path('.')
    csv_path = base_dir / 'input' / 'train_curated.csv'
    audio_dir = base_dir / 'input' / 'trn_curated'
    output_dir = base_dir / 'work' / 'embeddings'

    # Load dataset
    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Audio directory: {audio_dir}\n")

    # Test on subset first (10 samples)
    print("="*80)
    print("TESTING ON 10 SAMPLES")
    print("="*80)

    test_df, test_avg, test_total = process_dataset(
        df, audio_dir, output_dir, max_samples=10
    )

    # Estimate full dataset time
    estimated_full = test_avg * len(df)
    print(f"Estimated time for full dataset: {estimated_full/60:.1f} minutes")
    print(f"="*80)

    # Ask to continue
    print(f"\nReady to process full dataset ({len(df)} samples)")
    print("Running full dataset extraction...\n")

    # Process full dataset
    full_df, full_avg, full_total = process_dataset(
        df, audio_dir, output_dir
    )

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final output: work/embeddings/VGGish_embeddings.csv")
    print(f"Shape: {full_df.shape}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
