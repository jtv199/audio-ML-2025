#!/usr/bin/env python3
"""
VGGish Embedding Extraction Script - Final Version
Extracts VGGish embeddings from audio files and saves to work/tokenized
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import torch and torchvggish
import torch
from torchvggish import vggish, waveform_to_examples
import librosa


def load_vggish_model():
    """Load VGGish model"""
    print("Loading VGGish model...")
    model = vggish()
    model.eval()
    print("  Model loaded successfully (CPU mode)")
    return model


def extract_vggish_embedding(model, audio_path):
    """Extract VGGish embedding from audio file"""
    # Load audio at 16kHz (VGGish requirement)
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Convert waveform to VGGish input examples (spectrograms)
    examples = waveform_to_examples(waveform, sr)

    # Skip if no examples (audio too short)
    if examples.shape[0] == 0:
        return None

    # Extract embeddings
    with torch.no_grad():
        embeddings = model.forward(examples)

    # Average over time dimension to get fixed-size embedding (128-dim)
    embedding = embeddings.mean(dim=0).cpu().numpy()

    return embedding


def process_dataset(df, audio_dir, output_file, dataset_name):
    """Process dataset and extract VGGish embeddings"""

    print(f"\n{'='*80}")
    print(f"Processing {dataset_name} dataset: {len(df)} samples")
    print(f"{'='*80}\n")

    # Load model
    model = load_vggish_model()

    results = []
    times = []
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="VGGish"):
        fname = row['fname']
        labels = row.get('labels', '')  # Get labels if available (train set)
        audio_path = audio_dir / fname

        if not audio_path.exists():
            errors.append(f"File not found: {fname}")
            continue

        try:
            start_time = time.time()
            embedding = extract_vggish_embedding(model, audio_path)
            elapsed = time.time() - start_time
            times.append(elapsed)

            # Skip if embedding couldn't be extracted
            if embedding is None:
                errors.append(f"Audio too short: {fname}")
                continue

            result = {'fname': fname}

            # Add labels if available
            if labels:
                result['labels'] = labels

            # Add embedding dimensions (128-dim)
            for i, val in enumerate(embedding):
                result[f'emb_{i}'] = val

            results.append(result)

        except Exception as e:
            errors.append(f"Error processing {fname}: {str(e)}")
            continue

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_file.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(output_file, index=False)

    # Print statistics
    avg_time = np.mean(times) if times else 0
    total_time = np.sum(times) if times else 0
    embedding_dim = len([c for c in results_df.columns if c.startswith('emb_')]) if len(results_df) > 0 else 0

    print(f"\n{'='*80}")
    print(f"VGGish Embedding Extraction Complete - {dataset_name}")
    print(f"{'='*80}")
    print(f"  Samples processed: {len(results_df)}/{len(df)}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Average time per sample: {avg_time:.4f}s")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Errors: {len(errors)}")
    print(f"  Output file: {output_file}")
    print(f"  CSV shape: {results_df.shape}")
    print(f"{'='*80}\n")

    # Print first few errors if any
    if errors:
        print(f"First 10 errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        print()

    return results_df


def main():
    """Main execution"""

    print("="*80)
    print("VGGISH AUDIO EMBEDDING EXTRACTION - FINAL")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setup paths
    base_dir = Path('.')
    output_dir = base_dir / 'work' / 'tokenized'

    # Process train_curated dataset
    print("\n" + "="*80)
    print("PROCESSING TRAIN_CURATED DATASET")
    print("="*80)

    csv_path = base_dir / 'input' / 'train_curated.csv'
    audio_dir = base_dir / 'input' / 'trn_curated'
    output_file = output_dir / 'vggish_embeddings_train_curated.csv'

    print(f"Loading dataset from {csv_path}")
    df_train = pd.read_csv(csv_path)
    print(f"  Dataset shape: {df_train.shape}")
    print(f"  Audio directory: {audio_dir}")

    # Process full dataset
    train_df = process_dataset(df_train, audio_dir, output_file, 'train_curated')

    # Process test dataset
    print("\n" + "="*80)
    print("PROCESSING TEST DATASET")
    print("="*80)

    test_audio_dir = base_dir / 'input' / 'test'
    test_output_file = output_dir / 'vggish_embeddings_test.csv'

    # Get test filenames
    test_files = sorted(list(test_audio_dir.glob('*.wav')))
    print(f"Found {len(test_files)} test audio files")

    if test_files:
        # Create DataFrame with filenames
        df_test = pd.DataFrame({'fname': [f.name for f in test_files]})
        print(f"  Dataset shape: {df_test.shape}")
        print(f"  Audio directory: {test_audio_dir}")

        # Process test dataset
        test_df = process_dataset(df_test, test_audio_dir, test_output_file, 'test')

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: work/tokenized/")
    print(f"Files generated:")
    print(f"  - vggish_embeddings_train_curated.csv (shape: {train_df.shape})")
    if test_files:
        print(f"  - vggish_embeddings_test.csv (shape: {test_df.shape})")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
