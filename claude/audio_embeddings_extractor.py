#!/usr/bin/env python3
"""
Audio Embedding Extraction Script
Extracts embeddings using CLAP2023, VGGish, and YAMNet
Tests on subset first, then runs fastest model on full dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import librosa
import torch
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AudioEmbeddingExtractor:
    """Base class for audio embedding extraction"""

    def __init__(self, model_name, device='cpu'):
        self.model_name = model_name
        self.device = 'cpu'  # Force CPU to avoid CUDA issues
        self.model = None

    def load_model(self):
        raise NotImplementedError

    def extract_embedding(self, audio_path):
        raise NotImplementedError

    def process_dataset(self, df, audio_dir, output_dir, max_samples=None):
        """Process dataset and extract embeddings"""
        if max_samples:
            df = df.head(max_samples)

        results = []
        times = []

        print(f"\n{'='*80}")
        print(f"Processing {len(df)} samples with {self.model_name}")
        print(f"{'='*80}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{self.model_name}"):
            fname = row['fname']
            labels = row['labels']
            audio_path = audio_dir / fname

            if not audio_path.exists():
                continue

            try:
                start_time = time.time()
                embedding = self.extract_embedding(audio_path)
                elapsed = time.time() - start_time
                times.append(elapsed)

                result = {
                    'fname': fname,
                    'labels': labels
                }

                # Add embedding dimensions
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
        output_file = output_dir / f'{self.model_name}_embeddings.csv'
        results_df.to_csv(output_file, index=False)

        # Print statistics
        avg_time = np.mean(times)
        total_time = np.sum(times)

        print(f"\n{self.model_name} Results:")
        print(f"  Samples processed: {len(results_df)}")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  Average time per sample: {avg_time:.4f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Saved to: {output_file}")

        return results_df, avg_time, total_time


class CLAPExtractor(AudioEmbeddingExtractor):
    """CLAP (Contrastive Language-Audio Pretraining) embeddings"""

    def __init__(self, device='cuda'):
        super().__init__('CLAP2023', device)

    def load_model(self):
        print(f"Loading {self.model_name} model...")
        try:
            from transformers import ClapModel, ClapProcessor

            model_id = "laion/clap-htsat-fused"
            self.processor = ClapProcessor.from_pretrained(model_id)
            self.model = ClapModel.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            print(f"  Model loaded successfully on {self.device}")

        except ImportError:
            print("  ERROR: transformers not installed. Install with: pip install transformers")
            raise

    def extract_embedding(self, audio_path):
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=48000, mono=True)

        # Process audio
        inputs = self.processor(audios=waveform, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract embedding
        with torch.no_grad():
            audio_embeds = self.model.get_audio_features(**inputs)

        return audio_embeds.cpu().numpy().flatten()


class VGGishExtractor(AudioEmbeddingExtractor):
    """VGGish audio embeddings"""

    def __init__(self, device='cpu'):
        super().__init__('VGGish', device)

    def load_model(self):
        print(f"Loading {self.model_name} model...")
        try:
            # Load VGGish from torch hub
            self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.model.to(self.device)
            self.model.eval()
            print(f"  Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"  ERROR loading VGGish: {e}")
            raise

    def extract_embedding(self, audio_path):
        # Load audio at 16kHz (VGGish requirement)
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Convert to tensor
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)

        # Extract embedding
        with torch.no_grad():
            embeddings = self.model.forward(waveform_tensor)

        # Average over time dimension to get fixed-size embedding
        embedding = embeddings.mean(dim=0).cpu().numpy()

        return embedding


class YAMNetExtractor(AudioEmbeddingExtractor):
    """YAMNet audio embeddings from TensorFlow Hub"""

    def __init__(self, device='cpu'):  # YAMNet uses TensorFlow, typically CPU
        super().__init__('YAMNet', device)

    def load_model(self):
        print(f"Loading {self.model_name} model...")
        try:
            # Load YAMNet from TensorFlow Hub
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            print(f"  Model loaded successfully")

        except Exception as e:
            print(f"  ERROR loading YAMNet: {e}")
            raise

    def extract_embedding(self, audio_path):
        # Load audio at 16kHz (YAMNet requirement)
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Ensure waveform is in correct format
        waveform = waveform.astype(np.float32)

        # Extract embeddings
        scores, embeddings, spectrogram = self.model(waveform)

        # Average embeddings over time to get fixed-size vector
        embedding = np.mean(embeddings.numpy(), axis=0)

        return embedding


def load_dataset(csv_path, audio_dir):
    """Load dataset CSV and verify audio directory"""
    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Audio directory: {audio_dir}")

    # Verify audio directory exists
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    return df


def compare_models_on_subset(df, audio_dir, output_dir, subset_size=10):
    """Test all models on a small subset to compare speed"""

    print("\n" + "="*80)
    print(f"TESTING MODELS ON {subset_size} SAMPLES")
    print("="*80)

    models = [
        # ('CLAP2023', CLAPExtractor),
        ('VGGish', VGGishExtractor),
        # ('YAMNet', YAMNetExtractor),
    ]

    results = {}

    for model_name, ExtractorClass in models:
        try:
            extractor = ExtractorClass()
            extractor.load_model()

            _, avg_time, total_time = extractor.process_dataset(
                df, audio_dir, output_dir, max_samples=subset_size
            )

            results[model_name] = {
                'avg_time': avg_time,
                'total_time': total_time
            }

        except Exception as e:
            print(f"\nERROR with {model_name}: {e}")
            results[model_name] = {
                'avg_time': float('inf'),
                'total_time': float('inf'),
                'error': str(e)
            }

    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)

    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('avg_time')

    print("\nRanking by speed (fastest first):")
    for idx, (model, row) in enumerate(comparison_df.iterrows(), 1):
        if 'error' in row:
            print(f"  {idx}. {model}: ERROR - {row['error']}")
        else:
            print(f"  {idx}. {model}: {row['avg_time']:.4f}s per sample ({row['total_time']:.2f}s total)")

    # Get fastest model
    fastest_model = comparison_df.index[0]
    print(f"\n✓ Fastest model: {fastest_model}")

    return fastest_model, results


def run_full_dataset(model_name, df, audio_dir, output_dir):
    """Run the fastest model on full dataset"""

    print("\n" + "="*80)
    print(f"RUNNING {model_name} ON FULL DATASET")
    print("="*80)

    # Select extractor
    extractors = {
        'CLAP2023': CLAPExtractor,
        'VGGish': VGGishExtractor,
        'YAMNet': YAMNetExtractor,
    }

    ExtractorClass = extractors.get(model_name)
    if not ExtractorClass:
        raise ValueError(f"Unknown model: {model_name}")

    extractor = ExtractorClass()
    extractor.load_model()

    results_df, avg_time, total_time = extractor.process_dataset(
        df, audio_dir, output_dir
    )

    print(f"\n✓ Processing complete!")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Output: {output_dir}/{model_name}_embeddings.csv")

    return results_df


def main():
    """Main execution"""

    print("="*80)
    print("AUDIO EMBEDDING EXTRACTION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setup paths
    base_dir = Path('.')
    csv_path = base_dir / 'input' / 'train_curated.csv'
    audio_dir = base_dir / 'input' / 'trn_curated'
    output_dir = base_dir / 'work' / 'embeddings'

    # Load dataset
    df = load_dataset(csv_path, audio_dir)

    # Test models on subset
    fastest_model, comparison_results = compare_models_on_subset(
        df, audio_dir, output_dir, subset_size=10
    )

    # Ask to continue with full dataset
    print(f"\n{'='*80}")
    print(f"Ready to run {fastest_model} on full dataset ({len(df)} samples)")

    # Estimate time
    avg_time = comparison_results[fastest_model]['avg_time']
    estimated_time = avg_time * len(df)
    print(f"Estimated time: {estimated_time/60:.1f} minutes")
    print(f"{'='*80}\n")

    # Run on full dataset
    results_df = run_full_dataset(fastest_model, df, audio_dir, output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
