"""
EnCodec Audio Tokenization Script for Kaggle Free Sound Audio Tagging 2019

This script tokenizes audio files using Facebook's EnCodec model and handles
variable-length audio inputs by:
1. Padding short audio to minimum length
2. Processing in chunks if audio exceeds maximum length
3. Averaging tokens across chunks for consistency

Usage:
    python encodec_tokenizer.py --dataset curated --max_samples 100
    python encodec_tokenizer.py --dataset noisy --debug
"""

import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Tuple, Optional
import warnings

# Try to import encodec
try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio
except ImportError:
    print("EnCodec not installed. Install with: pip install encodec")
    raise


class AudioTokenizer:
    """
    Tokenizes audio files using EnCodec with handling for variable-length inputs.

    EnCodec processes audio in fixed-length chunks. For variable-length audio:
    - Short audio (<target_duration): Pad with zeros
    - Long audio (>target_duration): Process in overlapping chunks and average
    - Exact length: Process directly
    """

    def __init__(
        self,
        model_name: str = "encodec_24khz",
        target_duration: float = 2.0,
        chunk_overlap: float = 0.1,
        device: str = None
    ):
        """
        Initialize the tokenizer.

        Args:
            model_name: EnCodec model to use ('encodec_24khz' or 'encodec_48khz')
            target_duration: Target duration in seconds for audio chunks
            chunk_overlap: Overlap ratio for processing long audio (0.0-0.5)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load EnCodec model
        print(f"Loading {model_name} model...")
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)  # Target bandwidth in kbps
        self.model.to(self.device)
        self.model.eval()

        self.target_duration = target_duration
        self.chunk_overlap = chunk_overlap
        self.sample_rate = self.model.sample_rate
        self.target_samples = int(self.target_duration * self.sample_rate)

        print(f"Model sample rate: {self.sample_rate} Hz")
        print(f"Target duration: {self.target_duration}s ({self.target_samples} samples)")

    def load_audio(self, file_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and convert to model's sample rate.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_tensor, original_sample_rate)
        """
        wav, sr = torchaudio.load(file_path)
        return wav, sr

    def preprocess_audio(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Preprocess audio to match model requirements.

        Handles variable-length audio:
        - Converts to model's sample rate
        - Converts to mono if needed
        - Pads or chunks to target duration

        Args:
            wav: Audio waveform tensor [channels, samples]
            sr: Original sample rate

        Returns:
            Preprocessed audio tensor [1, channels, samples]
        """
        # Convert to model's sample rate and channel count
        wav = convert_audio(wav, sr, self.sample_rate, self.model.channels)

        # Add batch dimension
        wav = wav.unsqueeze(0)  # [1, channels, samples]

        return wav

    def handle_variable_length(self, wav: torch.Tensor) -> List[torch.Tensor]:
        """
        Handle variable-length audio by padding or chunking.

        Strategy:
        - If shorter than target: pad with zeros
        - If longer than target: split into overlapping chunks
        - If exact length: return as-is

        Args:
            wav: Audio tensor [1, channels, samples]

        Returns:
            List of audio chunks, each [1, channels, target_samples]
        """
        batch, channels, samples = wav.shape

        if samples < self.target_samples:
            # Pad short audio
            pad_amount = self.target_samples - samples
            wav_padded = torch.nn.functional.pad(wav, (0, pad_amount))
            return [wav_padded]

        elif samples > self.target_samples:
            # Chunk long audio with overlap
            chunks = []
            hop_size = int(self.target_samples * (1 - self.chunk_overlap))

            for start in range(0, samples - self.target_samples + 1, hop_size):
                end = start + self.target_samples
                chunk = wav[:, :, start:end]
                chunks.append(chunk)

            # Handle remainder if any
            if len(chunks) == 0 or chunks[-1].shape[-1] < self.target_samples:
                last_chunk = wav[:, :, -self.target_samples:]
                if len(chunks) == 0 or not torch.equal(chunks[-1], last_chunk):
                    chunks.append(last_chunk)

            return chunks

        else:
            # Exact length
            return [wav]

    @torch.no_grad()
    def tokenize_audio(self, file_path: Path) -> Tuple[np.ndarray, dict]:
        """
        Tokenize a single audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (token_array, metadata_dict)
        """
        try:
            # Load and preprocess
            wav, sr = self.load_audio(file_path)
            wav = self.preprocess_audio(wav, sr)

            # Handle variable length
            chunks = self.handle_variable_length(wav)

            # Encode each chunk
            all_codes = []
            for chunk in chunks:
                chunk = chunk.to(self.device)
                encoded_frames = self.model.encode(chunk)

                # Extract codes: encoded_frames is a list of (codes, scale) tuples
                # codes shape: [batch, num_codebooks, time_steps]
                codes = encoded_frames[0][0]  # Get codes from first frame
                codes = codes.cpu().numpy()
                all_codes.append(codes)

            # Average codes across chunks if multiple
            if len(all_codes) > 1:
                # Stack and average
                stacked = np.stack(all_codes, axis=0)
                averaged_codes = np.mean(stacked, axis=0).astype(np.int32)
            else:
                averaged_codes = all_codes[0]

            # Flatten to 1D for CSV storage: [batch, codebooks, time] -> [codebooks * time]
            flattened = averaged_codes.squeeze(0).flatten()  # Remove batch dim and flatten

            metadata = {
                'original_sr': sr,
                'duration_sec': wav.shape[-1] / self.sample_rate,
                'num_chunks': len(chunks),
                'token_shape': str(averaged_codes.shape),
                'num_tokens': len(flattened)
            }

            return flattened, metadata

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            raise

    def tokenize_dataset(
        self,
        csv_path: Path,
        audio_dir: Path,
        output_dir: Path,
        max_samples: Optional[int] = None,
        debug: bool = False
    ) -> pd.DataFrame:
        """
        Tokenize an entire dataset and save results.

        Args:
            csv_path: Path to CSV with audio filenames and labels
            audio_dir: Directory containing audio files
            output_dir: Directory to save tokenized data
            max_samples: Maximum number of samples to process (for debugging)
            debug: If True, print detailed progress information

        Returns:
            DataFrame with tokenized data
        """
        # Load dataset
        df = pd.read_csv(csv_path)

        if max_samples:
            df = df.head(max_samples)
            print(f"Processing {len(df)} samples (limited for debugging)")
        else:
            print(f"Processing {len(df)} samples")

        # Create output directory
        output_dir.mkdir(exist_ok=True, parents=True)

        # Process each audio file
        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
            fname = row['fname']
            labels = row['labels']
            audio_path = audio_dir / fname

            if not audio_path.exists():
                print(f"Warning: File not found: {audio_path}")
                continue

            try:
                tokens, metadata = self.tokenize_audio(audio_path)

                # Store result
                result = {
                    'fname': fname,
                    'labels': labels,
                    'num_tokens': metadata['num_tokens'],
                    'token_shape': metadata['token_shape'],
                    'num_chunks': metadata['num_chunks'],
                    'duration_sec': metadata['duration_sec']
                }

                # Add token values as separate columns
                for i, token_val in enumerate(tokens):
                    result[f'token_{i}'] = token_val

                results.append(result)

                if debug and idx < 5:
                    print(f"\nFile: {fname}")
                    print(f"  Labels: {labels}")
                    print(f"  Tokens: {tokens[:10]}... ({len(tokens)} total)")
                    print(f"  Metadata: {metadata}")

            except Exception as e:
                print(f"Failed to process {fname}: {e}")
                continue

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Save to CSV
        output_file = output_dir / f"tokenized_{csv_path.stem}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved tokenized data to: {output_file}")
        print(f"Shape: {results_df.shape}")

        # Save summary statistics
        summary_file = output_dir / f"summary_{csv_path.stem}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Tokenization Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Total samples: {len(results_df)}\n")
            f.write(f"Average tokens per sample: {results_df['num_tokens'].mean():.2f}\n")
            f.write(f"Average duration: {results_df['duration_sec'].mean():.2f}s\n")
            f.write(f"Average chunks: {results_df['num_chunks'].mean():.2f}\n")
            f.write(f"\nToken shape distribution:\n")
            f.write(str(results_df['token_shape'].value_counts()))

        print(f"Saved summary to: {summary_file}")

        return results_df


def main():
    parser = argparse.ArgumentParser(description='Tokenize audio files using EnCodec')
    parser.add_argument('--dataset', type=str, default='curated',
                       choices=['curated', 'noisy', 'both'],
                       help='Which dataset to process')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for debugging)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--model', type=str, default='encodec_24khz',
                       choices=['encodec_24khz', 'encodec_48khz'],
                       help='EnCodec model to use')
    parser.add_argument('--target_duration', type=float, default=2.0,
                       help='Target duration for audio chunks in seconds')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path('.')
    input_dir = base_dir / 'input'
    output_dir = base_dir / 'work' / 'tokenized'

    # Initialize tokenizer
    tokenizer = AudioTokenizer(
        model_name=args.model,
        target_duration=args.target_duration
    )

    # Process datasets
    datasets_to_process = []

    if args.dataset in ['curated', 'both']:
        datasets_to_process.append({
            'csv': input_dir / 'train_curated.csv',
            'audio': input_dir / 'trn_curated',
            'name': 'curated'
        })

    if args.dataset in ['noisy', 'both']:
        datasets_to_process.append({
            'csv': input_dir / 'train_noisy.csv',
            'audio': input_dir / 'train_noisy',
            'name': 'noisy'
        })

    # Process each dataset
    for dataset in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {dataset['name']} dataset")
        print(f"{'='*60}")

        tokenizer.tokenize_dataset(
            csv_path=dataset['csv'],
            audio_dir=dataset['audio'],
            output_dir=output_dir,
            max_samples=args.max_samples,
            debug=args.debug
        )

    print("\n✓ Tokenization complete!")


if __name__ == '__main__':
    main()
