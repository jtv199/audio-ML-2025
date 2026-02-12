# EnCodec Audio Tokenization

## Overview

This script tokenizes audio files using Facebook's EnCodec neural audio codec. It converts raw audio waveforms into discrete tokens that can be used for machine learning models.

## How EnCodec Handles Variable-Length Audio

### Facebook's Approach
- **24 kHz model**: Processes entire file at once (can cause OOM for long files)
- **48 kHz model**: Automatically chunks into 1-second segments with 1% overlap

### Our Implementation Strategy

Since EnCodec doesn't intelligently handle variable lengths by default, we implement:

1. **Short audio (< 2 seconds)**: Zero-pad to reach target duration
2. **Long audio (> 2 seconds)**: Split into 2-second chunks with 10% overlap, then average tokens
3. **Exact length**: Process directly

This ensures consistent token representation across variable-length inputs.

## Usage

### Basic Usage
```bash
# Tokenize curated dataset (test with 100 samples)
python claude/encodec_tokenizer.py --dataset curated --max_samples 100

# Tokenize full curated dataset
python claude/encodec_tokenizer.py --dataset curated

# Tokenize noisy dataset
python claude/encodec_tokenizer.py --dataset noisy

# Tokenize both datasets
python claude/encodec_tokenizer.py --dataset both
```

### Advanced Options
```bash
# Enable debug output
python claude/encodec_tokenizer.py --dataset curated --debug --max_samples 10

# Use 48 kHz model
python claude/encodec_tokenizer.py --dataset curated --model encodec_48khz

# Custom target duration
python claude/encodec_tokenizer.py --dataset curated --target_duration 3.0
```

## Output

### CSV File Structure
The script creates `work/tokenized/tokenized_train_curated.csv` with columns:
- `fname`: Original audio filename
- `labels`: Audio labels
- `num_tokens`: Total number of tokens
- `token_shape`: Shape of token array before flattening
- `num_chunks`: Number of chunks processed
- `duration_sec`: Original audio duration
- `token_0`, `token_1`, ...: Individual token values

### Summary File
A text summary is also created with statistics:
- Total samples processed
- Average tokens per sample
- Average duration
- Token shape distribution

## Technical Details

### EnCodec Model
- **Sample rate**: 24 kHz (configurable to 48 kHz)
- **Target bandwidth**: 6.0 kbps
- **Architecture**: Neural audio codec with VQ-VAE
- **Output**: Discrete tokens representing compressed audio

### Preprocessing Pipeline
1. Load audio with torchaudio
2. Convert to model's sample rate (24 kHz) and channel count
3. Handle variable length (pad/chunk)
4. Encode to tokens
5. Average tokens across chunks if multiple
6. Flatten to 1D array for CSV storage

### Why This Matters for Freesound Dataset
- Freesound audio varies in length (typically ~2 seconds)
- EnCodec provides a learned representation vs. hand-crafted features
- Tokens can be used directly with transformer models
- Reduces audio to discrete vocabulary (useful for sequence models)

## Key Features

- **GPU acceleration**: Automatically uses CUDA if available
- **Progress tracking**: tqdm progress bar for batch processing
- **Error handling**: Continues processing if individual files fail
- **Batch processing**: Handles entire datasets efficiently
- **Flexible output**: CSV format compatible with pandas/scikit-learn

## Dependencies

- torch >= 2.0
- torchaudio
- encodec
- pandas
- numpy
- tqdm

Install with:
```bash
pip install encodec torchaudio pandas numpy tqdm
```
