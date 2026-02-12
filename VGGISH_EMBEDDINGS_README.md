# VGGish Audio Embeddings Generation

## Overview
This project uses torchvggish to generate audio embeddings for the Free Sound Audio Tagging 2019 Kaggle competition.

## Files Created

### Main Script
- **generate_vggish_final.py**: Final production script that processes all audio files and generates VGGish embeddings

### Output Files (in work/tokenized/)
- **vggish_embeddings_train_curated.csv**: VGGish embeddings for training data (4970 samples)
  - Columns: `fname`, `labels`, `emb_0` through `emb_127` (128-dimensional embeddings)

- **vggish_embeddings_test.csv**: VGGish embeddings for test data
  - Columns: `fname`, `emb_0` through `emb_127`

### Monitoring
- **check_vggish_progress.sh**: Shell script to check the progress of embedding generation
- **vggish_final_output.log**: Log file with detailed progress

## Current Status

The embedding generation script is currently RUNNING in the background (PID: 6185).

To check progress:
```bash
bash check_vggish_progress.sh
```

Or view the log directly:
```bash
tail -f vggish_final_output.log
```

## VGGish Model Details

- **Model**: VGGish (pre-trained on AudioSet)
- **Input**: Audio waveforms (16 kHz, mono)
- **Processing**: Converts waveforms to log-mel spectrograms
- **Output**: 128-dimensional embeddings per audio segment
- **Aggregation**: Embeddings are averaged across time to create a single fixed-size vector per audio file

## Implementation Details

### Preprocessing
1. Load audio at 16 kHz using librosa
2. Convert waveform to VGGish input examples (mel-spectrograms)
3. Process through VGGish model
4. Average embeddings across time dimension

### Error Handling
- Files not found: Skipped with warning
- Audio too short: Skipped (cannot generate spectrograms)
- Other errors: Logged and continue processing

### Performance
- Processing speed: ~3-7 samples/second on CPU
- Estimated time for full training set: ~20-30 minutes
- Test set will be processed after training set completes

## Usage

### Running the script manually:
```bash
~/miniconda3/envs/freesound/bin/python generate_vggish_final.py
```

### Running in background:
```bash
nohup ~/miniconda3/envs/freesound/bin/python generate_vggish_final.py > vggish_final_output.log 2>&1 &
```

## Loading the Embeddings

```python
import pandas as pd

# Load embeddings
train_embeddings = pd.read_csv('work/tokenized/vggish_embeddings_train_curated.csv')
test_embeddings = pd.read_csv('work/tokenized/vggish_embeddings_test.csv')

# Extract embedding features
embedding_cols = [f'emb_{i}' for i in range(128)]
X_train = train_embeddings[embedding_cols].values
X_test = test_embeddings[embedding_cols].values

# Get labels (for training)
y_train = train_embeddings['labels'].values
```

## Next Steps

Once the embeddings are generated, you can:

1. Use them directly for classification (e.g., with sklearn models)
2. Combine with other features (e.g., mel-spectrograms, MFCCs)
3. Fine-tune a classifier on top of the embeddings
4. Compare with other audio embedding models (PANNs, wav2vec2, etc.)

## Notes

- VGGish embeddings are extracted from the torchvggish library
- The model is pre-trained and weights are downloaded automatically on first use
- All processing is done on CPU (no GPU required)
- Files are saved in CSV format for easy loading and inspection
