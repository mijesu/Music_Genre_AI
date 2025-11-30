# Model Notes - FMA, MERT, JMLA, MSD

Complete reference for all feature extraction and classification models used in Music_ReClass.

---

## FMA Model (Feature-Based)

### Overview
- **Type**: Hand-crafted audio features
- **Dimensions**: 518 features
- **Training Time**: 2 minutes
- **Accuracy**: 77%
- **Model Size**: 672 KB
- **Genres**: 16

### Feature Breakdown (518 dimensions)

#### 1. Chroma Features (16 dims)
- **Chroma STFT**: 12 pitch classes + 4 statistics (mean, std, min, max)
- **Purpose**: Pitch class distribution, harmony analysis
- **Use**: Identifying tonal characteristics

#### 2. Tonnetz Features (10 dims)
- **Tonal Centroids**: 6 dimensions + 4 statistics
- **Purpose**: Harmonic relationships
- **Use**: Chord and key detection

#### 3. MFCC Features (80 dims)
- **Coefficients**: 20 MFCCs × 4 statistics each
- **Purpose**: Timbral texture
- **Use**: Voice/instrument characteristics

#### 4. Spectral Features (36 dims)
- **Spectral Centroid**: 4 statistics (brightness)
- **Spectral Bandwidth**: 4 statistics (frequency spread)
- **Spectral Contrast**: 7 bands × 4 statistics (28 dims)
- **Spectral Rolloff**: 4 statistics (frequency cutoff)

#### 5. Rhythm Features (9 dims)
- **Zero Crossing Rate**: 4 statistics (noisiness)
- **RMS Energy**: 4 statistics (loudness)
- **Tempo**: 1 value (BPM)

#### 6. Mel Spectrogram Statistics (512 dims)
- **Mel Bands**: 128 bands × 4 statistics each
- **Purpose**: Frequency distribution over time
- **Use**: Overall spectral shape

### Model Architecture
```
Input: 518 features
├── Linear(518 → 256) + ReLU + Dropout(0.3)
├── Linear(256 → 128) + ReLU + Dropout(0.3)
└── Linear(128 → 16) [Output: 16 genres]
```

### Training Details
- **Dataset**: 17,000 FMA tracks
- **Epochs**: 7
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss
- **Validation Accuracy**: 77.09%

### File Locations
- **Features**: `features/FMA_features.npy` (211 MB)
- **Model**: `models/msd_model.pth` (672 KB)
- **Extractor (DB)**: `extractors/extract_fma.py`
- **Extractor (Standalone)**: `extractors/extract_fma_features.py`

### Advantages
- ✅ Very fast training (2 minutes)
- ✅ Small model size (672 KB)
- ✅ No GPU required for extraction
- ✅ Interpretable features
- ✅ Good baseline accuracy (77%)

### Limitations
- ❌ Fixed feature set (not learned)
- ❌ Lower accuracy than deep learning
- ❌ Manual feature engineering required

---

## MERT Model (Music Transformer)

### Overview
- **Type**: Music-specific Transformer (pre-trained)
- **Full Name**: MERT-v1-330M
- **Dimensions**: 768 features
- **Parameters**: 330 million
- **Model Size**: 1.2 GB
- **Accuracy**: 82-88% (with FMA)
- **Genres**: 16

### Architecture
- **Base**: Transformer encoder (similar to BERT)
- **Pre-training**: Self-supervised on large music corpus
- **Specialization**: Music understanding (rhythm, melody, harmony)
- **Embedding Layer**: 768-dimensional representations

### Feature Characteristics
- **Learned Representations**: Automatically learned from audio
- **Multi-scale**: Captures both local and global patterns
- **Contextual**: Understands temporal relationships
- **Transfer Learning**: Pre-trained on diverse music

### Model Details
```
Input: Audio waveform (24kHz)
├── Feature Extraction (CNN frontend)
├── Transformer Encoder (12 layers)
│   ├── Multi-head Self-Attention
│   ├── Feed-forward Network
│   └── Layer Normalization
└── Output: 768-dim embedding
```

### Training Details
- **Pre-training Dataset**: Large-scale music corpus
- **Fine-tuning**: FMA dataset (16 genres)
- **Sample Rate**: 24,000 Hz
- **Duration**: 30 seconds per track
- **Batch Size**: 8-16 (GPU dependent)

### File Locations
- **Model**: Downloaded from Hugging Face (`m-a-p/MERT-v1-330M`)
- **Features**: `features/MERT_features.npy` (768 dims per track)
- **Extractor (DB)**: `extractors/extract_mert.py`
- **Extractor (Standalone)**: `extractors/extract_mert_features.py`

### Advantages
- ✅ High accuracy (82-88% with FMA)
- ✅ Learned representations (no manual engineering)
- ✅ Music-specific pre-training
- ✅ Captures complex patterns
- ✅ Good for Stage 2 classification

### Limitations
- ❌ Large model size (1.2 GB)
- ❌ Requires GPU for extraction
- ❌ Slower extraction (~30-60s per track)
- ❌ Less interpretable than hand-crafted features

### Usage Example
```python
from transformers import AutoModel
import torch

# Load model
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
model.eval()

# Extract features
with torch.no_grad():
    features = model(audio_tensor)  # Output: [batch, 768]
```

---

## JMLA Model (OpenJMLA Vision Transformer)

### Overview
- **Type**: Vision Transformer for audio spectrograms
- **Full Name**: OpenJMLA (Joint Music-Language Audio)
- **Dimensions**: 768 features
- **Parameters**: 86 million
- **Model Size**: 1.3 GB
- **Accuracy**: 85-94% (with FMA+MERT)
- **Genres**: 16

### Architecture
- **Base**: Vision Transformer (ViT)
- **Input**: Log-mel spectrograms (treated as images)
- **Specialization**: Audio-visual pattern recognition
- **Embedding**: 768-dimensional representations

### Model Details
```
Input: Audio waveform (16kHz)
├── Mel Spectrogram Conversion
│   ├── n_fft: 1024
│   ├── hop_length: 160
│   └── n_mels: 128
├── Vision Transformer
│   ├── Patch Embedding (16×16 patches)
│   ├── Transformer Encoder (12 layers)
│   │   ├── Multi-head Self-Attention
│   │   ├── MLP
│   │   └── Layer Normalization
│   └── Classification Head
└── Output: 768-dim embedding
```

### Training Details
- **Pre-training**: Large-scale audio-visual dataset
- **Fine-tuning**: Music genre classification
- **Sample Rate**: 16,000 Hz
- **Duration**: 30 seconds per track
- **Spectrogram Size**: 128 × 1876 (mel bins × time frames)

### File Locations
- **Model**: `AI_models/OpenJMLA/epoch_4-step_8639-allstep_60000.pth` (1.3 GB)
- **Checkpoint**: `AI_models/OpenJMLA/epoch_20.pth` (330 MB - early version)
- **Features**: `features/JMLA_features.npy` (768 dims per track)
- **Extractor (DB)**: `extractors/extract_jmla.py`
- **Extractor (Standalone)**: `extractors/extract_jmla_features.py`
- **Text-based**: `extractors/extract_jmla_simple.py`
- **Config**: `openjmla_parameters.json`

### Advantages
- ✅ Highest accuracy (85-94% in ensemble)
- ✅ Visual pattern recognition on spectrograms
- ✅ Excellent for difficult classifications
- ✅ Best for Stage 3 (final decision)
- ✅ Complementary to MERT (different approach)

### Limitations
- ❌ Largest model size (1.3 GB)
- ❌ Slowest extraction (~50-100s per track)
- ❌ Requires significant GPU memory
- ❌ Most computationally expensive

### Usage Example
```python
from transformers import AutoModel
import librosa

# Load model
model = AutoModel.from_pretrained('UniMus/OpenJMLA', trust_remote_code=True)
model.eval()

# Convert audio to mel spectrogram
y, sr = librosa.load(audio_path, sr=16000, duration=30)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=160, n_mels=128)
log_mel = librosa.power_to_db(mel_spec)

# Extract features
with torch.no_grad():
    features = model(torch.tensor(log_mel).unsqueeze(0))  # Output: [1, 768]
```

---

## MSD Model (Million Song Dataset)

### Overview
- **Type**: Feature-based classifier (trained on MSD features)
- **Dimensions**: 518 input features (same as FMA)
- **Model Size**: 672 KB
- **Accuracy**: 77.09%
- **Genres**: 16
- **Training Dataset**: 17,000 FMA tracks

### Dataset Information

#### MSD Files Available
1. **msd_model.pth** (672 KB)
   - Trained PyTorch neural network
   - 6-layer MLP architecture
   - Scaler parameters for 518 features
   - 16 genre classes
   - Epoch 7, 77.09% validation accuracy

2. **msd_tagtraum_cd1.cls** (3.8 MB)
   - Plain text label file
   - Format: `TRACKID\tGenre1\tGenre2`
   - Total labels: 133,676 songs
   - Purpose: Training/validation labels

3. **H5 Feature Files** (~2.6 GB)
   - Location: `Data/A/` and `Data/B/`
   - Count: 10,000 songs
   - Format: HDF5 with pre-computed features
   - Features: Timbre (12), Pitch (12), Tempo, Loudness, Duration

### Model Architecture
```
Input: 518 features (FMA-style)
├── Linear(518 → 256) + ReLU + Dropout(0.3)
├── Linear(256 → 128) + ReLU + Dropout(0.3)
└── Linear(128 → 16) [Output: 16 genres]
```

### Training Details
- **Dataset**: 17,000 FMA tracks (not actual MSD)
- **Features**: FMA 518-dimensional features
- **Epochs**: 7
- **Validation Accuracy**: 77.09%
- **Training Time**: 2 minutes

### File Locations
- **Model**: `AI_models/MSD/msd_model.pth` (672 KB)
- **Labels**: `AI_models/MSD/msd_tagtraum_cd1.cls` (3.8 MB)
- **H5 Files**: `AI_models/MSD/Data/` (10,000 files)
- **Training Script**: `training/train_msd.py`

### Data Organization
```
MSD/
├── msd_model.pth              # Trained model
├── msd_tagtraum_cd1.cls       # Genre labels (133K songs)
└── Data/
    ├── A/                     # H5 files (5,000 songs)
    └── B/                     # H5 files (5,000 songs)
```

### Advantages
- ✅ Very fast training (2 minutes)
- ✅ Small model size (672 KB)
- ✅ Good baseline accuracy (77%)
- ✅ Large label dataset available (133K songs)

### Limitations
- ❌ Model trained on FMA features, not actual MSD features
- ❌ Only 10,000 H5 files available (not full 1M dataset)
- ❌ Fixed feature set
- ❌ Lower accuracy than deep learning approaches

### Getting Better MSD Models

#### Option 1: Train Your Own (Recommended)
- Use existing 10,000 H5 files
- Modify `training/train_msd_features.py`
- Add more layers, increase epochs
- Tune hyperparameters

#### Option 2: Official MSD Resources
- Website: http://millionsongdataset.com/
- GitHub: https://github.com/tbertinmahieux/MSongsDB
- Official benchmarks and pre-trained models

#### Option 3: AcousticBrainz
- Website: https://acousticbrainz.org/
- Pre-computed audio features for MSD
- Hundreds of additional features

#### Option 4: Research Papers
- "Deep content-based music recommendation" (van den Oord et al.)
- Various MSD-trained deep learning models
- Search GitHub for MSD implementations

#### Option 5: Full MSD Download
- Complete subset: 280 GB
- Contains: All 1M songs features
- Command: `wget http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_summary_file.h5`

---

## Model Comparison

| Model | Type | Size | Dims | Accuracy | Speed | GPU | Best For |
|-------|------|------|------|----------|-------|-----|----------|
| **FMA** | Hand-crafted | 672 KB | 518 | 77% | Very Fast | No | Stage 1, Baseline |
| **MERT** | Transformer | 1.2 GB | 768 | 82-88% | Medium | Yes | Stage 2, Music understanding |
| **JMLA** | ViT | 1.3 GB | 768 | 85-94% | Slow | Yes | Stage 3, Maximum accuracy |
| **MSD** | Feature-based | 672 KB | 518 | 77% | Very Fast | No | Quick testing |

---

## Progressive Voting Strategy

### Stage 1: FMA Only (30% of songs)
- **Features**: 518 dimensions
- **Time**: 0 seconds (pre-computed)
- **Accuracy**: 77%
- **Use**: High-confidence classifications

### Stage 2: FMA + MERT (50% of songs)
- **Features**: 1,286 dimensions (518 + 768)
- **Time**: 30-60 seconds
- **Accuracy**: 82-88%
- **Use**: Medium-confidence classifications

### Stage 3: FMA + MERT + JMLA (20% of songs)
- **Features**: 2,054 dimensions (518 + 768 + 768)
- **Time**: 50-100 seconds
- **Accuracy**: 85-94%
- **Use**: Difficult classifications, maximum accuracy

### Weighted Voting
```python
# Confidence thresholds
if fma_confidence > 0.85:
    return fma_prediction  # Stage 1
elif fma_mert_confidence > 0.80:
    return weighted_vote(fma, mert)  # Stage 2
else:
    return weighted_vote(fma, mert, jmla)  # Stage 3
```

**Average Processing Time**: 20-40 seconds per track

---

## Feature Extraction Scripts

### Database Versions (Production)
- `extractors/extract_fma.py` - FMA features with SQLite
- `extractors/extract_mert.py` - MERT features with SQLite
- `extractors/extract_jmla.py` - JMLA features with SQLite
- `extractors/extract_all_features.py` - Master orchestrator

### Standalone Versions (Testing)
- `extractors/extract_fma_features.py` - FMA standalone
- `extractors/extract_mert_features.py` - MERT standalone
- `extractors/extract_jmla_features.py` - JMLA standalone

### Special Purpose
- `extractors/extract_jmla_simple.py` - Text-based JMLA
- `extractors/compare_features.py` - Feature comparison
- `extractors/visualize_fma_features.py` - Feature visualization

---

## Training Scripts

- `training/train_msd.py` - Fast FMA-based training (2 min, 77%)
- `training/train_mert_classifier.py` - MERT classifier training
- `training/train_jmla_classifier.py` - JMLA classifier training
- `training/train_fma_progressive.py` - Progressive voting ensemble
- `training/train_combined_4hr.py` - Combined training

---

## Recommendations

### For Quick Testing
- Use **FMA** features
- Training time: 2 minutes
- Accuracy: 77%
- No GPU required

### For Production
- Use **Progressive Voting** (FMA + MERT + JMLA)
- Average time: 20-40 seconds per track
- Accuracy: 85-94%
- GPU required

### For Maximum Accuracy
- Use **Full Ensemble** (all three models)
- Processing time: 50-100 seconds per track
- Accuracy: 85-94%
- GPU required

### For Research
- Train custom models on **MSD** dataset
- Experiment with different architectures
- Combine with other feature extractors

---

**Last Updated**: November 30, 2025  
**Version**: 2.0  
**Status**: ✅ Production Ready
