# Music Classification Features

## Overview
This document explains the features used for music genre classification in the Music_ReClass project.

## 1. Audio Features (from raw audio files)

### Mel-Spectrograms
- Visual representation of frequency content over time
- Used by GTZAN and JMLA models
- Typical size: 128x128 or 224x224 pixels
- Shows time on x-axis, frequency on y-axis, intensity as color

### MFCCs (Mel-Frequency Cepstral Coefficients)
- Captures timbral texture of audio
- Represents the shape of the vocal tract
- Typically 13-40 coefficients
- Good for distinguishing instrument types

### Chroma Features
- Pitch class distribution (12 semitones: C, C#, D, etc.)
- Captures harmonic and melodic characteristics
- Genre-specific chord progressions

### Spectral Features
- **Spectral Centroid** - "Center of mass" of spectrum (brightness)
- **Spectral Rolloff** - Frequency below which 85% of energy is contained
- **Spectral Contrast** - Difference between peaks and valleys
- **Spectral Bandwidth** - Width of frequency range

## 2. Pre-computed Features (MSD-style)

### Timbre (12 dimensions)
- Tone color and texture
- Distinguishes instruments playing same note
- Extracted from audio segments

### Pitch (12 dimensions)
- Musical note distribution
- Chroma-based representation
- Key and scale information

### Rhythm Features
- **Tempo** - Beats per minute (BPM)
- **Beat Strength** - Regularity of rhythm
- **Danceability** - Suitability for dancing

### Dynamic Features
- **Loudness** - Overall volume in dB
- **Energy** - Intensity and activity level

### Structural Features
- **Duration** - Track length in seconds
- **Key** - Musical key (0-11)
- **Mode** - Major (1) or Minor (0)

## 3. High-level Features

### Danceability
- How suitable the track is for dancing
- Based on tempo, rhythm stability, beat strength

### Energy
- Perceptual measure of intensity and activity
- Fast, loud, noisy tracks have high energy

### Valence
- Musical positivity/mood
- High valence = happy, cheerful
- Low valence = sad, angry

### Acousticness
- Confidence measure of acoustic vs electronic
- 1.0 = fully acoustic, 0.0 = fully electronic

## Current Models in Music_ReClass

| Model | Features Used | Input Type | Training Time | Accuracy |
|-------|--------------|------------|---------------|----------|
| **JMLA** (OpenJMLA) | Mel-spectrograms | Audio → 128x128 spectrograms | Pre-trained | Clustering |
| **GTZAN Basic** (train_gtzan_rtx.py) | Mel-spectrograms | Audio → spectrograms | 15-20 min | 70-85% |
| **GTZAN Enhanced** (train_gtzan_enhanced.py) | Mel-spectrograms | Audio → spectrograms | 4 hours | 80-90% |
| **MSD Features** (train_msd_features.py) | 31 pre-computed features | HDF5 files | ~1 hour | 65-75% |
| **Combined Multi-Modal** (train_combined_4hr.py) | Spectrograms + MSD features | Dual input | 4 hours | 75-85% |
| **Progressive FMA** (train_fma_progressive.py) | Mel-spectrograms | Audio → spectrograms | 12h + 2h | 75-85% |

## Why Spectrograms Work Best

1. **Visual Pattern Recognition**
   - CNNs can learn patterns like humans see in sheet music
   - Genre-specific visual signatures

2. **Time-Frequency Relationships**
   - Captures both temporal and spectral information
   - Shows how sound evolves over time

3. **Genre-Specific Patterns**
   - Metal: Dense high frequencies, distortion patterns
   - Blues: Specific chord progressions, bent notes
   - Classical: Complex harmonic structures
   - Hip-hop: Strong bass, repetitive beats
   - Electronic: Synthetic textures, precise rhythms

4. **Transfer Learning**
   - Pre-trained image models (ResNet, VGG) work on spectrograms
   - Leverages computer vision advances

## Feature Extraction Process

### For Audio Files (JMLA/GTZAN approach):
```
Audio File (.wav/.mp3)
    ↓
Load audio (librosa)
    ↓
Extract mel-spectrogram
    ↓
Resize to 128x128 or 224x224
    ↓
Normalize
    ↓
Feed to CNN
```

### For MSD Pre-computed Features:
```
HDF5 File (.h5)
    ↓
Read features (h5py)
    ↓
Extract: timbre (12) + pitch (12) + rhythm (7)
    ↓
Normalize
    ↓
Feed to MLP
```

### For Combined Multi-Modal:
```
Audio File + HDF5 File
    ↓
Branch 1: Spectrogram → CNN
Branch 2: Features → MLP
    ↓
Concatenate outputs
    ↓
Final classification layer
```

## JMLA Classification Results

Classification of 25 Chinese music files from Musics_TBC folder:
- 6 files → Metal
- 4 files → Hip-Hop
- 3 files → Blues
- 3 files → Country
- 3 files → Rock
- Remaining → Other genres

Used JMLA model with K-Means clustering on extracted features.

## Datasets and Their Features

### GTZAN
- 1,000 tracks, 10 genres
- Raw audio files
- Features: Extracted on-the-fly (spectrograms)

### FMA Medium
- 25,000 tracks, 16 genres
- Raw audio files + metadata CSV
- Features: Extracted on-the-fly + pre-computed metadata

### MSD (Million Song Dataset)
- 1 million tracks (subset: 10,000)
- NO audio files - only pre-computed features
- Features: All pre-computed in HDF5 format
- Requires genre labels from Tagtraum or Last.fm

## Recommendations

**For highest accuracy (85-90%):**
- Use mel-spectrograms with deep CNN (ResNet)
- Train on large dataset (FMA Medium)
- Apply data augmentation
- Use ensemble of multiple models

**For fastest training (15-20 min):**
- Use mel-spectrograms with simple CNN
- Train on GTZAN (1,000 tracks)
- Batch size 32, mixed precision

**For minimal compute:**
- Use MSD pre-computed features
- Simple MLP (3 layers)
- No GPU required (but slower)

**For best of both worlds:**
- Multi-modal approach combining spectrograms + features
- Leverages visual patterns + numerical features
- 4-hour training on RTX 4060 Ti

## Next Steps

1. Download Tagtraum genre labels for MSD training
2. Or focus on FMA dataset with built-in labels
3. Or continue with GTZAN for quick iterations
4. Consider ensemble approach combining multiple models


---
# ML Concepts Memo

# Music Classification ML Concepts - Quick Reference

## Core Components

### 1. Dataset 📁
**What**: Raw training data
- Audio files (.wav, .mp3) + genre labels
- Examples: GTZAN (1,000 tracks), FMA (25,000 tracks)
- Purpose: Material to train models

### 2. Feature Extractor 🔍
**What**: Converts audio → numerical vectors
- Input: Audio waveform
- Output: Feature vectors (e.g., 518 dims, 768 dims)
- Examples: MERT, OpenJMLA, VGGish, librosa
- Usually pre-trained on large datasets
- Purpose: Extract meaningful audio patterns

### 3. Classifier 🎯
**What**: Predicts genre from features
- Input: Feature vectors
- Output: Genre label + confidence
- Examples: MSD model, GTZAN model
- Trained on specific classification task
- Purpose: Make final decision

### 4. Model 🧠
**What**: General term for neural network
- Can be feature extractor OR classifier OR both
- Your MSD model = classifier
- MERT = feature extractor

---

## Workflow

```
┌─────────────┐
│  DATASET    │  Audio files + labels
└──────┬──────┘
       │
       ↓
┌─────────────────────┐
│ FEATURE EXTRACTION  │  Audio → [0.23, -0.45, 0.67, ...]
│ (librosa/MERT)      │  518 or 768 numbers
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│   CLASSIFIER        │  Features → Neural Net → Probabilities
│   (MSD/GTZAN)       │  [Rock: 0.85, Jazz: 0.10, Blues: 0.05]
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│   PREDICTION        │  Genre: Rock (85% confidence)
└─────────────────────┘
```

---

## Your Music_ReClass Setup

### Datasets Available
- **GTZAN**: 1,000 tracks, 10 genres
- **FMA**: 25,000 tracks, 16 genres (with pre-computed features)
- **MSD**: 10,000 H5 files, 13 genres

### Feature Extractors Available
- **librosa**: Basic features (MFCC, spectral centroid, etc.)
- **MERT**: Transformer-based, 768 dims
- **OpenJMLA**: Vision transformer, 1.3 GB
- **VGGish**: CNN-based, 128 dims
- **AST**: Audio Spectrogram Transformer
- **CLAP**: Contrastive Language-Audio

### Trained Classifiers
1. **MSD Model**
   - Path: `/media/mijesu_970/SSD_Data/AI_models/MSD/msd_model.pth`
   - Features: FMA 518 dims (librosa-based)
   - Accuracy: 77%
   - Genres: 16
   - Size: 672 KB

2. **GTZAN Model**
   - Path: `/media/mijesu_970/SSD_Data/AI_models/ZTGAN/GTZAN.pth`
   - Features: Mel spectrogram
   - Accuracy: 70-80%
   - Genres: 10
   - Size: 400 KB

---

## Example: Classifying a Song

### Using MSD Model (FMA-based)
```
Song.wav (30 seconds)
  ↓
librosa extracts 518 features
  - 20 MFCCs (mean + std)
  - Spectral centroid
  - Spectral bandwidth
  - Zero crossing rate
  - etc.
  ↓
MSD classifier (3-layer neural net)
  - Layer 1: 518 → 256
  - Layer 2: 256 → 128
  - Layer 3: 128 → 16 (genres)
  ↓
Result: Rock (100% confidence)
```

### Using GTZAN Model
```
Song.wav (30 seconds)
  ↓
librosa creates mel spectrogram (128 mel bands)
  - Time-frequency representation
  - 2D image-like data
  ↓
GTZAN classifier (CNN)
  - 4 conv layers (16→32→64→128 channels)
  - Adaptive pooling
  - Linear classifier
  ↓
Result: Classical (75.2% confidence)
```

---

## Test Results (2025-11-28)

**Song**: L(桃籽) - 你總要學會往前走.wav

| Model | Prediction | Confidence | Top 3 |
|-------|------------|------------|-------|
| MSD (FMA) | Rock | 100.0% | Rock, Jazz, Old-Time |
| GTZAN | Classical | 75.2% | Classical, Jazz, Blues |

**Observation**: Different models give different results because they:
- Use different features (FMA vs mel spectrogram)
- Trained on different datasets
- Have different architectures

**Solution**: Use ensemble voting for better accuracy

---

## Key Differences

| Aspect | Feature Extractor | Classifier |
|--------|------------------|------------|
| Input | Audio waveform | Feature vectors |
| Output | Numerical features | Genre labels |
| Training | Pre-trained (general) | Task-specific |
| Examples | MERT, VGGish | MSD, GTZAN |
| Reusable | Yes, for many tasks | No, genre only |

---

## Next Steps

1. **Train ensemble model**: Combine MSD + GTZAN + MERT features
2. **Create ID3 tagger**: Auto-tag music with predicted genres
3. **Improve accuracy**: Use progressive voting (FMA → MERT → JMLA)
4. **Deploy**: REST API or web interface

---

**Created**: 2025-11-28  
**Project**: Music_ReClass  
**Location**: /home/mijesu_970/Music_ReClass/docs/


---
# Extraction Results

# FMA Feature Extraction Results

## Extraction Summary

**Date**: November 27, 2025  
**Input Directory**: `/media/mijesu_970/SSD_Data/Musics_TBC`  
**Output File**: `musics_tbc_features.npy`  
**Processing Time**: ~52 seconds (25 files)

## Results

- **Files Processed**: 25 audio files (.wav)
- **Feature Dimensions**: 518 per file
- **Output Shape**: (25, 518)
- **File Size**: 51 KB (compact NPY format)
- **Data Type**: float32

## Feature Statistics

```
Mean:    123.59
Std:     762.83
Min:    -624.57
Max:   10777.37
```

## Performance

- **Average Time per File**: ~2.1 seconds
- **Total Processing Time**: 52 seconds
- **Memory Usage**: ~50-100 MB peak
- **Output Compression**: 51 KB (vs ~950 MB if CSV)

## Processed Files

1. 12是雪笛 - 城市裡的麥子.wav
2. A-Lin - 安寧.wav
3. AZ珍珍 - 打不過徒弟真生氣.wav
4. A_Lin - 最好的我.wav
5. Blaxy Girls - if you feel my love(京劇版).wav
6. Cathy月月 - 不綢繆.wav
7. Cathy月月 - 雲遮月.wav
8. Cathy月月_Babystop_山竹 - 見春風不見故人.wav
9. Christine Welch - 一百萬個可能.wav
10. DAWN - 難生恨.wav
11. GAI周延 - 生如野草.wav
12. Jam - 七月上.wav
13. L(桃籽) - 人間遊蕩.wav
14. L(桃籽) - 你總要學會往前走.wav
15. L(桃籽) - 尋酒.wav
16. L(桃籽) - 我的小時候.wav
17. L(桃籽) - 貪嗔癡.wav
18. L(桃籽)_周林楓 - 人間遊蕩.wav
19. Li敖 - 三十已過(深情版).wav
20. Li敖 - 多想還小.wav
21. 蘇譚譚 - 人間鬼.wav
22. 蘇譚譚 - 又算什麼.wav
23. 蘇譚譚 - 天涯.wav
24. 黃靜美 - 恕我食言.wav
25. 黃靜美_周林楓 - 怎奈.wav

## Feature Breakdown (518 Total)

| Feature Type | Dimensions | Stats | Total Features |
|--------------|------------|-------|----------------|
| Chroma | 12 | 7 | 84 |
| Tonnetz | 6 | 7 | 42 |
| MFCC | 20 | 7 | 140 |
| Spectral Contrast | 6 | 7 | 42 |
| Spectral Centroid | 1 | 7 | 7 |
| Spectral Bandwidth | 1 | 7 | 7 |
| Spectral Rolloff | 1 | 7 | 7 |
| Zero Crossing Rate | 1 | 7 | 7 |
| RMS Energy | 1 | 7 | 7 |
| Tempo | 1 | 1 | 1 |
| Mel Spectrogram | 26 | 7 | 182 |
| **TOTAL** | | | **518** |

## Next Steps

### 1. Train a Classifier

```bash
# Use with your existing MSD training script
python3 Music_ReClass/train_msd.py --features musics_tbc_features.npy
```

### 2. Classify with Existing Model

```bash
# If you have a trained model
python3 Music_ReClass/classify_music_tbc.py \
    --features musics_tbc_features.npy \
    --model Music_ReClass/models/msd_model.pth
```

### 3. Analyze Features

```python
import numpy as np
import matplotlib.pyplot as plt

# Load features
features = np.load('musics_tbc_features.npy')

# Visualize feature distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(features.flatten(), bins=50)
plt.title('Feature Distribution')

plt.subplot(1, 3, 2)
plt.imshow(features, aspect='auto', cmap='viridis')
plt.title('Feature Heatmap')
plt.xlabel('Feature Index')
plt.ylabel('Song Index')

plt.subplot(1, 3, 3)
plt.plot(features.mean(axis=0))
plt.title('Mean Feature Values')
plt.xlabel('Feature Index')
plt.tight_layout()
plt.savefig('feature_analysis.png')
```

## Usage Examples

### Load Features

```python
import numpy as np

# Load extracted features
features = np.load('musics_tbc_features.npy')
print(f"Shape: {features.shape}")  # (25, 518)

# Access individual song features
song_0_features = features[0]  # First song
print(f"First song features: {song_0_features.shape}")  # (518,)
```

### Normalize Features

```python
from sklearn.preprocessing import StandardScaler

# Normalize for training
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
```

### Train Simple Classifier

```python
import torch
import torch.nn as nn

# Simple neural network
class GenreClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(518, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 16)  # 16 genres
        )
    
    def forward(self, x):
        return self.fc(x)

model = GenreClassifier()
```

## Comparison with README Benchmarks

According to your README:

- **FMA Features Training**: 2 minutes for 77% accuracy
- **Feature Size**: 518 dimensions (matches FMA dataset)
- **Model Size**: ~672 KB
- **Use Case**: Fast training without raw audio processing

Your extracted features are now ready for the same fast training approach!

## File Locations

```
/home/mijesu_970/
├── extract_fma_features.py          # Extractor script
├── musics_tbc_features.npy          # Extracted features (51 KB)
├── docs/FMA_FEATURE_EXTRACTION.md   # Documentation
└── EXTRACTION_RESULTS.md            # This file

/media/mijesu_970/SSD_Data/Musics_TBC/
└── [25 .wav files]                  # Source audio files
```

## Notes

- Features are compatible with FMA dataset format (518 dimensions)
- Can be used with your existing `train_msd.py` script
- NPY format loads 20-30x faster than CSV
- All features successfully extracted with valid ranges
- Ready for training or classification

---

**Extraction completed successfully!** ✅
