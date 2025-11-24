# Session 4 Summary - MSD Feature Training & FMA RTX Setup
Date: 2025-11-24

## Key Accomplishments

### 1. MSD Feature-Based Training (Completed)
- Created `train_msd.py` - Uses FMA's pre-computed features instead of MSD
- Successfully trained on 17,000 FMA tracks with 518 features
- **Best Validation Accuracy: 77.09%** (achieved at epoch 8)
- Training time: ~2 minutes for 50 epochs on GPU
- 16 genres: Blues, Classical, Country, Easy Listening, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Jazz, Old-Time/Historic, Pop, Rock, Soul-RnB, Spoken
- Model saved: `/media/mijesu_970/SSD_Data/AI_models/msd_model.pth` (672 KB)

### 2. Documentation Created
- **CLASSIFICATION_FEATURES.md** - Comprehensive guide on music classification features
  - Audio features (mel-spectrograms, MFCCs, chroma, spectral)
  - Pre-computed features (timbre, pitch, rhythm, dynamics)
  - High-level features (danceability, energy, valence, acousticness)
  - Model comparison table
  - Feature extraction workflows
  - Accuracy expectations and recommendations

### 3. FMA RTX Training Script
- Created `train_fma_rtx.py` - RTX 4060 Ti optimized training
- Uses FMA Medium audio files (25,000 tracks)
- Simple CNN with mixed precision training
- Batch size 32, 30 epochs
- Expected: 70-80% accuracy in 2-4 hours
- Saves to: `/media/mijesu_970/SSD_Data/AI_models/fma_rtx_model.pth`

### 4. Tagtraum Dataset Investigation
- Attempted to download Tagtraum genre annotations for MSD
- Original URLs no longer available (404 errors)
- Alternative solution: Used FMA's built-in genre labels instead
- FMA provides better integration with pre-computed features

## Feature Types Explained

### Audio Features (from raw audio)
- **Mel-spectrograms**: Visual frequency representation (128x128 or 224x224)
- **MFCCs**: Timbral texture (13-40 coefficients)
- **Chroma**: Pitch class distribution (12 semitones)
- **Spectral**: Centroid, rolloff, contrast, bandwidth

### Pre-computed Features (MSD/FMA style)
- **Timbre**: 12 dimensions - tone color/texture
- **Pitch**: 12 dimensions - note distribution
- **Rhythm**: Tempo, beat strength, danceability
- **Dynamics**: Loudness, energy
- **Structure**: Duration, key, mode

### High-level Features
- **Danceability**: Suitability for dancing
- **Energy**: Intensity and activity
- **Valence**: Musical positivity/mood
- **Acousticness**: Acoustic vs electronic

## Model Performance Comparison

| Model | Features | Dataset | Accuracy | Training Time |
|-------|----------|---------|----------|---------------|
| JMLA (OpenJMLA) | Mel-spectrograms | Pre-trained | Clustering | N/A |
| GTZAN Basic (train_gtzan_rtx.py) | Mel-spectrograms | 1,000 tracks | 70-85% | 15-20 min |
| GTZAN Enhanced | Mel-spectrograms | 1,000 tracks | 80-90% | 4 hours |
| **MSD Features (train_msd.py)** | **518 pre-computed** | **17,000 tracks** | **77%** | **2 min** |
| FMA RTX (train_fma_rtx.py) | Mel-spectrograms | 25,000 tracks | 70-80% | 2-4 hours |
| Combined Multi-Modal | Both | FMA+MSD | 75-85% | 4 hours |

## Files Created/Modified

### New Scripts
1. **train_msd.py** - Feature-based training using FMA pre-computed features
2. **train_fma_rtx.py** - RTX-optimized FMA audio training
3. **download_tagtraum.py** - Tagtraum download helper (sources unavailable)

### Documentation
1. **CLASSIFICATION_FEATURES.md** - Complete feature guide
2. **SESSION_4_SUMMARY.md** - This file

### Models
1. **msd_model.pth** - 77% accuracy, 16 genres, 672 KB
   - Location: `/media/mijesu_970/SSD_Data/AI_models/`
   - Contains: model weights, scaler parameters, genre list

## Key Insights

### Why Feature-Based Training is Fast
- No audio loading/processing overhead
- Features pre-computed in CSV format
- Simple MLP vs complex CNN
- 518 features → 256 → 128 → 16 classes
- ~1.5 seconds per epoch vs ~30 seconds for audio

### Why Spectrograms Work Better (Usually)
- Captures time-frequency relationships
- CNNs learn visual patterns
- Genre-specific signatures visible
- Transfer learning from image models
- But requires more compute and time

### FMA Dataset Advantages
- 25,000 tracks (vs GTZAN's 1,000)
- 16 genres (vs GTZAN's 10)
- Pre-computed features available
- Metadata includes Echo Nest features
- Better for generalization

## Training Recommendations

### For Quick Testing (2 min)
```bash
python3 train_msd.py  # 77% accuracy, feature-based
```

### For Best Accuracy (2-4 hours)
```bash
python3 train_fma_rtx.py  # 70-80% accuracy, audio-based
```

### For Production (8-12 hours)
- Ensemble: MSD features + FMA audio + GTZAN fine-tuning
- Expected: 85-90% accuracy

## Dataset Locations

- **FMA Medium Audio**: `/media/mijesu_970/SSD_Data/DataSets/FMA/Data/fma_medium/`
- **FMA Metadata**: `/media/mijesu_970/SSD_Data/DataSets/FMA/Misc/fma_metadata/`
- **FMA Features CSV**: `features.csv` (518 pre-computed features)
- **FMA Tracks CSV**: `tracks.csv` (genre labels, metadata)
- **GTZAN**: `/media/mijesu_970/SSD_Data/DataSets/GTZAN/Data/genres_original/`
- **AI Models**: `/media/mijesu_970/SSD_Data/AI_models/`

## Model Storage Policy
- All `.pth` model files stored in: `/media/mijesu_970/SSD_Data/AI_models/`
- Updated `train_msd.py` to save directly to AI_models folder
- `train_fma_rtx.py` configured to save to AI_models folder

## Next Steps (Not Started)

1. **Run FMA RTX Training**
   ```bash
   cd /media/mijesu_970/SSD_Data/Python/Music_Reclass
   python3 train_fma_rtx.py
   ```

2. **Create Classification Script for MSD Model**
   - Load msd_model.pth
   - Extract features from audio
   - Classify using trained model

3. **Create Ensemble Model**
   - Combine MSD features + audio spectrograms
   - Voting or weighted average
   - Target: 85-90% accuracy

4. **Test on Musics_TBC Folder**
   - Classify 25 Chinese music files
   - Compare with JMLA results
   - Validate accuracy improvements

## Technical Notes

### FMA Directory Structure
```
fma_medium/
├── 000/
│   ├── 000002.mp3
│   ├── 000005.mp3
│   └── ...
├── 001/
│   └── ...
```

### Feature Extraction (MSD/FMA)
- Chroma CENS: 12 features
- MFCC: 20 features
- Spectral: Centroid, rolloff, contrast, bandwidth
- Tonnetz: Tonal centroid features
- Zero crossing rate
- Total: 518 features per track

### GPU Optimization
- Mixed precision training (torch.cuda.amp)
- Batch size 32 (optimal for RTX 4060 Ti 16GB)
- Gradient scaling for stability
- AdaptiveAvgPool for flexible input sizes

## Session Statistics
- Scripts created: 3
- Documentation files: 2
- Models trained: 1 (msd_model.pth)
- Training time: 2 minutes
- Best accuracy achieved: 77.09%
- Dataset size: 17,000 tracks
- Feature dimensions: 518
