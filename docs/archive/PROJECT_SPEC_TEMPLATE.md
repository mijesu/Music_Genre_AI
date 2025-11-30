# Music_ReClass Project Specification

## 1. Project Overview

**Project Name:** Music_ReClass  
**Version:** 1.0  
**Date:** 2025-11-28  
**Owner:** mijesu  
**Status:** Production Ready

**Description:**  
Automatic music genre classification using AI/Deep Learning on NVIDIA Jetson and RTX platforms.

**Goals:**
- Fast training (2 min for 77% accuracy)
- High accuracy (up to 94% with ensemble)
- Production-ready inference
- Multi-model support

---

## 2. System Architecture

### 2.1 Hardware Requirements
- **Primary:** NVIDIA Jetson (ARM64 with CUDA)
- **Secondary:** RTX 4060 Ti 16GB or similar
- **Storage:** 50+ GB SSD
- **RAM:** 16+ GB

### 2.2 Software Stack
- **OS:** Linux (Ubuntu 22.04)
- **Python:** 3.10.12
- **CUDA:** 12.1+
- **Framework:** PyTorch 2.8.0

### 2.3 Key Libraries
```
torch==2.8.0
torchaudio==2.8.0
librosa==0.11.0
numpy==1.26.4
scikit-learn
xgboost==3.1.2
```

---

## 3. Data Specifications

### 3.1 Datasets

| Dataset | Size | Files | Genres | Format | Location |
|---------|------|-------|--------|--------|----------|
| GTZAN | 1.2 GB | 1,000 | 10 | WAV | `/media/mijesu_970/SSD_Data/DataSets/GTZAN/` |
| FMA | 22 GB | 25,000 | 16 | MP3 | `/media/mijesu_970/SSD_Data/DataSets/FMA/` |
| MSD | 2.6 GB | 10,000 | 13 | H5 | `/media/mijesu_970/SSD_Data/DataSets/MSD/` |
| MTT | - | 25,863 | 50 tags | MP3 | `/media/mijesu_970/SSD_Data/DataSets/MTT/` |"

### 3.2 Genre Labels

**GTZAN (10):** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

**FMA (16):** Blues, Classical, Country, Easy Listening, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Jazz, Old-Time/Historic, Pop, Rock, Soul-RnB, Spoken

**MSD (13):** Blues, Country, Electronic, Folk, International, Jazz, Latin, New Age, Pop_Rock, Rap, Reggae, RnB, Vocal

### 3.3 Feature Specifications

**FMA Features (518 dims):**
- MFCCs: 20 coefficients (mean + std)
- Spectral: centroid, bandwidth, rolloff
- Temporal: zero crossing rate
- Chroma features
- Tonnetz features

**Mel Spectrogram:**
- Sample rate: 22,050 Hz
- N_mels: 128
- Duration: 30 seconds
- Shape: (128, time_steps)

---

## 4. Model Specifications

### 4.1 MSD Model (Feature-Based)

**Architecture:**
```
Input (518) → Dense(256) → ReLU → Dropout(0.3)
           → Dense(128) → ReLU → Dropout(0.3)
           → Dense(16)
```

**Specifications:**
- Input size: 518 features
- Output size: 16 genres
- Parameters: ~150K
- Model size: 672 KB
- Training time: 2 minutes
- Accuracy: 77.09%

**Location:** `/media/mijesu_970/SSD_Data/AI_models/MSD/msd_model.pth`

### 4.2 GTZAN Model (CNN-Based)

**Architecture:**
```
Conv2d(1→16) → BN → ReLU → MaxPool → Dropout(0.2)
Conv2d(16→32) → BN → ReLU → MaxPool → Dropout(0.2)
Conv2d(32→64) → BN → ReLU → MaxPool → Dropout(0.3)
Conv2d(64→128) → BN → ReLU → AdaptiveAvgPool
Linear(128→10)
```

**Specifications:**
- Input: Mel spectrogram (1, 128, time)
- Output size: 10 genres
- Parameters: ~500K
- Model size: 400 KB
- Training time: 45 min - 4 hours
- Accuracy: 70-90%

**Location:** `/media/mijesu_970/SSD_Data/AI_models/ZTGAN/GTZAN.pth`

### 4.3 Feature Extractors

| Model | Size | Type | Output Dims | Use Case |
|-------|------|------|-------------|----------|
| OpenJMLA | 1.3 GB | Vision Transformer | 768 | Best accuracy |
| MERT | 1.2 GB | Audio Transformer | 768 | Music understanding |
| HuBERT | 1.2 GB | Speech Transformer | 768 | Speech/music |
| CLAP | 741 MB | Audio-Text | 512 | Multi-modal |
| VGGish | 276 MB | CNN | 128 | Fast extraction |

---

## 5. Training Specifications

### 5.1 Training Approaches

| Approach | Time | Accuracy | GPU | Model Size |
|----------|------|----------|-----|------------|
| XGBoost | 10 min | 55-60% | No | <1 MB |
| FMA Features | 2 min | 77% | Yes | 672 KB |
| CNN Basic | 45 min | 70-80% | Yes | ~50 MB |
| Transfer Learning | 4 hrs | 80-90% | Yes | ~50 MB |
| Ensemble | 8-12 hrs | 85-92% | Yes | ~100 MB |

### 5.2 Hyperparameters

**MSD Model:**
- Batch size: 128
- Learning rate: 0.001
- Optimizer: Adam
- Epochs: 7
- Loss: CrossEntropyLoss

**GTZAN Model:**
- Batch size: 32
- Learning rate: 0.0001
- Optimizer: Adam
- Epochs: 50-100
- Loss: CrossEntropyLoss
- Data augmentation: Yes

### 5.3 Early Stopping Strategy

| Stage | Features | Time | Accuracy | Usage |
|-------|----------|------|----------|-------|
| 1. FMA only | 518 dims | 0s | 77% | 30% songs |
| 2. FMA + MERT | 1286 dims | 30-60s | 82-88% | 50% songs |
| 3. FMA + MERT + JMLA | 2054 dims | 50-100s | 85-92% | 20% songs |

**Average processing time:** 20-40s per track

---

## 6. Performance Specifications

### 6.1 Accuracy Targets

| Model | Target | Achieved | Status |
|-------|--------|----------|--------|
| MSD | 75% | 77.09% | ✅ |
| GTZAN | 70-80% | 70-90% | ✅ |
| Ensemble | 85% | 85-92% | 🔄 In Progress |

### 6.2 Speed Requirements

- Training: <5 min for baseline
- Inference: <1s per track (single model)
- Batch processing: >100 tracks/min

### 6.3 Resource Limits

- GPU Memory: <8 GB
- CPU Memory: <16 GB
- Disk I/O: <100 MB/s

---

## 7. API Specifications

### 7.1 Classification API

**Input:**
```python
{
  "audio_path": "/path/to/song.wav",
  "model": "msd",  # or "gtzan", "ensemble"
  "top_k": 3
}
```

**Output:**
```python
{
  "genre": "Rock",
  "confidence": 0.85,
  "top_predictions": [
    {"genre": "Rock", "confidence": 0.85},
    {"genre": "Blues", "confidence": 0.10},
    {"genre": "Jazz", "confidence": 0.05}
  ],
  "processing_time": 0.23
}
```

### 7.2 Batch Classification

**Input:**
```python
{
  "audio_dir": "/path/to/music/",
  "model": "msd",
  "write_tags": true
}
```

**Output:**
```python
{
  "total_files": 100,
  "processed": 100,
  "failed": 0,
  "results": [...]
}
```

---

## 8. Testing Specifications

### 8.1 Test Results (2025-11-28)

**Test Song:** L(桃籽) - 你總要學會往前走.wav

| Model | Prediction | Confidence | Top 3 |
|-------|------------|------------|-------|
| MSD | Rock | 100.0% | Rock, Jazz, Old-Time |
| GTZAN | Classical | 75.2% | Classical, Jazz, Blues |

### 8.2 Test Coverage

- Unit tests: Model loading, feature extraction
- Integration tests: End-to-end classification
- Performance tests: Speed, memory usage
- Accuracy tests: Validation set evaluation

---

## 9. Deployment Specifications

### 9.1 Production Environment

- **Platform:** NVIDIA Jetson Orin
- **Model:** MSD (672 KB) for speed
- **Fallback:** GTZAN for accuracy
- **Caching:** Feature cache for repeated files

### 9.2 Monitoring

- Classification accuracy
- Processing time per track
- GPU utilization
- Error rates

---

## 10. Future Enhancements

### 10.1 Planned Features

- [ ] Multi-label classification
- [ ] REST API deployment
- [ ] Web interface
- [ ] Real-time classification
- [ ] Mobile app support

### 10.2 Model Improvements

- [ ] Train on combined datasets (FMA + GTZAN + MSD)
- [ ] Implement progressive voting system
- [ ] Fine-tune OpenJMLA for genre classification
- [ ] Add confidence calibration

---

## 11. File Locations

### 11.1 Project Structure
```
/home/mijesu_970/Music_ReClass/          # Main project
/media/mijesu_970/SSD_Data/
  ├── AI_models/                          # Trained models
  ├── DataSets/                           # Training data
  ├── Musics_TBC/                         # Test music
  └── Music_Classified/                   # Output
```

### 11.2 Key Files
- Models: `/media/mijesu_970/SSD_Data/AI_models/`
- Datasets: `/media/mijesu_970/SSD_Data/DataSets/`
- Training scripts: `~/Music_ReClass/training/`
- Documentation: `~/Music_ReClass/docs/`

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-28 | Initial production release |
| 0.9 | 2025-11-26 | MSD + GTZAN models trained |
| 0.5 | 2025-11-24 | Dataset collection complete |

---

**Last Updated:** 2025-11-28  
**Next Review:** 2025-12-28
