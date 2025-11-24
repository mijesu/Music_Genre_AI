# Music_ReClass

Reclassify music files using AI models (JMLA, GTZAN, FMA datasets).

## Project Overview

Music genre classification system using:
- **OpenJMLA** pretrained model (Vision Transformer for audio)
- **GTZAN dataset** (10 genres, ~1000 audio files)
- **FMA Medium dataset** (25,000 tracks, 16 genres)

## Directory Structure

```
Music_Reclass/
├── training/          # Model training scripts
├── analysis/          # Data analysis and model inspection
├── utils/            # Utility functions and helpers
├── examples/         # Example code and templates
├── classify_music_tbc.py      # Classify Music_TBC folder
└── classify_with_jmla.py      # Classify using JMLA clustering
```

## Training Scripts (`training/`)

### Quick Start
- **quick_baseline.py** - Fast XGBoost baseline (2-5 min, no GPU)
- **train_gtzan_v2.py** ⭐ - Production-ready with OpenJMLA (RECOMMENDED)
- **compare_models.py** - Compare XGBoost vs Deep Learning

### Deep Learning
- **train_gtzan_openjmla.py** - Transfer learning with GPU monitoring
- **train_with_openjmla.py** - Original transfer learning script
- **Classifed_JMLA_GTZAN.py** - Enhanced training with memory management

### Traditional ML
- **train_xgboost_fma.py** - XGBoost with FMA pre-computed features

## Analysis Scripts (`analysis/`)

- **analyze_data.py** - Dataset analysis, genre distribution, mel-spectrograms
- **check_model.py** - Basic model structure check
- **check_model_compatibility.py** - Verify model compatibility
- **extract_openjmla_params.py** - Extract OpenJMLA parameters

## Utilities (`utils/`)

- **gpu_monitor.py** - GPU memory monitoring and batch size suggestions

## Examples (`examples/`)

- **load_jmla_model.py** - Model loading example
- **music_genre_classifier.py** - Basic classifier template
- **pytorch_example.py** - PyTorch example

## Usage

### Quick Baseline
```bash
python3 training/quick_baseline.py
```

### Analyze Datasets
```bash
python3 analysis/analyze_data.py
```

### Train Production Model (Recommended)
```bash
python3 training/train_gtzan_v2.py
```

### Classify Music Files
```bash
python3 classify_music_tbc.py
# or
python3 classify_with_jmla.py
```

### Compare Approaches
```bash
python3 training/compare_models.py
```

### Monitor GPU
```bash
python3 utils/gpu_monitor.py
```

## Requirements

- Python 3.10+
- PyTorch 2.8.0 with CUDA
- librosa, torchaudio, matplotlib
- xgboost (for baseline)

## Project Documentation

Full documentation available at: `/media/mijesu_970/SSD_Data/Kiro_Projects/Music_Reclass/`
