# Music_ReClass - Project Summary

**Project Name:** Music_ReClass  
**Goal:** Automatic music genre classification using AI/Deep Learning  
**Platform:** NVIDIA Jetson (ARM64) + RTX GPU Support  
**Status:** ✅ Production Ready  
**Last Updated:** November 30, 2025

---

## 🎯 Key Achievements

- **Fast Training**: 77% accuracy in 2 minutes using FMA features
- **High Accuracy**: Up to 94% accuracy with FMA + MERT + JMLA progressive voting
- **Smart Early Stopping**: Average 20-40s processing (vs 50-100s full pipeline)
- **Progressive Voting**: Weighted ensemble for +2-3% accuracy boost
- **Production Ready**: Optimized inference pipeline with database integration

---

## 📊 Performance Overview

| Approach | Time | Accuracy | GPU Required | Model Size |
|----------|------|----------|--------------|------------|
| XGBoost | 10 min | 55-60% | No | <1 MB |
| **FMA Features** | **2 min** | **77%** | Yes | 672 KB |
| CNN Basic | 45 min | 70-80% | Yes | ~50 MB |
| Transfer Learning | 4 hrs | 80-90% | Yes | ~50 MB |
| **FMA + MERT + JMLA** | **8-12 hrs** | **85-94%** | Yes | ~100 MB |

### Progressive Voting Strategy

| Stage | Features | Time | Accuracy | Usage |
|-------|----------|------|----------|-------|
| 1. FMA only | 518 dims | 0s | 77% | 30% of songs |
| 2. FMA + MERT | 1286 dims | 30-60s | 82-88% | 50% of songs |
| 3. FMA + MERT + JMLA | 2054 dims | 50-100s | 85-94% | 20% of songs |

**Average Processing Time: 20-40s per track**

---

## 📁 Project Structure

```
Music_ReClass/
├── README.md                          # Main documentation
├── requirements_rtx.txt               # Python dependencies
├── setup_rtx.sh                       # RTX setup script
├── sync_and_push.sh                   # Git sync script
├── openjmla_parameters.json           # JMLA model config
│
├── extractors/                        # Feature extraction scripts
│   ├── extract_fma.py                # FMA features (database version)
│   ├── extract_fma_features.py       # FMA features (standalone)
│   ├── extract_mert.py               # MERT features (database)
│   ├── extract_mert_features.py      # MERT features (standalone)
│   ├── extract_jmla.py               # JMLA features (database)
│   ├── extract_jmla_features.py      # JMLA features (standalone)
│   ├── extract_jmla_simple.py        # JMLA text-based version
│   ├── extract_all_features.py       # Master orchestrator
│   ├── test_gtzan.py                 # Test GTZAN model
│   ├── test_msd.py                   # Test MSD model
│   ├── test_musicnn.py               # Test musicnn library
│   ├── compare_features.py           # Feature comparison
│   └── visualize_fma_features.py     # Feature visualization
│
├── training/                          # Training scripts
│   ├── train_msd.py                  # Fast feature-based (2 min, 77%)
│   ├── train_gtzan_v2.py             # Balanced CNN (45 min, 70-80%)
│   ├── train_gtzan_enhanced.py       # Best accuracy (4 hrs, 80-90%)
│   ├── train_fma_rtx.py              # Large-scale FMA training
│   ├── train_fma_progressive.py      # Progressive voting training
│   ├── train_mert_classifier.py      # MERT classifier
│   ├── train_jmla_classifier.py      # JMLA classifier
│   ├── train_xgboost_fma.py          # Traditional ML
│   ├── train_combined_4hr.py         # Combined training
│   └── compare_models.py             # Model comparison
│
├── classification/                    # Classification scripts
│   ├── classify_music_tbc.py         # Main classifier
│   ├── classify_with_jmla.py         # JMLA-based classifier
│   ├── classify_jmla_only.py         # JMLA only
│   ├── classify_two_phase.py         # Two-phase approach
│   ├── classify_and_tag.py           # Classify and tag files
│   ├── Reclass_FMJ_EV.py             # FMA+MERT+JMLA ensemble
│   └── Reclass_FMJ_Simple.py         # Simplified ensemble
│
├── analysis/                          # Analysis tools
│   ├── analyze_data.py               # Dataset visualization
│   ├── check_model.py                # Model inspection
│   ├── check_model_compatibility.py  # Compatibility check
│   └── extract_openjmla_params.py    # JMLA parameter extraction
│
├── utils/                             # Utility scripts
│   ├── gpu_monitor.py                # GPU memory tracking
│   ├── training_logger.py            # Training logs
│   ├── early_stopping.py             # Early stopping
│   ├── plex_sync.py                  # Plex integration
│   ├── config.py                     # Configuration
│   ├── download_tagtraum.py          # Dataset downloader
│   ├── convert_mtt_features.py       # MTT conversion
│   └── combine_mtt_*.py              # MTT utilities
│
├── features/                          # Extracted features (.npy files)
│   ├── FMA_features.npy              # 518-dim FMA features
│   ├── MERT_features.npy             # 768-dim MERT features
│   └── JMLA_features.npy             # 768-dim JMLA features
│
├── logs/                              # Training logs
│   ├── training.log                  # Main training log
│   ├── fma_base.log                  # FMA training log
│   ├── fma_base_metrics.csv          # FMA metrics
│   └── chat_history_*.json           # Chat histories
│
├── docs/                              # Documentation
│   ├── README.md                     # Documentation index
│   ├── SUMMARY.md                    # This file
│   ├── PROJECT_HISTORY.md            # Development timeline
│   ├── CLASSIFICATION_FEATURES.md    # Feature types guide
│   ├── EXTRACTION_RESULTS.md         # Extraction benchmarks
│   ├── Flowchart.md                  # Architecture diagrams
│   ├── REFERENCES.md                 # Academic references
│   ├── SIMILAR_PROJECTS.md           # Related projects
│   ├── ML_CONCEPTS_MEMO.md           # ML concepts
│   ├── guides/                       # Implementation guides
│   │   ├── Ensemble_Implementation_Guide.md
│   │   ├── PLEX_IMPLEMENTATION_OUTLINE.md
│   │   ├── FMA_FEATURE_EXTRACTION.md
│   │   ├── openjmla_readme.md
│   │   ├── RTX_TRAINING_CHECKLIST.md
│   │   └── M1_Porting_Guide.md
│   ├── technical/                    # Technical comparisons
│   │   ├── Feature_Extractors_Comparison.md
│   │   ├── MERT_vs_VGGish_Comparison.md
│   │   ├── Kaggle_vs_JMLA.md
│   │   └── [7 more files]
│   ├── archive/                      # Old versions
│   └── Reference/                    # Academic papers, notebooks
│
├── KeyFile/                           # Business documents
│   └── BusinessPlan
│
└── .git/                              # Git repository
```

---

## 🗂️ Datasets

### GTZAN Dataset ✅
- **Size**: 1,000 tracks (~1.2 GB)
- **Genres**: 10 (Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock)
- **Format**: WAV files, 30 seconds each
- **Use**: Baseline training and validation

### FMA Medium ✅
- **Size**: 25,000 tracks (~22 GB)
- **Genres**: 16 (Blues, Classical, Country, Easy Listening, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Jazz, Old-Time/Historic, Pop, Rock, Soul-RnB, Spoken)
- **Features**: 518 pre-computed features available
- **Use**: Large-scale training, better generalization

### Million Song Dataset (MSD) ✅
- **Size**: 10,000 H5 files (~2.6 GB)
- **Labels**: 133,676 genre annotations
- **Format**: HDF5 with pre-computed features
- **Use**: Feature-based training, fast prototyping

---

## 🤖 Models & Features

### 1. FMA Features (518 dimensions)
- **Type**: Hand-crafted audio features
- **Components**: MFCC (20), Chroma (12), Spectral (11), Rhythm (2), Statistics (467)
- **Training Time**: 2 minutes
- **Accuracy**: 77%
- **Best For**: Fast baseline, Stage 1 classification

### 2. MERT Features (768 dimensions)
- **Type**: Music-specific Transformer embeddings
- **Model**: MERT-v1-330M (330M parameters)
- **Training Time**: 4-6 hours
- **Accuracy**: 82-88% (with FMA)
- **Best For**: Stage 2 classification, music understanding

### 3. JMLA Features (768 dimensions)
- **Type**: Vision Transformer for audio
- **Model**: OpenJMLA (86M parameters)
- **Training Time**: 8-12 hours
- **Accuracy**: 85-94% (with FMA+MERT)
- **Best For**: Stage 3 classification, maximum accuracy

### 4. Trained Classifiers
- **MSD Model**: 672 KB, 77% accuracy, 16 genres
- **GTZAN Models**: ~50 MB, 70-90% accuracy, 10 genres
- **Ensemble Models**: ~100 MB, 85-94% accuracy, 16 genres

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/mijesu/Music_ReClass.git
cd Music_ReClass
pip install torch torchaudio librosa numpy matplotlib xgboost scikit-learn pandas h5py tqdm
```

### Fast Training (2 minutes)
```bash
python3 training/train_msd.py
# Result: 77% accuracy, 672 KB model
```

### Production Training (45 minutes)
```bash
python3 training/train_gtzan_v2.py
# Result: 70-80% accuracy, full metrics
```

### Best Accuracy (4 hours)
```bash
python3 training/train_gtzan_enhanced.py
# Result: 80-90% accuracy, best model
```

### Progressive Ensemble (8-12 hours)
```bash
python3 training/train_fma_progressive.py
# Result: 85-94% accuracy, progressive voting
```

### Feature Extraction
```bash
# Extract FMA features (standalone)
python3 extractors/extract_fma_features.py /path/to/audio/

# Extract all features (database version)
python3 extractors/extract_all_features.py
```

### Classification
```bash
# Classify music files
python3 classification/classify_music_tbc.py --input /path/to/music

# Ensemble classification
python3 classification/Reclass_FMJ_EV.py
```

---

## 🔧 Technical Stack

### Hardware
- **Primary**: NVIDIA Jetson (ARM64 with CUDA)
- **Secondary**: RTX 4060 Ti 16GB or similar
- **Storage**: 50+ GB SSD recommended

### Software
- **OS**: Linux (Ubuntu 22.04)
- **Python**: 3.10.12
- **CUDA**: 12.1+

### Key Libraries
```
torch==2.8.0 (with CUDA)
torchaudio==2.8.0
librosa==0.11.0
numpy==1.26.4
matplotlib==3.5.1
xgboost==3.1.2
scikit-learn
pandas
h5py
tqdm
transformers
```

---

## 💡 Key Insights

### 1. Progressive Voting Strategy
- Stage 1 (FMA): Fast classification for 30% of songs with high confidence
- Stage 2 (FMA+MERT): Medium processing for 50% of songs
- Stage 3 (FMA+MERT+JMLA): Full processing for 20% difficult songs
- **Result**: 20-40s average vs 50-100s full pipeline

### 2. Feature-Based Training is Faster
- MSD approach: 2 minutes for 77% accuracy
- Audio approach: 30-45 minutes for 70-80% accuracy
- Trade-off: Less flexible but highly efficient

### 3. File Format Optimization
- CSV: 951 MB, slow loading (30-60 seconds)
- NPY: 211 MB, fast loading (1-2 seconds)
- **Result**: 4.5x smaller, 20-30x faster

### 4. Transfer Learning Works Best
- From scratch: 60-70% accuracy
- With OpenJMLA: 80-90% accuracy
- Requires fewer training samples

### 5. Ensemble Methods
- Single model: 70-80% accuracy
- Ensemble: 85-90% accuracy
- Progressive voting: 85-94% accuracy
- **Best approach**: Weighted voting with early stopping

---

## 📈 Training Approaches

### Approach 1: Quick Baseline (5 minutes)
```bash
python3 training/quick_baseline.py
```
- **Accuracy**: 50-55%
- **Use**: Fast testing and validation

### Approach 2: Feature-Based (2 minutes) ⭐ RECOMMENDED
```bash
python3 training/train_msd.py
```
- **Accuracy**: 77%
- **Use**: Production deployment

### Approach 3: CNN Basic (45 minutes)
```bash
python3 training/train_gtzan_v2.py
```
- **Accuracy**: 70-80%
- **Use**: Balanced speed and accuracy

### Approach 4: Transfer Learning (4 hours) ⭐ BEST ACCURACY
```bash
python3 training/train_gtzan_enhanced.py
```
- **Accuracy**: 80-90%
- **Use**: Maximum accuracy requirements

### Approach 5: Progressive Ensemble (8-12 hours) ⭐ PRODUCTION
```bash
python3 training/train_fma_progressive.py
```
- **Accuracy**: 85-94%
- **Use**: Production with early stopping

---

## 🎵 Supported Genres

### GTZAN (10 genres)
Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

### FMA (16 genres)
Blues, Classical, Country, Easy Listening, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Jazz, Old-Time/Historic, Pop, Rock, Soul-RnB, Spoken

### MSD (13 genres)
Blues, Country, Electronic, Folk, International, Jazz, Latin, New Age, Pop_Rock, Rap, Reggae, RnB, Vocal

---

## ✅ Completed Features

- [x] Multiple training approaches (8 methods)
- [x] Feature extraction (FMA, MERT, JMLA)
- [x] Progressive voting ensemble
- [x] Early stopping strategy
- [x] Database integration (SQLite)
- [x] GPU optimization (Jetson + RTX)
- [x] Plex integration support
- [x] Comprehensive documentation
- [x] 25+ scripts organized
- [x] Test and analysis tools

---

## 🔄 In Progress

- [ ] FMA large-scale training (25K tracks)
- [ ] Music_TBC classification
- [ ] Multi-label classification
- [ ] Real-time classification

---

## 📋 Planned Features

- [ ] REST API deployment
- [ ] Web interface
- [ ] Mobile app support
- [ ] Streaming service integration
- [ ] Custom genre training
- [ ] Audio similarity search

---

## 📚 Documentation

- **[README.md](../README.md)** - Main project documentation
- **[docs/README.md](README.md)** - Documentation index
- **[PROJECT_HISTORY.md](PROJECT_HISTORY.md)** - Development timeline
- **[guides/](guides/)** - Implementation guides
- **[technical/](technical/)** - Technical comparisons
- **[Reference/](Reference/)** - Academic papers and notebooks

---

## 🙏 Acknowledgments

- **OpenJMLA Team** - Pre-trained Vision Transformer model
- **GTZAN Dataset** - Genre classification benchmark
- **FMA** - Free Music Archive dataset and features
- **Million Song Dataset** - Large-scale music features
- **PyTorch Team** - Deep learning framework
- **librosa** - Audio processing library

---

## 📞 Contact

- **GitHub**: [@mijesu](https://github.com/mijesu)
- **Project**: [Music_ReClass](https://github.com/mijesu/Music_ReClass)

---

## 📊 Project Statistics

**Scripts**: 25+ organized scripts
- Extractors: 13 scripts
- Training: 10 scripts
- Classification: 7 scripts
- Analysis: 4 scripts
- Utils: 8 scripts

**Documentation**: 25+ files
- Core docs: 8 files
- Guides: 6 files
- Technical: 7 files
- Archive: 6 files

**Models**: 4 trained models
- MSD: 672 KB, 77%
- GTZAN: ~50 MB, 70-90%
- Ensemble: ~100 MB, 85-94%

**Features**: 2054 total dimensions
- FMA: 518 dims
- MERT: 768 dims
- JMLA: 768 dims

---

**Version**: 2.0  
**Status**: ✅ Production Ready  
**Last Updated**: November 30, 2025  
**Next Milestone**: Music_TBC classification with progressive voting

---

*For detailed information, see individual documentation files in the docs/ folder.*
