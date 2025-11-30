# Music_ReClass 🎵

Automatic music genre classification using AI/Deep Learning on NVIDIA Jetson and RTX platforms.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Key Features

- **Fast Training**: 77% accuracy in just 2 minutes using FMA features
- **High Accuracy**: Up to 94% accuracy with FMA + MERT + JMLA progressive voting
- **Smart Early Stopping**: Average 20-40s processing (vs 50-100s)
- **Progressive Voting**: Weighted ensemble for +2-3% accuracy boost
- **Multiple Approaches**: Hand-crafted features, Transformers, and Ensemble methods
- **GPU Optimized**: Runs on NVIDIA Jetson (ARM64) and RTX GPUs
- **Production Ready**: Early stopping + voting strategy for efficient inference

## 📊 Performance

| Approach | Time | Accuracy | GPU Required | Model Size |
|----------|------|----------|--------------|------------|
| XGBoost | 10 min | 55-60% | No | <1 MB |
| **FMA Features** | **2 min** | **77%** | Yes | 672 KB |
| CNN Basic | 45 min | 70-80% | Yes | ~50 MB |
| Transfer Learning | 4 hrs | 80-90% | Yes | ~50 MB |
| **FMA + MERT + JMLA** | **8-12 hrs** | **85-92%** | Yes | ~100 MB |

### Early Stopping Strategy (Production)

| Stage | Features | Time | Accuracy | Songs |
|-------|----------|------|----------|-------|
| 1. FMA only | 518 dims | 0s | 77% | 30% |
| 2. FMA + MERT | 1286 dims | 30-60s | 82-88% | 50% |
| 3. FMA + MERT + JMLA | 2054 dims | 50-100s | 85-92% | 20% |

**Average Processing Time: 20-40s per track**

## 🗂️ Datasets

### GTZAN Dataset
- **Size**: 1,000 tracks (~1.2 GB)
- **Genres**: 10 (Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock)
- **Format**: WAV files, 30 seconds each
- **Use**: Baseline training and validation

### FMA Medium
- **Size**: 25,000 tracks (~22 GB)
- **Genres**: 16 (Blues, Classical, Country, Easy Listening, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Jazz, Old-Time/Historic, Pop, Rock, Soul-RnB, Spoken)
- **Features**: 518 pre-computed features available
- **Use**: Large-scale training, better generalization

### Million Song Dataset (MSD)
- **Size**: 10,000 H5 files (~2.6 GB)
- **Labels**: 133,676 genre annotations
- **Format**: HDF5 with pre-computed features
- **Use**: Feature-based training, fast prototyping

## 🤖 Models

### 1. MSD Model (Feature-Based) ✅
- **Accuracy**: 77.09%
- **Training Time**: 2 minutes
- **Size**: 672 KB
- **Genres**: 16
- **Best For**: Quick testing and production deployment

### 2. GTZAN CNN Models ✅
- **Accuracy**: 70-90%
- **Training Time**: 45 min - 4 hours
- **Genres**: 10
- **Best For**: High accuracy requirements

### 3. OpenJMLA (Pre-trained) ✅
- **Type**: Vision Transformer
- **Size**: 1.3 GB
- **Parameters**: 86 million
- **Best For**: Transfer learning and feature extraction

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mijesu/Music_ReClass.git
cd Music_ReClass

# Install dependencies
pip install torch torchaudio librosa numpy matplotlib xgboost scikit-learn pandas h5py tqdm
```

### Quick Training (2 minutes)

```bash
python3 train_msd.py
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

### Classify Music

```bash
python3 classify_music_tbc.py --input /path/to/music --model msd_model.pth
```

## 📁 Project Structure

```
Music_ReClass/
├── training/                      # Training scripts
│   ├── train_msd.py              # Fast feature-based (2 min, 77%)
│   ├── train_gtzan_v2.py         # Balanced CNN (45 min, 70-80%)
│   ├── train_gtzan_enhanced.py   # Best accuracy (4 hrs, 80-90%)
│   ├── train_fma_rtx.py          # Large-scale FMA training
│   ├── train_xgboost_fma.py      # Traditional ML approach
│   └── compare_models.py         # Model comparison tool
│
├── analysis/                      # Analysis tools
│   ├── analyze_data.py           # Dataset visualization
│   ├── check_model.py            # Model inspection
│   └── feature_analysis.py       # Feature importance
│
├── utils/                         # Utilities
│   ├── gpu_monitor.py            # GPU memory tracking
│   ├── training_logger.py        # Training logs
│   └── early_stopping.py         # Early stopping
│
├── examples/                      # Example scripts
│   └── quick_demo.py             # Quick demonstration
│
├── models/                        # Trained models
│   ├── msd_model.pth             # 77% accuracy (672 KB)
│   └── best_model.pth            # Best trained model
│
├── docs/                          # Documentation
│   ├── SUMMARY.md                # Complete project summary
│   ├── PROJECT_HISTORY.md        # Development history
│   ├── CLASSIFICATION_FEATURES.md # Feature types guide
│   └── RTX_TRAINING_CHECKLIST.md # RTX setup guide
│
└── README.md                      # This file
```

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
```

## 💡 Key Insights

### 1. Feature-Based Training is Much Faster
- MSD approach: 2 minutes for 77% accuracy
- Audio approach: 30-45 minutes for 70-80% accuracy
- Trade-off: Less flexible but highly efficient

### 2. File Format Optimization
- CSV: 951 MB, slow loading (30-60 seconds)
- NPY: 211 MB, fast loading (1-2 seconds)
- **Result**: 4.5x smaller, 20-30x faster

### 3. Transfer Learning Works Best
- From scratch: 60-70% accuracy
- With OpenJMLA: 80-90% accuracy
- Requires fewer training samples

### 4. Data Augmentation is Critical
- Without augmentation: 70-75% accuracy
- With augmentation: 80-90% accuracy
- Essential for small datasets

## 📈 Training Approaches

### Approach 1: Quick Baseline (5 minutes)
```bash
python3 training/quick_baseline.py
```
- **Accuracy**: 50-55%
- **Use**: Fast testing and validation

### Approach 2: Feature-Based (2 minutes) ⭐ RECOMMENDED
```bash
python3 train_msd.py
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

### Approach 5: Ensemble (8-12 hours)
```bash
python3 training/train_ensemble.py
```
- **Accuracy**: 85-90%
- **Use**: State-of-the-art results

## 🎵 Supported Genres

### GTZAN (10 genres)
Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

### FMA (16 genres)
Blues, Classical, Country, Easy Listening, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Jazz, Old-Time/Historic, Pop, Rock, Soul-RnB, Spoken

### MSD (13 genres)
Blues, Country, Electronic, Folk, International, Jazz, Latin, New Age, Pop_Rock, Rap, Reggae, RnB, Vocal

## 📊 Results

### MSD Model Performance
```
Training: 17,000 FMA tracks
Epochs: 7
Time: 2 minutes
Validation Accuracy: 77.09%
Model Size: 672 KB

Top Performing Genres:
- Blues: 85%
- Classical: 90%
- Jazz: 82%
```

### GTZAN CNN Performance
```
Dataset: 1,000 tracks (10 genres)
Training Time: 45 minutes - 4 hours
Accuracy: 70-90%

Common Confusion Pairs:
- Rock ↔ Blues
- Electronic ↔ Hip-Hop
- Metal ↔ Rock
```

## 🔄 Workflow

1. **Data Preparation**
   - Download datasets (GTZAN, FMA, or MSD)
   - Extract features or use pre-computed features
   - Split into train/validation/test sets

2. **Model Training**
   - Choose approach based on time/accuracy requirements
   - Monitor GPU memory usage
   - Save best model checkpoints

3. **Evaluation**
   - Test on validation set
   - Generate confusion matrix
   - Analyze per-genre performance

4. **Deployment**
   - Load trained model
   - Classify new music files
   - Generate classification reports

## 📋 Roadmap

### Completed ✅
- [x] Multiple training approaches implemented
- [x] 3 models trained (MSD, GTZAN, ZTGAN)
- [x] 4 datasets organized
- [x] GPU optimization for Jetson
- [x] RTX PC support
- [x] Comprehensive documentation

### In Progress 🔄
- [ ] FMA large-scale training
- [ ] Ensemble model development
- [ ] Music_TBC classification

### Planned 📋
- [ ] Multi-label classification
- [ ] REST API deployment
- [ ] Web interface
- [ ] Real-time classification
- [ ] Mobile app support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Documentation

- [Complete Summary](docs/SUMMARY.md) - Comprehensive project overview
- [Project History](docs/PROJECT_HISTORY.md) - Development timeline
- [Classification Features](docs/CLASSIFICATION_FEATURES.md) - Feature types guide
- [RTX Training Guide](docs/RTX_TRAINING_CHECKLIST.md) - RTX setup instructions

## 🙏 Acknowledgments

- **OpenJMLA Team** - Pre-trained Vision Transformer model
- **GTZAN Dataset** - Genre classification benchmark
- **FMA** - Free Music Archive dataset and features
- **Million Song Dataset** - Large-scale music features
- **PyTorch Team** - Deep learning framework
- **librosa** - Audio processing library

## 📞 Contact

- **GitHub**: [@mijesu](https://github.com/mijesu)
- **Project Link**: [https://github.com/mijesu/Music_ReClass](https://github.com/mijesu/Music_ReClass)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated**: November 26, 2025  
**Version**: 1.0  
**Status**: ✅ Production Ready
