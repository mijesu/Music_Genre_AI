# Scripts Organization

## Directory Structure

```
scripts/
├── training/          # Model training scripts
├── analysis/          # Data analysis and model inspection
├── utils/            # Utility functions and helpers
└── examples/         # Example code and templates
```

## Training Scripts (`training/`)

### Quick Start
- **quick_baseline.py** - Fast XGBoost baseline (2-5 min, no GPU)
- **compare_models.py** - Compare XGBoost vs Deep Learning

### Deep Learning
- **train_gtzan_openjmla.py** - Transfer learning with GPU monitoring (RECOMMENDED)
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

### Get Quick Baseline
```bash
python3 scripts/training/quick_baseline.py
```

### Analyze Datasets
```bash
python3 scripts/analysis/analyze_data.py
```

### Train with Deep Learning
```bash
python3 scripts/training/train_gtzan_openjmla.py
```

### Compare Approaches
```bash
python3 scripts/training/compare_models.py
```

### Monitor GPU
```bash
python3 scripts/utils/gpu_monitor.py
```
