# M1 MacBook Pro Porting Guide

## Project Migration Plan (2026)

This document outlines the steps and considerations for porting the Music Reclassification project from Linux (current) to MacBook Pro M1 (Apple Silicon).

---

## System Requirements

### Hardware
- **Device**: MacBook Pro M1/M2/M3
- **RAM**: 16 GB minimum (32 GB recommended for large models)
- **Storage**: 256 GB minimum (512 GB recommended)
  - Datasets: ~35 GB
  - Models: ~4 GB
  - Project files: ~1 GB
  - **Total needed**: ~40 GB + workspace

### Software
- **OS**: macOS 12.0+ (Monterey or later)
- **Python**: 3.9-3.11 (ARM64 native)
- **PyTorch**: 2.0+ with MPS (Metal Performance Shaders) support

---

## Installation Steps

### 1. Python Environment Setup

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python (ARM64 native)
brew install python@3.10

# Create virtual environment
python3.10 -m venv ~/venv/music_reclass
source ~/venv/music_reclass/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. PyTorch Installation (M1 Optimized)

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 3. Core Dependencies

```bash
# Audio processing libraries
pip install librosa soundfile audioread

# Deep learning frameworks
pip install transformers huggingface-hub

# Data processing
pip install numpy pandas scipy scikit-learn

# Visualization
pip install matplotlib seaborn

# Feature extractors
pip install torchvggish musicnn essentia opensmile

# Jupyter for notebooks
pip install jupyter ipykernel
```

### 4. Model-Specific Libraries

```bash
# For CLAP
pip install laion-clap

# For audio processing
pip install resampy pydub

# For dataset handling
pip install h5py tables
```

---

## Code Modifications for M1

### Device Selection

**Current (Linux with CUDA):**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Updated (M1 with MPS):**
```python
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")
```

### Model Loading

**Add device mapping for all models:**
```python
# Example for MERT
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "m-a-p/MERT-v1-330M",
    trust_remote_code=True
)
model = model.to(device)
model.eval()
```

### Data Loading Optimization

**Use M1-optimized data loaders:**
```python
from torch.utils.data import DataLoader

# Adjust num_workers for M1 (typically 4-8)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # M1 has 8 performance cores
    pin_memory=False  # Not needed for MPS
)
```

### Mixed Precision Training

**M1 supports float16 on MPS:**
```python
from torch.cuda.amp import autocast, GradScaler

# For M1, use torch.autocast with 'mps' device
with torch.autocast(device_type='mps', dtype=torch.float16):
    outputs = model(inputs)
```

---

## Model Compatibility

### Fully Compatible (No Changes Needed)
✅ **VGGish** - Works natively on M1  
✅ **MERT-v1-330M** - Hugging Face transformers supports M1  
✅ **CLAP** - Compatible with MPS backend  
✅ **EnCodec** - Meta models support Apple Silicon  
✅ **AST** - Transformers-based, fully compatible  
✅ **HuBERT** - Facebook models work on M1  
✅ **PANNs** - PyTorch CNN, no issues  

### Library-Based Extractors
✅ **Musicnn** - Has ARM64 wheels  
✅ **Essentia** - Available via pip for M1  
✅ **librosa** - Fully compatible  
✅ **openSMILE** - Python bindings work on M1  

---

## Data Transfer Plan

### Option 1: External Drive Transfer
```bash
# On Linux system
rsync -avh --progress /media/mijesu_970/SSD_Data/DataSets /Volumes/ExternalDrive/
rsync -avh --progress /media/mijesu_970/SSD_Data/AI_models /Volumes/ExternalDrive/
rsync -avh --progress /media/mijesu_970/SSD_Data/Kiro_Projects /Volumes/ExternalDrive/

# On M1 Mac
rsync -avh --progress /Volumes/ExternalDrive/DataSets ~/Music_Reclass/DataSets
rsync -avh --progress /Volumes/ExternalDrive/AI_models ~/Music_Reclass/AI_models
rsync -avh --progress /Volumes/ExternalDrive/Kiro_Projects ~/Music_Reclass/Projects
```

### Option 2: Network Transfer
```bash
# On Linux (start server)
cd /media/mijesu_970/SSD_Data
python -m http.server 8000

# On M1 Mac (download)
wget -r -np -nH --cut-dirs=1 http://<linux-ip>:8000/
```

### Option 3: Cloud Storage
- Upload to AWS S3, Google Drive, or Dropbox
- Download on M1 Mac

---

## Performance Expectations

### Inference Speed (Relative to Linux CPU)
- **M1 CPU**: 2-3x faster (ARM efficiency cores)
- **M1 GPU (MPS)**: 5-10x faster for large models
- **Memory bandwidth**: 200-400 GB/s (unified memory advantage)

### Training Speed
- **Small models** (VGGish, EnCodec): 3-5x faster than CPU
- **Large models** (MERT, HuBERT): 2-4x faster than CPU
- **Comparable to**: NVIDIA GTX 1660 Ti / RTX 2060

### Memory Usage
- **Unified memory**: Shared between CPU and GPU
- **16 GB RAM**: Sufficient for inference + small batch training
- **32 GB RAM**: Recommended for large batch training

---

## Known Issues and Workarounds

### Issue 1: MPS Backend Limitations
**Problem**: Some PyTorch operations not yet supported on MPS  
**Workaround**: Fall back to CPU for unsupported ops
```python
try:
    output = model(input.to('mps'))
except RuntimeError:
    output = model(input.to('cpu'))
```

### Issue 2: Hugging Face Cache Location
**Problem**: Default cache may fill up system drive  
**Solution**: Set custom cache directory
```bash
export HF_HOME=~/Music_Reclass/hf_cache
export TRANSFORMERS_CACHE=~/Music_Reclass/hf_cache
```

### Issue 3: Audio Library Dependencies
**Problem**: Some audio codecs may need additional installation  
**Solution**: Install ffmpeg via Homebrew
```bash
brew install ffmpeg
```

### Issue 4: Multiprocessing on macOS
**Problem**: Different multiprocessing behavior than Linux  
**Solution**: Use 'spawn' method explicitly
```python
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

---

## Directory Structure on M1

```
~/Music_Reclass/
├── DataSets/
│   ├── GTZAN/
│   ├── FMA/
│   ├── MSD/
│   └── MagnaTagATune/
├── AI_models/
│   ├── VGGish/
│   ├── MERT/
│   ├── CLAP/
│   ├── EnCodec/
│   ├── AST/
│   ├── HuBERT/
│   └── PANNs/
├── Projects/
│   └── Music_Reclass/
│       ├── training/
│       ├── classification/
│       ├── utils/
│       ├── examples/
│       └── docs/
└── hf_cache/  # Hugging Face cache
```

---

## Testing Checklist

After porting, verify:

- [ ] PyTorch MPS backend available
- [ ] All 7 models load successfully
- [ ] Feature extraction works for each model
- [ ] Dataset loading (GTZAN, FMA, MSD, MTT)
- [ ] Training script runs without errors
- [ ] Inference speed acceptable
- [ ] Memory usage within limits
- [ ] Jupyter notebooks functional
- [ ] Audio playback works
- [ ] Visualization tools working

---

## Performance Optimization Tips

### 1. Batch Size Tuning
```python
# Start with smaller batches, increase until memory limit
batch_sizes = {
    'VGGish': 64,
    'MERT': 16,
    'CLAP': 32,
    'EnCodec': 32,
    'AST': 16,
    'HuBERT': 8,
    'PANNs': 32
}
```

### 2. Enable Metal Performance Shaders
```python
# Ensure MPS is used for all operations
torch.set_default_device('mps')
```

### 3. Optimize Data Loading
```python
# Use memory mapping for large datasets
import numpy as np
data = np.load('features.npy', mmap_mode='r')
```

### 4. Profile Code
```python
# Use PyTorch profiler to identify bottlenecks
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
    model(input)
print(prof.key_averages().table())
```

---

## Backup Strategy

Before migration:
1. **Git repository**: Push all code changes
2. **Model checkpoints**: Backup trained models
3. **Processed datasets**: Keep preprocessed features
4. **Documentation**: Ensure all docs are up-to-date
5. **Environment**: Export requirements.txt
   ```bash
   pip freeze > requirements.txt
   ```

---

## Additional Resources

### Official Documentation
- [PyTorch on Apple Silicon](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
- [Hugging Face on M1](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)

### Community Resources
- [M1 ML Performance Benchmarks](https://github.com/tlkh/asitop)
- [PyTorch MPS Examples](https://github.com/pytorch/pytorch/tree/master/test/test_mps.py)

---

## Timeline Estimate

| Task | Estimated Time |
|------|----------------|
| Environment setup | 1-2 hours |
| Data transfer | 2-4 hours (depends on method) |
| Code modifications | 2-3 hours |
| Testing and debugging | 3-5 hours |
| Performance tuning | 2-3 hours |
| **Total** | **10-17 hours** |

---

## Notes

- M1 Max/Pro variants have more GPU cores (better performance)
- M2/M3 chips offer incremental improvements
- Consider external SSD for datasets if internal storage limited
- Time Machine backup recommended before migration
- Keep Linux system as backup until M1 setup verified

**Migration Target**: Q1 2026  
**Created**: 2025-11-25  
**Last Updated**: 2025-11-25
