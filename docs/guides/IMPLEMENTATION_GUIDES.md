# Ensemble Implementation: MERT + CLAP + PANNs (Spectrogram)

## Overview

Combine three feature extractors to achieve 75-90% accuracy:
- **MERT**: 768-dim (music understanding)
- **CLAP**: 512-dim (audio-text alignment)
- **PANNs**: 2048-dim (audio tagging)
- **Total**: 3328-dim combined features

---

## Method 1: Feature Concatenation (Recommended)

### Step 1: Extract Features from All Models

```python
import torch
import librosa
import numpy as np
from transformers import AutoModel, Wav2Vec2FeatureExtractor, ClapModel, ClapProcessor

# Load all models
mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M")

clap_model = ClapModel.from_pretrained("laion/larger_clap_music")
clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

panns_model = AutoModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
panns_processor = Wav2Vec2FeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Set to eval mode
mert_model.eval()
clap_model.eval()
panns_model.eval()

def extract_ensemble_features(audio_path):
    """Extract features from all three models"""
    
    # Load audio
    audio_mert, _ = librosa.load(audio_path, sr=24000)
    audio_clap, _ = librosa.load(audio_path, sr=48000)
    audio_panns, _ = librosa.load(audio_path, sr=16000)
    
    with torch.no_grad():
        # MERT features (768-dim)
        inputs_mert = mert_processor(audio_mert, sampling_rate=24000, return_tensors="pt")
        outputs_mert = mert_model(**inputs_mert)
        mert_features = outputs_mert.last_hidden_state.mean(dim=1).numpy()[0]
        
        # CLAP features (512-dim)
        inputs_clap = clap_processor(audios=audio_clap, sampling_rate=48000, return_tensors="pt")
        clap_features = clap_model.get_audio_features(**inputs_clap).numpy()[0]
        
        # PANNs features (2048-dim)
        inputs_panns = panns_processor(audio_panns, sampling_rate=16000, return_tensors="pt")
        outputs_panns = panns_model(**inputs_panns)
        panns_features = outputs_panns.last_hidden_state.mean(dim=1).numpy()[0]
    
    # Concatenate all features
    ensemble_features = np.concatenate([mert_features, clap_features, panns_features])
    
    return ensemble_features  # 3328-dim
```

### Step 2: Extract Features for Training Data

```python
from pathlib import Path
import pickle

# Extract features from GTZAN dataset
gtzan_path = "/media/mijesu_970/SSD_Data/DataSets/GTZAN/genres_original"
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

X_train = []
y_train = []

for genre in genres:
    genre_path = Path(gtzan_path) / genre
    for audio_file in genre_path.glob("*.wav"):
        print(f"Processing: {audio_file.name}")
        
        # Extract ensemble features
        features = extract_ensemble_features(str(audio_file))
        
        X_train.append(features)
        y_train.append(genre)

# Save extracted features
with open('ensemble_features.pkl', 'wb') as f:
    pickle.dump({'X': X_train, 'y': y_train}, f)
```

### Step 3: Train Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load features
with open('ensemble_features.pkl', 'rb') as f:
    data = pickle.load(f)
    X = np.array(data['X'])
    y = np.array(data['y'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(clf, 'ensemble_classifier.pkl')
```

---

## Method 2: Weighted Voting

### Step 1: Train Individual Classifiers

```python
# Train separate classifiers for each model
clf_mert = RandomForestClassifier(n_estimators=100)
clf_clap = RandomForestClassifier(n_estimators=100)
clf_panns = RandomForestClassifier(n_estimators=100)

clf_mert.fit(X_train_mert, y_train)
clf_clap.fit(X_train_clap, y_train)
clf_panns.fit(X_train_panns, y_train)
```

### Step 2: Weighted Voting

```python
def ensemble_predict_voting(audio_path, weights=[0.5, 0.3, 0.2]):
    """
    Weighted voting ensemble
    weights: [mert_weight, clap_weight, panns_weight]
    """
    # Extract features
    mert_feat = extract_mert(audio_path)
    clap_feat = extract_clap(audio_path)
    panns_feat = extract_panns(audio_path)
    
    # Get probability predictions
    prob_mert = clf_mert.predict_proba([mert_feat])[0]
    prob_clap = clf_clap.predict_proba([clap_feat])[0]
    prob_panns = clf_panns.predict_proba([panns_feat])[0]
    
    # Weighted average
    prob_ensemble = (
        weights[0] * prob_mert +
        weights[1] * prob_clap +
        weights[2] * prob_panns
    )
    
    # Get final prediction
    predicted_class = clf_mert.classes_[prob_ensemble.argmax()]
    confidence = prob_ensemble.max()
    
    return predicted_class, confidence
```

---

## Method 3: Stacking (Advanced)

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define base estimators
estimators = [
    ('mert', RandomForestClassifier(n_estimators=100)),
    ('clap', RandomForestClassifier(n_estimators=100)),
    ('panns', RandomForestClassifier(n_estimators=100))
]

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# Prepare data (each model's features separately)
X_stacking = np.hstack([X_mert, X_clap, X_panns])

# Train
stacking_clf.fit(X_stacking, y_train)

# Predict
y_pred = stacking_clf.predict(X_test)
```

---

## Complete Implementation Script

```python
#!/usr/bin/env python3
"""
Ensemble Feature Extraction and Classification
MERT + CLAP + PANNs
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import AutoModel, Wav2Vec2FeatureExtractor, ClapModel, ClapProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pickle

class EnsembleExtractor:
    def __init__(self):
        print("Loading models...")
        
        # MERT
        self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M")
        self.mert_model.eval()
        
        # CLAP
        self.clap_model = ClapModel.from_pretrained("laion/larger_clap_music")
        self.clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
        self.clap_model.eval()
        
        # PANNs (using AST as proxy)
        self.panns_model = AutoModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.panns_processor = Wav2Vec2FeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.panns_model.eval()
        
        print("Models loaded!")
    
    def extract(self, audio_path):
        """Extract ensemble features from audio file"""
        
        # Load audio at different sample rates
        audio_mert, _ = librosa.load(audio_path, sr=24000, duration=30)
        audio_clap, _ = librosa.load(audio_path, sr=48000, duration=30)
        audio_panns, _ = librosa.load(audio_path, sr=16000, duration=30)
        
        with torch.no_grad():
            # MERT
            inputs = self.mert_processor(audio_mert, sampling_rate=24000, return_tensors="pt")
            outputs = self.mert_model(**inputs)
            mert_feat = outputs.last_hidden_state.mean(dim=1).numpy()[0]
            
            # CLAP
            inputs = self.clap_processor(audios=audio_clap, sampling_rate=48000, return_tensors="pt")
            clap_feat = self.clap_model.get_audio_features(**inputs).numpy()[0]
            
            # PANNs
            inputs = self.panns_processor(audio_panns, sampling_rate=16000, return_tensors="pt")
            outputs = self.panns_model(**inputs)
            panns_feat = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
        # Concatenate
        ensemble_feat = np.concatenate([mert_feat, clap_feat, panns_feat])
        
        return ensemble_feat

def extract_dataset_features(dataset_path, output_file):
    """Extract features from entire dataset"""
    
    extractor = EnsembleExtractor()
    
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    X = []
    y = []
    
    for genre in genres:
        genre_path = Path(dataset_path) / genre
        for audio_file in genre_path.glob("*.wav"):
            print(f"Processing: {genre}/{audio_file.name}")
            
            try:
                features = extractor.extract(str(audio_file))
                X.append(features)
                y.append(genre)
            except Exception as e:
                print(f"Error: {e}")
    
    # Save
    with open(output_file, 'wb') as f:
        pickle.dump({'X': np.array(X), 'y': np.array(y)}, f)
    
    print(f"Saved {len(X)} samples to {output_file}")

def train_classifier(features_file, model_file):
    """Train classifier on extracted features"""
    
    # Load features
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        y = data['y']
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    print("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save
    joblib.dump(clf, model_file)
    print(f"Model saved to {model_file}")

def predict(audio_path, model_file):
    """Predict genre for new audio file"""
    
    # Load model
    clf = joblib.load(model_file)
    
    # Extract features
    extractor = EnsembleExtractor()
    features = extractor.extract(audio_path)
    
    # Predict
    prediction = clf.predict([features])[0]
    probabilities = clf.predict_proba([features])[0]
    
    # Get top 3
    top_indices = probabilities.argsort()[-3:][::-1]
    top_genres = [(clf.classes_[i], probabilities[i]) for i in top_indices]
    
    return prediction, top_genres

if __name__ == "__main__":
    # Step 1: Extract features
    extract_dataset_features(
        "/media/mijesu_970/SSD_Data/DataSets/GTZAN/genres_original",
        "ensemble_features.pkl"
    )
    
    # Step 2: Train classifier
    train_classifier("ensemble_features.pkl", "ensemble_classifier.pkl")
    
    # Step 3: Test prediction
    prediction, top_3 = predict(
        "/media/mijesu_970/SSD_Data/Musics_TBC/A-Lin - 安寧.wav",
        "ensemble_classifier.pkl"
    )
    
    print(f"\nPrediction: {prediction}")
    print("Top 3:")
    for genre, prob in top_3:
        print(f"  {genre}: {prob:.2%}")
```

---

## Expected Performance

### Individual Models (GTZAN 10-genre)
- MERT alone: 78-82%
- CLAP alone: 68-72%
- PANNs alone: 65-70%

### Ensemble Methods
- **Feature Concatenation**: 80-85%
- **Weighted Voting**: 78-83%
- **Stacking**: 82-88%

### Best Configuration
- Method: Feature Concatenation
- Classifier: Random Forest (200 trees)
- Expected: **82-88% accuracy**

---

## Optimization Tips

### 1. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 1000 features
selector = SelectKBest(f_classif, k=1000)
X_selected = selector.fit_transform(X_train, y_train)
```

### 2. Dimensionality Reduction
```python
from sklearn.decomposition import PCA

# Reduce to 512 dimensions
pca = PCA(n_components=512)
X_reduced = pca.fit_transform(X_train)
```

### 3. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [20, 30, 40],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
```

---

## Processing Time Estimate

### For 1000 songs (GTZAN):
- Feature extraction: 8-12 hours (CPU)
- Training: 10-20 minutes
- Inference (per song): 60-90 seconds

### Optimization:
- Extract features once, save to disk
- Use GPU for 5-10x speedup
- Batch processing for efficiency

---

## Summary

**Recommended Approach:**
1. Extract ensemble features offline (one-time, 8-12 hours)
2. Train Random Forest classifier (20 minutes)
3. Save model for fast inference
4. Expected accuracy: **82-88%**

**Key Advantages:**
- Combines strengths of all three models
- MERT: music understanding
- CLAP: semantic features
- PANNs: audio patterns
- 10-15% better than single model

**Date**: 2025-11-25


---
# FMA Feature Extraction

# FMA Feature Extraction Process 🎵

Complete guide to extracting 518-dimensional features compatible with the FMA dataset.

## Overview

The FMA (Free Music Archive) dataset provides pre-computed audio features that enable fast training without processing raw audio. This extractor replicates those 518 features for compatibility.

## Feature Breakdown (518 Total)

### 1. Chroma Features (84 features)
- **Dimensions**: 12 pitch classes
- **Statistics per dimension**: 7 (mean, std, min, max, median, q25, q75)
- **Total**: 12 × 7 = 84 features
- **Purpose**: Captures harmonic and melodic characteristics

### 2. Tonnetz Features (42 features)
- **Dimensions**: 6 tonal centroids
- **Statistics per dimension**: 7
- **Total**: 6 × 7 = 42 features
- **Purpose**: Represents tonal relationships and harmony

### 3. MFCC Features (140 features)
- **Dimensions**: 20 coefficients
- **Statistics per dimension**: 7
- **Total**: 20 × 7 = 140 features
- **Purpose**: Captures timbral texture (most important for genre)

### 4. Spectral Contrast (49 features)
- **Dimensions**: 7 frequency bands
- **Statistics per dimension**: 7
- **Total**: 7 × 7 = 49 features
- **Purpose**: Measures spectral peak-valley differences

### 5. Spectral Centroid (7 features)
- **Dimensions**: 1
- **Statistics**: 7
- **Purpose**: "Center of mass" of spectrum (brightness)

### 6. Spectral Bandwidth (7 features)
- **Dimensions**: 1
- **Statistics**: 7
- **Purpose**: Spectral spread around centroid

### 7. Spectral Rolloff (7 features)
- **Dimensions**: 1
- **Statistics**: 7
- **Purpose**: Frequency below which 85% of energy is contained

### 8. Zero Crossing Rate (7 features)
- **Dimensions**: 1
- **Statistics**: 7
- **Purpose**: Measures noisiness and percussiveness

### 9. RMS Energy (7 features)
- **Dimensions**: 1
- **Statistics**: 7
- **Purpose**: Overall loudness/energy

### 10. Tempo (1 feature)
- **Value**: Single BPM estimate
- **Purpose**: Rhythmic tempo

### 11. Mel Spectrogram (182 features)
- **Dimensions**: 26 mel bands
- **Statistics per dimension**: 7
- **Total**: 26 × 7 = 182 features
- **Purpose**: Perceptually-scaled frequency representation

## Statistical Summary (7 per feature)

For each feature dimension, we compute:

1. **Mean**: Average value over time
2. **Standard Deviation**: Temporal variation
3. **Minimum**: Lowest value
4. **Maximum**: Highest value
5. **Median**: Middle value (robust to outliers)
6. **25th Percentile**: Lower quartile
7. **75th Percentile**: Upper quartile

## Usage Examples

### Basic Usage

```python
from extract_fma_features import FMAFeatureExtractor

# Initialize extractor
extractor = FMAFeatureExtractor(sr=22050)

# Extract from single file
features = extractor.extract_features('song.wav')
print(features.shape)  # (518,)
```

### Batch Processing

```python
# Extract from directory
features, filenames = extractor.extract_batch(
    audio_dir='./music',
    output_file='features.npy',
    file_extension='.wav'
)
print(features.shape)  # (n_files, 518)
```

### Command Line

```bash
# Extract to NPY (fast, compact)
python3 extract_fma_features.py \
    --input ./music \
    --output features.npy \
    --extension .wav

# Extract to CSV (human-readable)
python3 extract_fma_features.py \
    --input ./music \
    --output features.csv \
    --extension .mp3
```

## Processing Pipeline

```
Audio File (.wav, .mp3, etc.)
    ↓
Load with librosa (22050 Hz, 30s)
    ↓
Extract 11 feature types
    ↓
Compute 7 statistics per dimension
    ↓
Concatenate into 518-dim vector
    ↓
Save as NPY or CSV
```

## Performance

### Single File
- **Time**: ~0.5-1 second per 30s track
- **Memory**: ~50 MB peak
- **Output**: 518 float32 values (2 KB)

### Batch Processing (1000 files)
- **Time**: ~10-15 minutes
- **Memory**: ~100-200 MB
- **Output NPY**: ~2 MB (vs 951 MB CSV)
- **Loading**: 1-2 seconds (vs 30-60s CSV)

## File Format Comparison

| Format | Size | Load Time | Use Case |
|--------|------|-----------|----------|
| NPY | 2 MB | 1-2s | Training (recommended) |
| CSV | 951 MB | 30-60s | Analysis/inspection |
| HDF5 | 3 MB | 2-3s | Large datasets |

## Integration with Training

### Option 1: Pre-compute Features

```python
# 1. Extract features once
extractor = FMAFeatureExtractor()
features, files = extractor.extract_batch('./music', 'features.npy')

# 2. Train with features
import numpy as np
X = np.load('features.npy')
# ... training code
```

### Option 2: On-the-fly Extraction

```python
# Extract during training (slower)
class MusicDataset(Dataset):
    def __init__(self, audio_files):
        self.files = audio_files
        self.extractor = FMAFeatureExtractor()
    
    def __getitem__(self, idx):
        features = self.extractor.extract_features(self.files[idx])
        return features
```

## Feature Importance

Based on genre classification experiments:

1. **MFCC** (140 features): Most important for timbre
2. **Spectral Contrast** (49 features): Distinguishes rock/metal from classical
3. **Chroma** (84 features): Important for harmonic genres (jazz, classical)
4. **Mel Spectrogram** (182 features): General frequency content
5. **Rhythm features** (Tempo, ZCR): Separates electronic/hip-hop

## Optimization Tips

### Speed
- Use NPY format (20-30x faster loading)
- Pre-compute features before training
- Use multiprocessing for batch extraction
- Reduce sample rate if acceptable (16000 Hz)

### Memory
- Process in batches if dataset is large
- Use float32 instead of float64
- Delete audio after feature extraction

### Accuracy
- Use 30-second clips (full songs may vary)
- Normalize features before training
- Consider feature selection (top 200-300)

## Troubleshooting

### Issue: Features are all zeros
- **Cause**: Audio file corrupted or unsupported format
- **Solution**: Check file with `librosa.load()` first

### Issue: Extraction is slow
- **Cause**: High sample rate or long audio files
- **Solution**: Limit duration to 30s, use sr=22050

### Issue: Out of memory
- **Cause**: Processing too many files at once
- **Solution**: Process in smaller batches

### Issue: Features don't match FMA
- **Cause**: Different librosa version or parameters
- **Solution**: Use librosa 0.11.0, verify parameters

## Validation

To verify your extracted features match FMA format:

```python
# Check shape
assert features.shape[1] == 518, "Should be 518 features"

# Check data type
assert features.dtype == np.float32, "Should be float32"

# Check for NaN/Inf
assert not np.isnan(features).any(), "No NaN values"
assert not np.isinf(features).any(), "No Inf values"

# Check reasonable ranges
print(f"Mean: {features.mean():.4f}")  # Should be ~0-10
print(f"Std: {features.std():.4f}")    # Should be ~1-100
```

## Next Steps

After extracting features:

1. **Normalize**: Use StandardScaler or MinMaxScaler
2. **Split**: Train/validation/test sets
3. **Train**: Use with `train_msd.py` for 77% accuracy in 2 minutes
4. **Evaluate**: Check per-genre performance

## References

- FMA Dataset: https://github.com/mdeff/fma
- Librosa Documentation: https://librosa.org/
- Audio Feature Extraction: Müller, M. (2015). Fundamentals of Music Processing

---

**Last Updated**: November 27, 2025


---
# OpenJMLA Usage

---
language:
- en
pipeline_tag: text-generation
tags:
- audio2text
- music2text
- musicllm
- music foundation model
license: cc
---
<img src="https://huggingface.co/UniMus/OpenJMLA/resolve/main/UniMus_logo_0.png" alt="drawing" width="256"/>

# UniMus Project: OpenJMLA


<br>
 &nbsp<a href="https://arxiv.org/pdf/2310.10159.pdf"> reImplementation of JMLA</a>  
</p>
<br>

Music tagging is a task to predict the tags of music recordings. 
However, previous music tagging research primarily focuses on close-set music tagging tasks which can not be generalized to new tags. 
In this work, we propose a zero-shot music tagging system modeled by a joint music and language attention (**JMLA**) model to address the open-set music tagging problem. 
The **JMLA** model consists of an audio encoder modeled by a pretrained masked autoencoder and a decoder modeled by a Falcon7B. 
We introduce preceiver resampler to convert arbitrary length audio into fixed length embeddings. 
We introduce dense attention connections between encoder and decoder layers to improve the information flow between the encoder and decoder layers. 
We collect a large-scale music and description dataset from the internet. 
We propose to use ChatGPT to convert the raw descriptions into formalized and diverse descriptions to train the **JMLA** models. 
Our proposed **JMLA** system achieves a zero-shot audio tagging accuracy of 64.82% on the GTZAN dataset, outperforming previous zero-shot systems and achieves comparable results to previous systems on the FMA and the MagnaTagATune datasets.


## Requirements
* conda create -name SpectPrompt python=3.9
* pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
* pip install transformers datasets librosa einops_exts einops mmcls peft ipdb torchlibrosa
* pip install -U openmim
* mim install mmcv==1.7.1
  <br>

## Quickstart
Below, we provide simple examples to show how to use **OpenJMLA** with 🤗 Transformers.  

#### 🤗 Transformers

To use OpenJMLA for the inference, all you need to do is to input a few lines of codes as demonstrated below.

```python
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

model = AutoModel.from_pretrained('UniMus/OpenJMLA', trust_remote_code=True)
device = model.device
# sample rate: 16k
music_path = '/path/to/music.wav'
# 1. get logmelspectrogram
# get the file wav_to_mel.py from https://github.com/taugastcn/SpectPrompt.git
from wav_to_mel import wav_to_mel
lms = wav_to_mel(music_path)

import os
from torch.nn.utils.rnn import pad_sequence
import random
# get the file transforms.py from https://github.com/taugastcn/SpectPrompt.git
from transforms import Normalize, SpecRandomCrop, SpecPadding, SpecRepeat
transforms = [ Normalize(-4.5, 4.5), SpecRandomCrop(target_len=2992), SpecPadding(target_len=2992), SpecRepeat() ]
lms = lms.numpy()
for trans in transforms:
    lms = trans(lms)

# 2. template of input
input_dic = dict()
input_dic['filenames'] = [music_path.split('/')[-1]]
input_dic['ans_crds'] = [0]
input_dic['audio_crds'] = [0]
input_dic['attention_mask'] = torch.tensor([[1, 1, 1, 1, 1]]).to(device)
input_dic['input_ids'] = torch.tensor([[1, 694, 5777, 683, 13]]).to(device)
input_dic['spectrogram'] = torch.from_numpy(lms).unsqueez(dim=0).to(device)
# 3. generation
model.eval()
gen_ids = model.forward_test(input)
gen_text = model.neck.tokenizer.batch_decode(gen_ids.clip(0))
# 4. Post-processing
# Given that the training data may contain biases, the generated texts might need some straightforward post-processing to ensure accuracy.
# In future versions, we will enhance the quality of the data.
gen_text = gen_text.split('<s>')[-1].split('\n')[0].strip()
gen_text = gen_text.replace(' in Chinese','')
gen_text = gen_text.replace(' Chinese','')
print(gen_text)
```

## Example

### music: 
https://www.youtube.com/watch?v=Q_yuO8UNGmY

### caption: 
Instruments: Vocals, piano, strings
Genre: pop
Theme: Heartbreak.
Mood: Melancholy.
Era: Contemporary.
Tempo: Fast
Best scene: A small, dimly lit bar. The melancholy mood of this song will complement the stage-inspired melody.

## Citation
If you find our paper and code useful in your research, please consider giving a star and citation

```BibTeX
@article{JMLA,
  title={JOINT MUSIC AND LANGUAGE ATTENTION MODELS FOR ZERO-SHOT MUSIC TAGGING},
  author={Xingjian Du, Zhesong Yu, Jiaju Lin, Bilei Zhu, Qiuqiang Kong},
  journal={arXiv preprint arXiv:2310.10159},
  year={2023}
}
```
<br>