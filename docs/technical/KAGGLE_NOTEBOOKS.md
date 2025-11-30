# FMA Genre Classification Kaggle Notebook Analysis

**Date:** November 25, 2025  
**Time:** 07:43 (GMT+8)  
**Source:** `Reference/genre-classification-with-fma-data.ipynb`

---

## Overview

**Approach:** Traditional ML (XGBoost) with pre-computed features  
**Dataset:** FMA Medium (25,000 tracks, 16 genres)  
**Features:** 518 pre-computed audio features  
**Method:** Feature selection + PCA + XGBoost

---

## Data Processing Pipeline

### 1. Feature Loading
**Input:** `features.csv` (951 MB)
- 106,574 tracks × 518 features
- Features include: MFCC, chroma, spectral, tonnetz, zcr, rmse

**Processing:**
```python
def rename_fma_features(features):
    # Rename columns to: feature_number_statistic
    # Example: mfcc_01_mean, chroma_cqt_05_std
```

### 2. Label Extraction
**Input:** `tracks.csv` + `genres.csv`

**Steps:**
1. Extract `genre_top` from tracks.csv
2. 56,976 tracks missing genre_top
3. Fill missing using genre hierarchy from genres.csv
4. Final: 104,343 tracks with labels, 2,231 dropped (no genre)

**Genre Distribution (before filtering):**
- Electronic: ~15,000
- Experimental: ~12,000
- Rock: ~10,000
- Hip-Hop: ~8,000
- Folk: ~6,000
- Instrumental: ~5,000
- Pop: ~4,000
- International: ~3,000
- Others: <3,000 each

### 3. Data Filtering
**Removed:**
- "International" genre (too diverse)
- Genres with <1,000 samples (too small)

**Final Dataset:**
- 10 genres
- ~90,000+ tracks
- Balanced distribution

---

## Feature Engineering

### Manual Feature Selection
**Rationale:** Based on MIR (Music Information Retrieval) experience

**Removed Features:**
- Chromagram variants (STFT, CENS) - kept only CQT
- Spectral bandwidth & rolloff - highly correlated with centroid

**Selected Features:**
- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma CQT
- Spectral centroid
- Tonnetz (tonal centroid features)
- Zero crossing rate (ZCR)
- RMS energy

### Normalization
```python
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
```

### Dimensionality Reduction
**PCA (Principal Component Analysis):**
- Input: 518 features
- Output: 60 components
- Explained variance: 80%
- Reduces computational cost
- Removes noise and redundancy

**Visualization:**
- Cumulative explained variance plot
- 60 components capture 80% variance
- 90% variance requires ~100 components

---

## Model Training

### XGBoost Classifier
```python
xgb = XGBClassifier(n_estimators=50)
xgb.fit(X_train, y_train)
```

**Parameters:**
- n_estimators: 50 trees
- Default hyperparameters

**Train/Test Split:**
- Training: 70%
- Testing: 30%
- Shuffle: True
- Random state: 123

---

## Results

### Performance Metrics
- **Accuracy:** ~55-60%
- **F1 Score (macro):** ~0.50

### Confusion Matrix Analysis

**High Confusion Pairs:**
1. Rock ↔ Blues (similar instrumentation)
2. Electronic ↔ Pop (electronic elements in pop)
3. Electronic ↔ Hip-Hop (beats and production)

**Observations:**
1. Model biased toward common labels (Electronic, Experimental, Rock)
2. Genre similarity causes confusion
3. "Experimental" and "Instrumental" are not musically distinct genres

---

## Key Insights

### Advantages
✅ **Fast training:** Minutes vs hours for deep learning  
✅ **Interpretable:** Feature importance analysis possible  
✅ **No GPU required:** Runs on CPU  
✅ **Small model size:** <10 MB vs 100+ MB for neural networks  
✅ **Good baseline:** 55-60% accuracy without audio processing

### Limitations
❌ **Lower accuracy:** 55-60% vs 70-90% for deep learning  
❌ **Fixed features:** Cannot learn new representations  
❌ **Manual selection:** Requires domain expertise  
❌ **Genre confusion:** Similar genres hard to distinguish

### Comparison with Deep Learning

| Approach | Accuracy | Training Time | Model Size | GPU |
|----------|----------|---------------|------------|-----|
| XGBoost + PCA | 55-60% | 5-10 min | <10 MB | No |
| CNN (Basic) | 70-80% | 45 min | ~50 MB | Yes |
| Transfer Learning | 80-90% | 4 hours | ~50 MB | Yes |

---

## Methodology Summary

```
FMA features.csv (518 features)
         ↓
Manual Feature Selection
         ↓
StandardScaler Normalization
         ↓
PCA (518 → 60 components)
         ↓
XGBoost Classifier (50 trees)
         ↓
Predictions (10 genres)
```

---

## Recommendations

### For Quick Baseline
- Use this XGBoost approach
- Fast results for initial testing
- Good for feature importance analysis

### For Production
- Use deep learning (CNN or transfer learning)
- Higher accuracy (70-90%)
- Better generalization

### For Ensemble
- Combine XGBoost + CNN predictions
- Voting or averaging
- Potential 5-10% accuracy boost

---

## Related Files

**Notebook:** `Reference/genre-classification-with-fma-data.ipynb`  
**Dataset:** FMA Medium (25,000 tracks)  
**Features:** `DataSets/FMA/Misc/fma_metadata/features.csv`  
**Converted:** `AI_models/FMA/FMA.npy` (211 MB)

---

**Status:** ✅ Analyzed - Traditional ML baseline approach documented


---
# Kaggle Notebook Summary

# Summary: Genre Classification with FMA Data

**Source:** https://www.kaggle.com/code/jojothepizza/genre-classification-with-fma-data

---

## Overview

This notebook demonstrates music genre classification using the FMA (Free Music Archive) dataset with traditional machine learning approaches using pre-computed audio features.

---

## Approach

### 1. Data Preprocessing

**Features Used:**
- Pre-computed audio features from FMA metadata (`features.csv`)
- Features include: MFCCs, spectral features, chroma features, tempo, etc.
- Total: ~500+ audio features per track

**Labels:**
- Uses `genre_top` (top-level genre) from FMA metadata
- Original dataset: 106,574 tracks
- After cleaning: ~104,000 tracks with valid genres

**Data Cleaning:**
- Removed tracks without `genre_top` labels
- Filled missing `genre_top` using parent genre from `genres.csv`
- Removed 'International' genre (not musically characteristic)
- Removed genres with <1,000 songs (small sample size)
- Final dataset: Focused on major genres only

---

## Feature Engineering

### Manual Feature Selection

**Removed highly correlated features:**
- Kept only CQT chromagram (removed STFT and CENS versions)
- Removed spectral bandwidth and rolloff (kept centroids only)
- Rationale: Reduce redundancy and computational cost

**Feature Analysis:**
- Created correlation heatmaps to identify redundant features
- Focused on "mean" statistics of audio features
- Reduced feature set from 500+ to more manageable subset

### Dimensionality Reduction

**PCA (Principal Component Analysis):**
- Applied StandardScaler for normalization
- Used PCA to reduce dimensions
- Selected 60 components (explains ~80% variance)
- Transformed features: 500+ → 60 PCA components

---

## Model Training

### Algorithm: XGBoost Classifier

**Configuration:**
- Model: XGBClassifier
- n_estimators: 50
- Features: 60 PCA components
- Train/Test split: 70/30

**Data Split:**
- Training set: 70% of data
- Test set: 30% of data
- Random state: 123 (for reproducibility)
- Shuffle: True

---

## Key Differences from Our Approach

| Aspect | Kaggle Notebook | Our Approach |
|--------|----------------|--------------|
| **Features** | Pre-computed FMA features | Mel-spectrograms (raw audio) |
| **Model** | XGBoost (traditional ML) | CNN / Transfer Learning (Deep Learning) |
| **Feature Extraction** | Manual selection + PCA | OpenJMLA / CNN automatic learning |
| **Dataset** | FMA Medium/Large | GTZAN + FMA Medium |
| **Preprocessing** | StandardScaler + PCA | Audio → Mel-spectrogram |
| **Approach** | Feature engineering heavy | End-to-end learning |

---

## Advantages of Kaggle Approach

✅ **Fast training** - XGBoost is quick with tabular data
✅ **Interpretable** - Can analyze feature importance
✅ **Low memory** - No need to load audio files
✅ **Pre-computed features** - Uses FMA's provided features
✅ **Traditional ML** - Works well with limited compute

---

## Advantages of Our Approach

✅ **End-to-end learning** - Learns features automatically
✅ **Transfer learning** - Leverages OpenJMLA pre-training
✅ **Raw audio** - Not limited to pre-computed features
✅ **Deep learning** - Can capture complex patterns
✅ **Scalable** - Can work with any audio dataset

---

## What We Can Learn

### 1. Feature Selection Strategy
- Remove highly correlated features
- Use correlation heatmaps for analysis
- Focus on meaningful feature subsets

### 2. Data Cleaning
- Remove ambiguous genres ('International')
- Filter out genres with insufficient samples
- Use parent genres for missing labels

### 3. Dimensionality Reduction
- PCA can reduce features while preserving variance
- 60 components for ~80% variance is a good target
- Normalize features before PCA

### 4. Baseline Comparison
- XGBoost provides a strong baseline
- Can compare our deep learning results against this

---

## Potential Improvements to Our Project

### 1. Hybrid Approach
```python
# Combine deep learning features + traditional features
openjmla_features = extract_features(audio)  # 768-dim
fma_features = load_fma_features(track_id)  # Pre-computed
combined = concatenate([openjmla_features, fma_features])
classifier(combined)
```

### 2. Use FMA Pre-computed Features
- FMA provides pre-computed audio features
- Can use as additional input or for comparison
- Located in: `fma_metadata/features.csv`

### 3. Genre Filtering
- Apply same filtering: remove small genres
- Focus on top 8 genres with >1000 samples
- Remove ambiguous categories

### 4. Baseline Comparison
- Train XGBoost on FMA features as baseline
- Compare against our CNN/OpenJMLA approach
- Measure improvement from deep learning

---

## Implementation Ideas

### Quick Baseline with FMA Features

```python
# Load FMA pre-computed features
features = pd.read_csv('fma_metadata/features.csv')
tracks = pd.read_csv('fma_metadata/tracks.csv')

# Extract genre labels
labels = tracks['genre_top']

# Train XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Compare with our deep learning model
```

### Feature Importance Analysis

```python
# After training XGBoost
importance = model.feature_importances_
# Identify which audio features matter most
# Use insights to improve our CNN architecture
```

---

## Conclusion

The Kaggle notebook demonstrates a **traditional ML approach** using pre-computed features and XGBoost. It's fast, interpretable, and effective.

Our approach uses **deep learning with raw audio**, which is more flexible and can learn features automatically through OpenJMLA transfer learning.

**Best strategy:** Use both approaches
1. XGBoost baseline for quick results and feature analysis
2. Deep learning for maximum performance
3. Compare and combine insights

---

*Summary created: 2025-11-23*
*Notebook analyzed: genre-classification-with-fma-data.ipynb*
