#!/usr/bin/env python3
"""Simple example of using VGGish for music feature extraction"""

import torch
import numpy as np

# Install if needed: pip install torchvggish
from torchvggish import vggish, vggish_input

# Load model
model = vggish()
model.eval()

# Process audio file
audio_path = 'path/to/music.mp3'
examples = vggish_input.wavfile_to_examples(audio_path)

# Extract embeddings
with torch.no_grad():
    embeddings = model(examples)  # Shape: (num_segments, 128)

# Average over segments for single embedding
embedding_mean = embeddings.mean(dim=0)  # Shape: (128,)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Mean embedding shape: {embedding_mean.shape}")

# Save
np.save('vggish_features.npy', embedding_mean.numpy())
torch.save({'embedding': embedding_mean}, 'vggish_features.pth')
