import torch
import torchaudio
from pathlib import Path

# Paths
music_dir = Path("/media/mijesu_970/SSD_Data/Music_TBC")
model_dir = Path("/media/mijesu_970/SSD_Data/AI_models")
dataset_dir = Path("/media/mijesu_970/SSD_Data/DataSets")

# Get audio files
audio_files = list(music_dir.glob("**/*.mp3")) + list(music_dir.glob("**/*.wav")+ list(music_dir.glob("**/*.flac"))
print(f"Found {len(audio_files)} audio files")

# Load and process first file as example
if audio_files:
    waveform, sample_rate = torchaudio.load(audio_files[0])
    print(f"Sample: {audio_files[0].name}")
    print(f"Shape: {waveform.shape}, Sample rate: {sample_rate}")
