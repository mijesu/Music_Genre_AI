import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import gc
import signal
import sys
import time

# Paths
MODEL_PATH = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
GTZAN_PATH = "/media/mijesu_970/SSD_Data/DataSets/GTZAN/Data"
FMA_PATH = "/media/mijesu_970/SSD_Data/DataSets/FMA/Data/fma_medium"
CHECKPOINT_PATH = "/media/mijesu_970/SSD_Data/Kiro_Projects/Music_Reclass/checkpoint.pth"

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Training control flags
paused = False
should_break = False

def signal_handler(sig, frame):
    """Handle Ctrl+C for pause/resume"""
    global paused
    paused = not paused
    if paused:
        print("\n⏸️  Training PAUSED. Press Ctrl+C again to resume, or Ctrl+\\ to stop.")
    else:
        print("▶️  Training RESUMED.")

signal.signal(signal.SIGINT, signal_handler)

def format_time(seconds):
    """Format seconds to readable time string"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

def clear_memory():
    """Clear memory and run garbage collection"""
    gc.collect()
    torch.cuda.empty_cache()

def show_gpu_memory():
    """Display GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

def save_checkpoint(model, optimizer, epoch, loss, acc, batch_idx=None):
    """Save complete training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'timestamp': torch.tensor(0)  # Placeholder for timestamp
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    status = f"epoch {epoch}" + (f", batch {batch_idx}" if batch_idx else "")
    print(f"\n💾 Checkpoint saved at {status}")

def load_checkpoint(model, optimizer):
    """Load training checkpoint and return resume info"""
    if Path(CHECKPOINT_PATH).exists():
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        batch_idx = checkpoint.get('batch_idx', None)
        
        print(f"📂 Resuming from epoch {epoch + 1}")
        if batch_idx:
            print(f"   Last completed batch: {batch_idx}")
        print(f"   Previous loss: {checkpoint['loss']:.4f}, accuracy: {checkpoint['accuracy']:.2f}%")
        
        return epoch + 1, checkpoint['loss'], checkpoint['accuracy']
    return 0, None, None

class AudioDataset(Dataset):
    def __init__(self, data_path, genres, sr=22050, duration=30):
        self.data_path = Path(data_path)
        self.genres = genres
        self.sr = sr
        self.duration = duration
        self.target_length = sr * duration
        self.files = []
        
        for idx, genre in enumerate(genres):
            genre_path = self.data_path / genre
            if genre_path.exists():
                for audio_file in genre_path.glob('*.wav'):
                    self.files.append((str(audio_file), idx))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path, label = self.files[idx]
        audio, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        
        # Pad or crop to fixed length
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        else:
            audio = audio[:self.target_length]
        
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        return mel_spec_tensor, label

class GenreClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    global paused, should_break
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_batches = len(train_loader)
    start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Check for pause
        while paused:
            time.sleep(0.5)
        
        # Check for break
        if should_break:
            print("\n🛑 Training stopped by user")
            return total_loss / max(batch_idx, 1), 100. * correct / max(total, 1)
        
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Calculate time estimates
        elapsed = time.time() - start_time
        batches_done = batch_idx + 1
        batches_left = total_batches - batches_done
        time_per_batch = elapsed / batches_done
        eta = time_per_batch * batches_left
        
        # Show completion rate with time estimate
        completion = batches_done / total_batches * 100
        print(f"\rBatch {batches_done}/{total_batches} ({completion:.1f}%) - Loss: {loss.item():.4f} - ETA: {format_time(eta)}", end='')
        
        # Clear memory after each batch
        del inputs, labels, outputs, loss
        clear_memory()
    
    total_time = time.time() - start_time
    print(f" - Time: {format_time(total_time)}")
    return total_loss / len(train_loader), 100. * correct / total

def main():
    # Clear memory before starting
    clear_memory()
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    show_gpu_memory()
    
    # Load dataset
    print("\nLoading GTZAN dataset...")
    dataset = AudioDataset(GTZAN_PATH, GENRES)
    print(f"Found {len(dataset)} audio files")
    show_gpu_memory()
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0)
    
    # Model
    model = GenreClassifier(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Load checkpoint if exists (for replay)
    start_epoch, prev_loss, prev_acc = load_checkpoint(model, optimizer)
    
    print("\nModel loaded:")
    show_gpu_memory()
    
    # Training
    print("\n🎯 Starting training...")
    print("Controls: Ctrl+C = Pause/Resume, Ctrl+\\ = Stop")
    print("Training state auto-saves after each epoch\n")
    epochs = 10
    training_start = time.time()
    epoch_times = []
    
    try:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Calculate overall ETA
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            overall_eta = avg_epoch_time * remaining_epochs
            
            print(f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - Epoch time: {format_time(epoch_time)}")
            if remaining_epochs > 0:
                print(f"⏱️  Estimated time remaining: {format_time(overall_eta)}")
            show_gpu_memory()
            
            # Save checkpoint after each epoch
            save_checkpoint(model, optimizer, epoch, train_loss, train_acc)
            clear_memory()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        save_checkpoint(model, optimizer, epoch, train_loss, train_acc)
        print("💡 Run the script again to resume from this point")
        return
    
    total_training_time = time.time() - training_start
    print(f"\n⏱️  Total training time: {format_time(total_training_time)}")
    
    # Save final model
    torch.save(model.state_dict(), 'genre_classifier.pth')
    print("✅ Training complete! Model saved as 'genre_classifier.pth'")
    
    # Clean up checkpoint
    if Path(CHECKPOINT_PATH).exists():
        Path(CHECKPOINT_PATH).unlink()
        print("🗑️  Checkpoint file removed")

if __name__ == '__main__':
    main()
