import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# =========================
# Configuration
# =========================
DATA_ROOT = "data"
SAMPLE_RATE = 16000
DURATION = 1.0
N_FFT = 512
WIN_LENGTH = 400      # 25 ms
HOP_LENGTH = 160      # 10 ms
N_MELS = 40
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1        # 10% for validation
NUM_CLASSES = 9       # 8 keywords + noise

LABELS = [
    "down", "go", "left", "no",
    "right", "stop", "up", "yes",
    "_noise_"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# =========================
# Feature Extraction
# =========================
def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    
    # Pad or truncate to exactly 1 second
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    
    # Log mel + small epsilon
    log_mel = np.log(mel + 1e-6)
    
    # Verify shape: should be (40, 101)
    assert log_mel.shape == (N_MELS, 101), f"Unexpected shape: {log_mel.shape}"
    
    return log_mel.astype(np.float32)

# =========================
# Dataset
# =========================
class KWSDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {label: idx for idx, label in enumerate(LABELS)}
        
        for label in LABELS:
            folder = os.path.join(root_dir, label)
            if not os.path.isdir(folder):
                raise RuntimeError(f"Missing folder: {folder}")
            for fname in os.listdir(folder):
                if fname.lower().endswith(".wav"):
                    self.samples.append((os.path.join(folder, fname), self.label_map[label]))
        
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = extract_features(path)           # (40, 101)
        feat = torch.from_numpy(feat).unsqueeze(0)  # (1, 40, 101)
        return feat, label

# =========================
# Model (slightly improved)
# =========================
class KWSModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (16, 20, 50)
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (32, 10, 25)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# =========================
# Training Loop with Validation
# =========================
def train():
    dataset = KWSDataset(DATA_ROOT)
    
    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = KWSModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Total samples: {len(dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                _, pred = logits.max(1)
                val_correct += pred.eq(y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "kws_best.pth")
    
    # Load best model
    model.load_state_dict(torch.load("kws_best.pth"))
    return model

# =========================
# ONNX Export (FIXED)
# =========================
def export_onnx(model):
    model.eval()
    
    # Correct dummy input: (batch, channels, mel_bins, time_frames)
    # With hop=160, 16000 samples → 101 time frames
    dummy_input = torch.randn(1, 1, N_MELS, 101).to(DEVICE)
    
    torch.onnx.export(
        model,
        dummy_input,
        "kws.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={          # Optional: allow variable batch size
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print("ONNX model exported successfully: kws.onnx (input shape: [B, 1, 40, 101])")

# =========================
# Main
# =========================
if __name__ == "__main__":
    model = train()
    export_onnx(model)