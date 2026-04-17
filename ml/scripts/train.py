from __future__ import annotations

from pathlib import Path
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from model import DrowsinessBiLSTM

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets" / "processed"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# LOAD DATA
# =========================
X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

with open(DATA_DIR / "meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# =========================
# VIDEO LEVEL SPLIT
# =========================
video_to_indices = {}

for idx, item in enumerate(meta):
    vid = item["video"]
    if vid not in video_to_indices:
        video_to_indices[vid] = []
    video_to_indices[vid].append(idx)

videos = list(video_to_indices.keys())
random.shuffle(videos)

n = len(videos)
train_videos = videos[: int(0.7 * n)]
val_videos = videos[int(0.7 * n): int(0.85 * n)]
test_videos = videos[int(0.85 * n):]

def collect_indices(video_list):
    idxs = []
    for v in video_list:
        idxs.extend(video_to_indices[v])
    return idxs

train_idx = collect_indices(train_videos)
val_idx = collect_indices(val_videos)
test_idx = collect_indices(test_videos)

print("Train videos:", len(train_videos))
print("Val videos:", len(val_videos))
print("Test videos:", len(test_videos))

print("Train samples:", len(train_idx))
print("Val samples:", len(val_idx))
print("Test samples:", len(test_idx))

# =========================
# NORMALIZATION
# =========================
X_train = X[train_idx]

mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6

X = (X - mean) / std

np.save(CKPT_DIR / "feature_mean.npy", mean)
np.save(CKPT_DIR / "feature_std.npy", std)

# =========================
# DATASET
# =========================
class SeqDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
            torch.tensor(X[idx], dtype=torch.float32),
            torch.tensor(y[idx], dtype=torch.long),
        )

train_loader = DataLoader(SeqDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SeqDataset(val_idx), batch_size=BATCH_SIZE)
test_loader = DataLoader(SeqDataset(test_idx), batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
model = DrowsinessBiLSTM().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_f1 = 0
train_losses = []
val_losses = []

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train = total_loss / len(train_loader)
    train_losses.append(avg_train)

    # Validation
    model.eval()
    preds = []
    true = []
    total_val_loss = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits, _ = model(xb)
            loss = criterion(logits, yb)

            total_val_loss += loss.item()

            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy())
            true.extend(yb.cpu().numpy())

    avg_val = total_val_loss / len(val_loader)
    val_losses.append(avg_val)

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")

    print(
        f"Epoch {epoch+1:02d} | "
        f"train_loss={avg_train:.4f} | "
        f"val_loss={avg_val:.4f} | "
        f"val_acc={acc:.4f} | "
        f"val_f1={f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "mean_file": "feature_mean.npy",
                "std_file": "feature_std.npy",
            },
            CKPT_DIR / "drowsiness_bilstm.pt",
        )
        print("✅ Best model saved")

# =========================
# TEST
# =========================
print("\nLoading best model...")
ckpt = torch.load(CKPT_DIR / "drowsiness_bilstm.pt")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

preds = []
true = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)

        logits, _ = model(xb)
        p = torch.argmax(logits, dim=1)

        preds.extend(p.cpu().numpy())
        true.extend(yb.numpy())

print("\nTEST ACC:", accuracy_score(true, preds))
print("TEST F1:", f1_score(true, preds, average="macro"))

print("\nClassification Report")
print(classification_report(true, preds))

print("\nConfusion Matrix")
print(confusion_matrix(true, preds))

# =========================
# PLOTS
# =========================
plt.plot(train_losses)
plt.title("Train Loss")
plt.savefig(CKPT_DIR / "train_loss.png")
plt.close()

plt.plot(val_losses)
plt.title("Val Loss")
plt.savefig(CKPT_DIR / "val_loss.png")
plt.close()