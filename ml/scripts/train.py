from __future__ import annotations

from pathlib import Path
import json
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

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
EPOCHS = 40
PATIENCE = 20
LR = 3e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# LOAD DATA
# =========================
X = np.load(DATA_DIR / "X.npy").astype(np.float32)
y = np.load(DATA_DIR / "y.npy").astype(np.int64)

with open(DATA_DIR / "meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# =========================
# VIDEO-LEVEL + CLASS-AWARE SPLIT
# =========================
# Build video -> class mapping
video_to_class: dict[str, int] = {}
for item in meta:
    video_to_class[item["video"]] = int(item["class_idx"])

class_to_videos = {0: [], 1: [], 2: []}
for video_path, class_idx in video_to_class.items():
    class_to_videos[class_idx].append(video_path)

for c in class_to_videos:
    class_to_videos[c] = sorted(set(class_to_videos[c]))

print("Unique videos per class:")
for c in [0, 1, 2]:
    print(f"  class {c}: {len(class_to_videos[c])} videos")

def split_class_videos(videos: list[str]):
    if len(videos) < 3:
        raise ValueError(
            f"Need at least 3 videos for class-wise split, got {len(videos)}"
        )

    train_v, temp_v = train_test_split(
        videos,
        test_size=0.30,
        random_state=SEED,
        shuffle=True,
    )

    val_v, test_v = train_test_split(
        temp_v,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
    )

    return train_v, val_v, test_v

train_videos: list[str] = []
val_videos: list[str] = []
test_videos: list[str] = []

for c in [0, 1, 2]:
    tr, va, te = split_class_videos(class_to_videos[c])
    train_videos.extend(tr)
    val_videos.extend(va)
    test_videos.extend(te)

# Reproducible shuffle of combined lists
random.shuffle(train_videos)
random.shuffle(val_videos)
random.shuffle(test_videos)

video_to_indices: dict[str, list[int]] = {}
for idx, item in enumerate(meta):
    video_to_indices.setdefault(item["video"], []).append(idx)

def collect_indices(video_list: list[str]) -> list[int]:
    idxs: list[int] = []
    for v in video_list:
        idxs.extend(video_to_indices[v])
    return idxs

train_idx = collect_indices(train_videos)
val_idx = collect_indices(val_videos)
test_idx = collect_indices(test_videos)

print("\nSplit summary:")
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

# Save split info for reproducibility
with open(CKPT_DIR / "split_info.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "train_videos": train_videos,
            "val_videos": val_videos,
            "test_videos": test_videos,
        },
        f,
        indent=2,
    )

# =========================
# DATASET
# =========================
class SeqDataset(Dataset):
    def __init__(self, indices, augment: bool = False):
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = torch.tensor(X[idx], dtype=torch.float32)
        y_item = torch.tensor(y[idx], dtype=torch.long)

        # Training-only feature jitter to help generalization
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = x + torch.randn_like(x) * 0.01
                x = torch.clamp(x, -5.0, 5.0)

        return x, y_item

train_ds = SeqDataset(train_idx, augment=True)
val_ds = SeqDataset(val_idx, augment=False)
test_ds = SeqDataset(test_idx, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL
# =========================
model = DrowsinessBiLSTM().to(DEVICE)

# Dynamic class weights from TRAIN split only
train_labels = y[train_idx]
counts = Counter(train_labels.tolist())
total = len(train_labels)

class_weights = torch.tensor(
    [
        total / (3.0 * counts.get(0, 1)),
        total / (3.0 * counts.get(1, 1)),
        total / (3.0 * counts.get(2, 1)),
    ],
    dtype=torch.float32,
    device=DEVICE,
)

print("\nTrain class counts:", dict(counts))
print("Class weights:", class_weights.detach().cpu().numpy().tolist())

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

best_f1 = -1.0
best_epoch = -1
early_stop_counter = 0
train_losses = []
val_losses = []

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_train = total_loss / max(1, len(train_loader))
    train_losses.append(avg_train)

    # Validation
    model.eval()
    preds = []
    true = []
    total_val_loss = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits, _ = model(xb)
            loss = criterion(logits, yb)
            total_val_loss += loss.item()

            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy())
            true.extend(yb.cpu().numpy())

    avg_val = total_val_loss / max(1, len(val_loader))
    val_losses.append(avg_val)

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1:02d} | "
        f"train_loss={avg_train:.4f} | "
        f"val_loss={avg_val:.4f} | "
        f"val_acc={acc:.4f} | "
        f"val_f1={f1:.4f} | "
        f"lr={current_lr:.6f}"
    )

    scheduler.step(f1)

    if f1 > best_f1 + 1e-4:
        best_f1 = f1
        best_epoch = epoch + 1
        early_stop_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "mean_file": "feature_mean.npy",
                "std_file": "feature_std.npy",
                "seq_len": X.shape[1],
                "input_dim": X.shape[2],
                "best_val_f1": best_f1,
                "best_epoch": best_epoch,
            },
            CKPT_DIR / "drowsiness_bilstm.pt",
        )
        print("✅ Best model saved")
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break

# =========================
# TEST
# =========================
print("\nLoading best model...")
ckpt = torch.load(CKPT_DIR / "drowsiness_bilstm.pt", map_location=DEVICE)
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
        true.extend(yb.cpu().numpy())

print("\nTEST ACC:", accuracy_score(true, preds))
print("TEST F1:", f1_score(true, preds, average="macro"))

print("\nClassification Report")
print(classification_report(true, preds, zero_division=0))

print("\nConfusion Matrix")
print(confusion_matrix(true, preds))

# =========================
# PLOTS
# =========================
plt.figure()
plt.plot(train_losses)
plt.title("Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(CKPT_DIR / "train_loss.png")
plt.close()

plt.figure()
plt.plot(val_losses)
plt.title("Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(CKPT_DIR / "val_loss.png")
plt.close()