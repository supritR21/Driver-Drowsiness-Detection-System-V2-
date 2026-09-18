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
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

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

CLASS_NAMES = ["alert", "drowsy", "microsleep"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# HELPERS
# =========================
def save_confusion_matrix(cm: np.ndarray, classes: list[str], path: Path, normalize: bool = False) -> None:
    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str], path: Path) -> dict[str, float]:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    plt.figure(figsize=(8, 7))

    auc_scores: dict[str, float] = {}

    for i, class_name in enumerate(classes):
        # Skip ROC if a class is absent in y_true
        if len(np.unique(y_true_bin[:, i])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = float(roc_auc)

        plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC = {roc_auc:.3f})")

    # micro-average
    try:
        micro_auc = roc_auc_score(y_true_bin, y_prob, average="micro", multi_class="ovr")
        auc_scores["micro"] = float(micro_auc)
    except Exception:
        micro_auc = None

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    return auc_scores


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
all_probs = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)

        logits, _ = model(xb)
        probs = torch.softmax(logits, dim=1)

        p = torch.argmax(probs, dim=1)

        preds.extend(p.cpu().numpy())
        true.extend(yb.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

true_arr = np.array(true, dtype=np.int64)
preds_arr = np.array(preds, dtype=np.int64)
probs_arr = np.array(all_probs, dtype=np.float32)

test_acc = accuracy_score(true_arr, preds_arr)
test_f1 = f1_score(true_arr, preds_arr, average="macro")

print("\nTEST ACC:", test_acc)
print("TEST F1:", test_f1)

report = classification_report(true_arr, preds_arr, target_names=CLASS_NAMES, zero_division=0)
cm = confusion_matrix(true_arr, preds_arr)

print("\nClassification Report")
print(report)

print("\nConfusion Matrix")
print(cm)

# Save report and metrics
with open(CKPT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

metrics = {
    "test_accuracy": float(test_acc),
    "test_f1_macro": float(test_f1),
    "best_val_f1": float(best_f1),
    "best_epoch": int(best_epoch),
    "train_class_counts": {str(k): int(v) for k, v in counts.items()},
}
with open(CKPT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# =========================
# PLOTS
# =========================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(CKPT_DIR / "loss_curves.png", dpi=200)
plt.close()

save_confusion_matrix(cm, CLASS_NAMES, CKPT_DIR / "confusion_matrix.png", normalize=False)
save_confusion_matrix(cm, CLASS_NAMES, CKPT_DIR / "confusion_matrix_normalized.png", normalize=True)

auc_scores = save_roc_curves(true_arr, probs_arr, CLASS_NAMES, CKPT_DIR / "roc_curves.png")
metrics["roc_auc"] = auc_scores
with open(CKPT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(" - loss_curves.png")
print(" - confusion_matrix.png")
print(" - confusion_matrix_normalized.png")
print(" - roc_curves.png")
print(" - classification_report.txt")
print(" - metrics.json")