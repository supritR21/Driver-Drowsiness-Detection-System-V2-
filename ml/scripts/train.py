from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import DrowsinessDataset
from model import DrowsinessBiLSTM

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets" / "processed"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_curve(values, title, filename):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.tight_layout()
    plt.savefig(CKPT_DIR / filename)
    plt.close()


def main():
    X_path = DATA_DIR / "X.npy"
    Y_path = DATA_DIR / "y.npy"

    dataset = DrowsinessDataset(X_path, Y_path)

    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=dataset.y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=dataset.y[temp_idx])

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)

    model = DrowsinessBiLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = 0.0
    train_losses = []
    val_losses = []

    for epoch in range(20):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        model.eval()
        val_preds = []
        val_targets = []
        val_running_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x)
                loss = criterion(logits, y)
                val_running_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = val_running_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        print(
            f"Epoch {epoch+1:02d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": 10,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "num_classes": 3,
                },
                CKPT_DIR / "drowsiness_bilstm.pt",
            )
            print("Saved best checkpoint.")

    plot_curve(train_losses, "Training Loss", "train_loss.png")
    plot_curve(val_losses, "Validation Loss", "val_loss.png")

    checkpoint = torch.load(CKPT_DIR / "drowsiness_bilstm.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y.cpu().numpy())

    print("\nTest accuracy:", accuracy_score(test_targets, test_preds))
    print("Test F1:", f1_score(test_targets, test_preds, average="macro"))
    print("\nClassification report:\n", classification_report(test_targets, test_preds))
    print("\nConfusion matrix:\n", confusion_matrix(test_targets, test_preds))


if __name__ == "__main__":
    main()