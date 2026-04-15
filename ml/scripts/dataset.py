from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DrowsinessDataset(Dataset):
    def __init__(self, x_path: str | Path, y_path: str | Path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y