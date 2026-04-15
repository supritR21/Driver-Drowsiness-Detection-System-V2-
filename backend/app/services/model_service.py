from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app.core.config import settings
from app.services.model_arch import DrowsinessBiLSTM


class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = 30
        self.input_dim = 10
        self.model = DrowsinessBiLSTM(input_dim=self.input_dim).to(self.device)
        self.model.eval()
        self.checkpoint_path = Path(settings.model_path)
        self.loaded = self._try_load_checkpoint()

    def _try_load_checkpoint(self) -> bool:
        if not self.checkpoint_path.exists():
            return False

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                return False
            self.model.eval()
            return True
        except Exception:
            return False

    def _heuristic_probs(self, sequence: np.ndarray) -> np.ndarray:
        """
        Fallback probabilities if the trained checkpoint is not available yet.
        sequence shape: (seq_len, 10)
        """
        ear_mean = float(sequence[:, 2].mean())
        mar_mean = float(sequence[:, 3].mean())
        pitch = float(np.abs(sequence[:, 4]).mean())
        yaw = float(np.abs(sequence[:, 5]).mean())
        roll = float(np.abs(sequence[:, 6]).mean())
        blink_flag = float(sequence[:, 7].mean())
        yawn_flag = float(sequence[:, 8].mean())

        risk = 0.0
        risk += max(0.0, 0.28 - ear_mean) * 8.0
        risk += max(0.0, mar_mean - 0.45) * 4.0
        risk += (pitch + yaw + roll) / 90.0
        risk += blink_flag * 0.4
        risk += yawn_flag * 0.8

        risk = float(np.clip(risk, 0.0, 3.0))

        alert = max(0.0, 1.0 - (risk / 3.0))
        drowsy = min(1.0, max(0.0, risk / 3.0))
        microsleep = min(1.0, max(0.0, (risk - 1.5) / 1.5))

        probs = np.array([alert, drowsy, microsleep], dtype=np.float32)
        probs = probs / probs.sum()
        return probs

    def predict(self, sequence: list[np.ndarray]) -> dict:
        seq = np.asarray(sequence, dtype=np.float32)

        if seq.shape != (self.seq_len, self.input_dim):
            raise ValueError(f"Expected sequence shape ({self.seq_len}, {self.input_dim}), got {seq.shape}")

        if not self.loaded:
            probs = self._heuristic_probs(seq)
            return {
                "source": "heuristic",
                "probabilities": probs.tolist(),
                "prediction": int(np.argmax(probs)),
                "attention_weights": None,
            }

        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, attention_weights = self.model.forward_with_attention(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

        return {
            "source": "model",
            "probabilities": probs.tolist(),
            "prediction": int(np.argmax(probs)),
            "attention_weights": attention_weights.squeeze(-1).detach().cpu().numpy().tolist(),
        }


model_service = ModelService()