from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from typing import Deque

import numpy as np


class SessionStateStore:
    def __init__(self, seq_len: int = 45):
        self.seq_len = seq_len
        self._buffers: dict[str, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=seq_len))
        self._last_level: dict[str, str] = {}
        self._lock = Lock()

    def append_features(self, session_id: str, features: np.ndarray) -> None:
        with self._lock:
            self._buffers[session_id].append(features.astype(np.float32))

    def get_sequence(self, session_id: str) -> list[np.ndarray]:
        with self._lock:
            return list(self._buffers[session_id])

    def get_last_level(self, session_id: str) -> str:
        with self._lock:
            return self._last_level.get(session_id, "safe")

    def set_last_level(self, session_id: str, level: str) -> None:
        with self._lock:
            self._last_level[session_id] = level

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._buffers.pop(session_id, None)
            self._last_level.pop(session_id, None)