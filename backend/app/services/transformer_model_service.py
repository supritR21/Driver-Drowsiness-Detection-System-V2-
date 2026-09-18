from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app.core.config import settings
from app.services.transformer_model_arch import (
    DrowsinessTransformer,
)


BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)


class TransformerModelService:

    def __init__(
        self,
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None

        self.seq_len = 60
        self.input_dim = 14

        self.feature_mean: (
            np.ndarray | None
        ) = None

        self.feature_std: (
            np.ndarray | None
        ) = None

        self.decision_threshold = 0.5

        self.loaded = False

        self.load_error: (
            str | None
        ) = None

        self.checkpoint_path = (
            self._resolve_checkpoint_path()
        )

        self._load_checkpoint()

    def _resolve_checkpoint_path(
        self,
    ) -> Path:

        path = Path(
            settings.transformer_model_path
        )

        if not path.is_absolute():

            path = (
                BACKEND_DIR
                / path
            ).resolve()

        return path

    def _load_checkpoint(
        self,
    ) -> None:

        try:

            if not self.checkpoint_path.exists():

                raise FileNotFoundError(
                    "Transformer checkpoint "
                    "not found:\n"
                    f"{self.checkpoint_path}"
                )

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            if (
                not isinstance(
                    checkpoint,
                    dict,
                )
                or "model_state_dict"
                not in checkpoint
            ):

                raise RuntimeError(
                    "Invalid Transformer "
                    "checkpoint format."
                )

            config = checkpoint.get(
                "config",
                {},
            )

            self.seq_len = int(
                config.get(
                    "seq_len",
                    60,
                )
            )

            self.input_dim = int(
                config.get(
                    "num_features",
                    14,
                )
            )

            self.model = (
                DrowsinessTransformer(
                    num_features=self.input_dim,

                    seq_len=self.seq_len,

                    d_model=int(
                        config.get(
                            "d_model",
                            64,
                        )
                    ),

                    lstm_hidden=int(
                        config.get(
                            "lstm_hidden",
                            32,
                        )
                    ),

                    num_heads=int(
                        config.get(
                            "num_heads",
                            4,
                        )
                    ),

                    ff_dim=int(
                        config.get(
                            "ff_dim",
                            128,
                        )
                    ),

                    transformer_layers=int(
                        config.get(
                            "transformer_layers",
                            2,
                        )
                    ),

                    transformer_dropout=float(
                        config.get(
                            "transformer_dropout",
                            0.10,
                        )
                    ),

                    classifier_dropout=float(
                        config.get(
                            "classifier_dropout",
                            0.30,
                        )
                    ),
                )
            ).to(
                self.device
            )

            self.model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            self.model.eval()

            self.feature_mean = (
                np.asarray(
                    checkpoint[
                        "feature_mean"
                    ],
                    dtype=np.float32,
                ).reshape(
                    -1
                )
            )

            self.feature_std = (
                np.asarray(
                    checkpoint[
                        "feature_std"
                    ],
                    dtype=np.float32,
                ).reshape(
                    -1
                )
            )

            if (
                self.feature_mean.shape
                != (
                    self.input_dim,
                )
            ):

                raise RuntimeError(
                    "Invalid feature mean "
                    f"shape: "
                    f"{self.feature_mean.shape}"
                )

            if (
                self.feature_std.shape
                != (
                    self.input_dim,
                )
            ):

                raise RuntimeError(
                    "Invalid feature std "
                    f"shape: "
                    f"{self.feature_std.shape}"
                )

            self.feature_std = np.where(
                self.feature_std < 1e-6,
                1.0,
                self.feature_std,
            ).astype(
                np.float32
            )

            self.decision_threshold = float(
                checkpoint.get(
                    "decision_threshold",
                    0.5,
                )
            )

            self.loaded = True
            self.load_error = None

        except Exception as exc:

            self.loaded = False

            self.load_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            self.model = None

    def _normalize(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:

        if (
            self.feature_mean is None
            or self.feature_std is None
        ):

            raise RuntimeError(
                "Normalization statistics "
                "are not loaded."
            )

        return (
            sequence
            - self.feature_mean[
                None,
                :
            ]
        ) / (
            self.feature_std[
                None,
                :
            ]
        )

    def predict(
        self,
        sequence,
    ) -> dict:

        if (
            not self.loaded
            or self.model is None
        ):

            raise RuntimeError(
                "Transformer model is not "
                "loaded. "
                f"{self.load_error}"
            )

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        expected_shape = (
            self.seq_len,
            self.input_dim,
        )

        if sequence.shape != expected_shape:

            raise ValueError(
                "Expected sequence shape "
                f"{expected_shape}, "
                f"got {sequence.shape}"
            )

        if not np.isfinite(
            sequence
        ).all():

            raise ValueError(
                "Sequence contains "
                "NaN or infinity."
            )

        normalized = self._normalize(
            sequence
        ).astype(
            np.float32
        )

        x = torch.from_numpy(
            normalized
        ).unsqueeze(
            0
        ).to(
            self.device
        )

        with torch.inference_mode():

            logit = self.model(
                x
            )

            probability = (
                torch.sigmoid(
                    logit
                )
                .squeeze()
                .item()
            )

        prediction = int(
            probability
            >= self.decision_threshold
        )

        return {
            "source":
                "transformer",

            "probability":
                float(
                    probability
                ),

            "prediction":
                prediction,

            "label":
                (
                    "drowsy"
                    if prediction == 1
                    else "alert"
                ),

            "threshold":
                float(
                    self.decision_threshold
                ),
        }


transformer_model_service = (
    TransformerModelService()
)