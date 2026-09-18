from __future__ import annotations

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):

    def __init__(
        self,
        seq_len: int = 60,
        d_model: int = 64,
    ):
        super().__init__()

        self.position = nn.Parameter(
            torch.zeros(
                1,
                seq_len,
                d_model,
            )
        )

        nn.init.normal_(
            self.position,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        seq_len = x.size(1)

        return (
            x
            + self.position[
                :,
                :seq_len,
                :
            ]
        )


class TemporalAttentionPooling(nn.Module):

    def __init__(
        self,
        d_model: int = 64,
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.Tanh(),
            nn.Linear(
                d_model,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        attention_logits = self.score(
            x
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        pooled = torch.sum(
            attention_weights * x,
            dim=1,
        )

        return pooled


class DrowsinessTransformer(nn.Module):

    def __init__(
        self,
        num_features: int = 14,
        seq_len: int = 60,
        d_model: int = 64,
        lstm_hidden: int = 32,
        num_heads: int = 4,
        ff_dim: int = 128,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.10,
        classifier_dropout: float = 0.30,
    ):
        super().__init__()

        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_norm = nn.LayerNorm(
            num_features
        )

        self.feature_projection = nn.Linear(
            num_features,
            d_model,
        )

        self.positional_encoding = (
            LearnablePositionalEncoding(
                seq_len=seq_len,
                d_model=d_model,
            )
        )

        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        bilstm_output_size = (
            lstm_hidden * 2
        )

        if bilstm_output_size != d_model:
            raise ValueError(
                "BiLSTM output dimension "
                "must equal d_model."
            )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
            )
        )

        self.temporal_pool = (
            TemporalAttentionPooling(
                d_model=d_model
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                d_model,
                64,
            ),

            nn.GELU(),

            nn.Dropout(
                classifier_dropout
            ),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, sequence, features)"
            )

        if x.size(-1) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} "
                f"features, got {x.size(-1)}"
            )

        x = self.input_norm(
            x
        )

        x = self.feature_projection(
            x
        )

        x = self.positional_encoding(
            x
        )

        x, _ = self.bilstm(
            x
        )

        x = self.transformer(
            x
        )

        x = self.temporal_pool(
            x
        )

        logits = self.classifier(
            x
        )

        return logits.squeeze(
            -1
        )

    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return torch.sigmoid(
            self.forward(x)
        )