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

        self.num_features = (
            num_features
        )

        self.seq_len = (
            seq_len
        )

        self.d_model = (
            d_model
        )

        # ---------------------------------
        # Input normalization
        # ---------------------------------

        self.input_norm = nn.LayerNorm(
            num_features
        )

        # ---------------------------------
        # 14 → 64 projection
        # ---------------------------------

        self.feature_projection = nn.Linear(
            num_features,
            d_model,
        )

        # ---------------------------------
        # Learnable position information
        # ---------------------------------

        self.positional_encoding = (
            LearnablePositionalEncoding(
                seq_len=seq_len,
                d_model=d_model,
            )
        )

        # ---------------------------------
        # BiLSTM
        #
        # hidden = 32 per direction
        #
        # output:
        # 32 + 32 = 64
        # ---------------------------------

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

        if (
            bilstm_output_size
            != d_model
        ):
            raise ValueError(
                "BiLSTM output dimension "
                "must equal d_model."
            )

        # ---------------------------------
        # Transformer encoder layer
        # ---------------------------------

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

        # ---------------------------------
        # Transformer × 2
        # ---------------------------------

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
            )
        )

        # ---------------------------------
        # Temporal attention
        # ---------------------------------

        self.temporal_pool = (
            TemporalAttentionPooling(
                d_model=d_model
            )
        )

        # ---------------------------------
        # Classifier
        # ---------------------------------

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

        # Expected:
        #
        # (batch, 60, 14)

        if x.ndim != 3:

            raise ValueError(
                "Expected input shape "
                "(batch, sequence, features)"
            )

        if x.size(-1) != (
            self.num_features
        ):

            raise ValueError(
                f"Expected "
                f"{self.num_features} "
                f"features, got "
                f"{x.size(-1)}"
            )

        # ---------------------------------
        # LayerNorm
        # ---------------------------------

        x = self.input_norm(
            x
        )

        # ---------------------------------
        # Dense 14 → 64
        # ---------------------------------

        x = self.feature_projection(
            x
        )

        # ---------------------------------
        # Positional encoding
        # ---------------------------------

        x = self.positional_encoding(
            x
        )

        # ---------------------------------
        # BiLSTM
        # ---------------------------------

        x, _ = self.bilstm(
            x
        )

        # ---------------------------------
        # Transformer ×2
        # ---------------------------------

        x = self.transformer(
            x
        )

        # ---------------------------------
        # Temporal attention pooling
        #
        # (B,60,64) → (B,64)
        # ---------------------------------

        x = self.temporal_pool(
            x
        )

        # ---------------------------------
        # Final logit
        # ---------------------------------

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

        logits = self.forward(
            x
        )

        return torch.sigmoid(
            logits
        )


if __name__ == "__main__":

    model = DrowsinessTransformer()

    dummy = torch.randn(
        8,
        60,
        14,
    )

    logits = model(
        dummy
    )

    probabilities = (
        model.predict_proba(
            dummy
        )
    )

    print(
        "Input shape:",
        dummy.shape,
    )

    print(
        "Logit shape:",
        logits.shape,
    )

    print(
        "Probability shape:",
        probabilities.shape,
    )

    print(
        "Probability min:",
        probabilities.min().item(),
    )

    print(
        "Probability max:",
        probabilities.max().item(),
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Total parameters:",
        total_parameters,
    )

    print(
        "Trainable parameters:",
        trainable_parameters,
    )