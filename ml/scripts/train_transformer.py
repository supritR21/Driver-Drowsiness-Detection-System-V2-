from __future__ import annotations

import json
import random

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from transformer_dataset import (
    BASE_DIR,
    TransformerSequenceDataset,
    compute_train_normalization,
    create_balanced_sampler,
    load_transformer_data,
    split_by_video,
    summarize_split,
)

from transformer_model import (
    DrowsinessTransformer,
)


# =========================================================
# CONFIG
# =========================================================

SEED = 42

BATCH_SIZE = 64

EPOCHS = 40

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 7

MIN_DELTA = 1e-4

GRADIENT_CLIP = 1.0


CHECKPOINT_DIR = (
    BASE_DIR
    / "checkpoints"
)


CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "best_transformer.pt"
)


NORMALIZATION_MEAN_PATH = (
    CHECKPOINT_DIR
    / "transformer_feature_mean.npy"
)


NORMALIZATION_STD_PATH = (
    CHECKPOINT_DIR
    / "transformer_feature_std.npy"
)


SPLIT_PATH = (
    CHECKPOINT_DIR
    / "transformer_split.json"
)


METRICS_PATH = (
    CHECKPOINT_DIR
    / "transformer_metrics.json"
)


# =========================================================
# RANDOM SEED
# =========================================================

def set_seed(
    seed: int,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold=0.5,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn
        / max(
            tn + fp,
            1,
        )
    )

    sensitivity = (
        tp
        / max(
            tp + fn,
            1,
        )
    )

    metrics = {
        "threshold":
            float(
                threshold
            ),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "sensitivity":
            float(
                sensitivity
            ),

        "specificity":
            float(
                specificity
            ),

        "f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "confusion_matrix": {
            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp),
        },
    }

    if len(
        np.unique(
            labels
        )
    ) == 2:

        metrics[
            "roc_auc"
        ] = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

        metrics[
            "pr_auc"
        ] = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )

    else:

        metrics[
            "roc_auc"
        ] = None

        metrics[
            "pr_auc"
        ] = None

    return metrics


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
):

    model.train()

    total_loss = 0.0

    total_samples = 0


    amp_enabled = (
        device.type
        == "cuda"
    )


    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            device,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        autocast_context = (
            torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
            )

            if amp_enabled

            else nullcontext()
        )


        with autocast_context:

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )


        if amp_enabled:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            optimizer.step()


        batch_size = (
            X_batch.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )


    return (
        total_loss
        / max(
            total_samples,
            1,
        )
    )


# =========================================================
# EVALUATION
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    all_labels = []

    all_probabilities = []


    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            device,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch,
        )


        probabilities = torch.sigmoid(
            logits
        )


        batch_size = (
            X_batch.size(0)
        )


        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )


        all_labels.extend(
            y_batch
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )


        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )


    average_loss = (
        total_loss
        / max(
            total_samples,
            1,
        )
    )


    return (
        average_loss,
        np.asarray(
            all_labels,
            dtype=np.float32,
        ),
        np.asarray(
            all_probabilities,
            dtype=np.float32,
        ),
    )


# =========================================================
# BEST VALIDATION THRESHOLD
# =========================================================

def find_best_threshold(
    labels,
    probabilities,
):

    best_threshold = 0.5

    best_f1 = -1.0

    best_recall = -1.0


    for threshold in np.arange(
        0.10,
        0.901,
        0.01,
    ):

        metrics = calculate_metrics(
            labels,
            probabilities,
            threshold=float(
                threshold
            ),
        )

        f1 = metrics[
            "f1"
        ]

        recall = metrics[
            "recall"
        ]


        if (
            f1 > best_f1
            or (
                abs(
                    f1
                    - best_f1
                )
                < 1e-12
                and recall
                > best_recall
            )
        ):

            best_f1 = (
                f1
            )

            best_recall = (
                recall
            )

            best_threshold = float(
                threshold
            )


    return best_threshold


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(
        SEED
    )


    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "\n=============================="
    )

    print(
        "TRANSFORMER TRAINING"
    )

    print(
        "=============================="
    )


    print(
        "Device:",
        device,
    )


    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )


    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    X, y, meta = (
        load_transformer_data()
    )


    samples = meta[
        "samples"
    ]


    print(
        "\nDataset:",
        X.shape,
    )


    # -----------------------------------------------------
    # VIDEO-level split
    # -----------------------------------------------------

    (
        train_indices,
        val_indices,
        test_indices,
        split_videos,
    ) = split_by_video(
        samples,
        seed=SEED,
    )


    summarize_split(
        "train",
        train_indices,
        y,
        samples,
    )


    summarize_split(
        "validation",
        val_indices,
        y,
        samples,
    )


    summarize_split(
        "test",
        test_indices,
        y,
        samples,
    )


    # -----------------------------------------------------
    # Train-only normalization
    # -----------------------------------------------------

    mean, std = (
        compute_train_normalization(
            X,
            train_indices,
        )
    )


    np.save(
        NORMALIZATION_MEAN_PATH,
        mean,
    )


    np.save(
        NORMALIZATION_STD_PATH,
        std,
    )


    print(
        "\nNormalization saved."
    )


    for index, name in enumerate(
        meta[
            "feature_names"
        ]
    ):

        print(
            f"{index + 1:02d}. "
            f"{name:12s} "
            f"mean={mean[index]:10.4f} "
            f"std={std[index]:10.4f}"
        )


    # -----------------------------------------------------
    # Save split manifest
    # -----------------------------------------------------

    split_payload = {
        name: sorted(
            list(
                videos
            )
        )

        for (
            name,
            videos,
        ) in split_videos.items()
    }


    with open(
        SPLIT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            split_payload,
            file,
            indent=2,
        )


    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = (
        TransformerSequenceDataset(
            X,
            y,
            train_indices,
            mean,
            std,
        )
    )


    val_dataset = (
        TransformerSequenceDataset(
            X,
            y,
            val_indices,
            mean,
            std,
        )
    )


    test_dataset = (
        TransformerSequenceDataset(
            X,
            y,
            test_indices,
            mean,
            std,
        )
    )


    # -----------------------------------------------------
    # Balanced training sampler
    # -----------------------------------------------------

    train_sampler = (
        create_balanced_sampler(
            samples,
            train_indices,
        )
    )


    pin_memory = (
        device.type
        == "cuda"
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = DrowsinessTransformer(
        num_features=14,
        seq_len=60,
        d_model=64,
        lstm_hidden=32,
        num_heads=4,
        ff_dim=128,
        transformer_layers=2,
        transformer_dropout=0.10,
        classifier_dropout=0.30,
    ).to(
        device
    )


    total_parameters = sum(
        parameter.numel()

        for parameter
        in model.parameters()
    )


    print(
        "\nModel parameters:",
        total_parameters,
    )


    # -----------------------------------------------------
    # Loss
    #
    # Sampler already handles imbalance,
    # so no pos_weight here.
    # -----------------------------------------------------

    criterion = (
        nn.BCEWithLogitsLoss()
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )


    amp_enabled = (
        device.type
        == "cuda"
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )


    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    patience_counter = 0


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
            )
        )


        (
            val_loss,
            val_labels,
            val_probabilities,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )


        scheduler.step(
            val_loss
        )


        val_metrics = (
            calculate_metrics(
                val_labels,
                val_probabilities,
                threshold=0.5,
            )
        )


        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )


        print(
            f"\nEpoch "
            f"{epoch:02d}/{EPOCHS}"
        )


        print(
            f"train_loss="
            f"{train_loss:.5f} "
            f"val_loss="
            f"{val_loss:.5f} "
            f"lr="
            f"{current_lr:.2e}"
        )


        print(
            f"val_precision="
            f"{val_metrics['precision']:.4f} "
            f"val_recall="
            f"{val_metrics['recall']:.4f} "
            f"val_f1="
            f"{val_metrics['f1']:.4f} "
            f"val_auc="
            f"{val_metrics['roc_auc']:.4f}"
        )


        improved = (
            val_loss
            < (
                best_val_loss
                - MIN_DELTA
            )
        )


        if improved:

            best_val_loss = (
                val_loss
            )

            patience_counter = 0


            checkpoint = {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_loss":
                    val_loss,

                "feature_names":
                    meta[
                        "feature_names"
                    ],

                "feature_mean":
                    mean,

                "feature_std":
                    std,

                "config": {
                    "num_features":
                        14,

                    "seq_len":
                        60,

                    "d_model":
                        64,

                    "lstm_hidden":
                        32,

                    "num_heads":
                        4,

                    "ff_dim":
                        128,

                    "transformer_layers":
                        2,

                    "transformer_dropout":
                        0.10,

                    "classifier_dropout":
                        0.30,
                },

                "split_videos":
                    split_payload,
            }


            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )


            print(
                "✓ Saved best checkpoint"
            )


        else:

            patience_counter += 1


            print(
                "No improvement. "
                f"Patience "
                f"{patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )


            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "\nEarly stopping."
                )

                break


    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    print(
        "\nLoading best checkpoint..."
    )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    # =====================================================
    # VALIDATION THRESHOLD
    # =====================================================

    (
        best_val_loss,
        val_labels,
        val_probabilities,
    ) = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )


    best_threshold = (
        find_best_threshold(
            val_labels,
            val_probabilities,
        )
    )


    validation_metrics = (
        calculate_metrics(
            val_labels,
            val_probabilities,
            threshold=best_threshold,
        )
    )


    print(
        "\nBest validation threshold:",
        f"{best_threshold:.2f}"
    )


    # =====================================================
    # TEST SET
    # =====================================================

    (
        test_loss,
        test_labels,
        test_probabilities,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )


    test_metrics = (
        calculate_metrics(
            test_labels,
            test_probabilities,
            threshold=best_threshold,
        )
    )


    test_metrics[
        "loss"
    ] = float(
        test_loss
    )


    # =====================================================
    # SAVE THRESHOLD INTO CHECKPOINT
    # =====================================================

    checkpoint[
        "decision_threshold"
    ] = best_threshold


    checkpoint[
        "validation_metrics"
    ] = validation_metrics


    checkpoint[
        "test_metrics"
    ] = test_metrics


    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )


    # =====================================================
    # SAVE HUMAN-READABLE METRICS
    # =====================================================

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "best_epoch":
                    checkpoint[
                        "epoch"
                    ],

                "best_validation_loss":
                    float(
                        checkpoint[
                            "val_loss"
                        ]
                    ),

                "decision_threshold":
                    best_threshold,

                "validation":
                    validation_metrics,

                "test":
                    test_metrics,
            },
            file,
            indent=2,
        )


    # =====================================================
    # FINAL REPORT
    # =====================================================

    print(
        "\n================================"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "================================"
    )


    print(
        f"Threshold   : "
        f"{best_threshold:.2f}"
    )


    print(
        f"Test loss   : "
        f"{test_loss:.5f}"
    )


    print(
        f"Accuracy    : "
        f"{test_metrics['accuracy']:.4f}"
    )


    print(
        f"Precision   : "
        f"{test_metrics['precision']:.4f}"
    )


    print(
        f"Recall      : "
        f"{test_metrics['recall']:.4f}"
    )


    print(
        f"Specificity : "
        f"{test_metrics['specificity']:.4f}"
    )


    print(
        f"F1          : "
        f"{test_metrics['f1']:.4f}"
    )


    print(
        f"ROC-AUC     : "
        f"{test_metrics['roc_auc']:.4f}"
    )


    print(
        f"PR-AUC      : "
        f"{test_metrics['pr_auc']:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        test_metrics[
            "confusion_matrix"
        ]
    )


    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


if __name__ == "__main__":
    main()