from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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
    load_transformer_data,
)

from transformer_model import (
    DrowsinessTransformer,
)


CHECKPOINT_PATH = (
    BASE_DIR
    / "checkpoints"
    / "best_transformer.pt"
)


OUTPUT_PATH = (
    BASE_DIR
    / "checkpoints"
    / "transformer_detailed_evaluation.json"
)


BATCH_SIZE = 128


def calculate_metrics(
    labels,
    probabilities,
    threshold,
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
        probabilities >= threshold
    ).astype(
        np.int64
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / max(tn + fp, 1)
    )

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(
    labels,
    probabilities,
):

    best_threshold = 0.50
    best_f1 = -1.0
    best_recall = -1.0

    for threshold in np.arange(
        0.01,
        1.00,
        0.01,
    ):

        metrics = calculate_metrics(
            labels,
            probabilities,
            threshold,
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
                    f1 - best_f1
                ) < 1e-12
                and recall > best_recall
            )
        ):

            best_f1 = f1
            best_recall = recall
            best_threshold = float(
                threshold
            )

    return best_threshold


@torch.no_grad()
def predict(
    model,
    dataset,
    device,
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    probabilities = []
    labels = []

    model.eval()

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            device,
            non_blocking=True,
        )

        logits = model(
            X_batch
        )

        probs = torch.sigmoid(
            logits
        )

        probabilities.extend(
            probs.cpu().numpy().tolist()
        )

        labels.extend(
            y_batch.numpy().tolist()
        )

    return (
        np.asarray(
            labels,
            dtype=np.float32,
        ),
        np.asarray(
            probabilities,
            dtype=np.float32,
        ),
    )


def make_indices(
    samples,
    videos,
):

    videos = set(
        videos
    )

    return np.asarray(
        [
            index
            for index, sample
            in enumerate(samples)
            if sample["video"] in videos
        ],
        dtype=np.int64,
    )


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    X, y, meta = (
        load_transformer_data()
    )

    samples = meta[
        "samples"
    ]

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint[
        "config"
    ]

    model = DrowsinessTransformer(
        **config
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    mean = np.asarray(
        checkpoint[
            "feature_mean"
        ],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint[
            "feature_std"
        ],
        dtype=np.float32,
    )

    split_videos = checkpoint[
        "split_videos"
    ]

    val_indices = make_indices(
        samples,
        split_videos[
            "val"
        ],
    )

    test_indices = make_indices(
        samples,
        split_videos[
            "test"
        ],
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

    val_labels, val_probs = predict(
        model,
        val_dataset,
        device,
    )

    test_labels, test_probs = predict(
        model,
        test_dataset,
        device,
    )

    # Search wider than training script:
    # 0.01 ... 0.99
    best_threshold = (
        find_best_threshold(
            val_labels,
            val_probs,
        )
    )

    print(
        "\nCheckpoint threshold:",
        checkpoint.get(
            "decision_threshold"
        ),
    )

    print(
        "Refined validation threshold:",
        f"{best_threshold:.2f}",
    )

    val_metrics = calculate_metrics(
        val_labels,
        val_probs,
        best_threshold,
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_probs,
        best_threshold,
    )

    print(
        "\n============================"
    )
    print(
        "VALIDATION"
    )
    print(
        "============================"
    )

    for key, value in (
        val_metrics.items()
    ):
        print(
            f"{key:12s}: {value}"
        )

    print(
        "\n============================"
    )
    print(
        "TEST"
    )
    print(
        "============================"
    )

    for key, value in (
        test_metrics.items()
    ):
        print(
            f"{key:12s}: {value}"
        )

    test_predictions = (
        test_probs
        >= best_threshold
    ).astype(
        np.int64
    )

    # ------------------------------------
    # ORIGINAL CLASS ANALYSIS
    # ------------------------------------

    print(
        "\n============================"
    )
    print(
        "ORIGINAL CLASS RESULTS"
    )
    print(
        "============================"
    )

    class_results = {}

    test_sample_meta = [
        samples[
            int(index)
        ]
        for index in test_indices
    ]

    for class_name in (
        "alert",
        "drowsy",
        "microsleep",
    ):

        positions = [
            i
            for i, sample
            in enumerate(
                test_sample_meta
            )
            if (
                sample[
                    "original_class"
                ]
                == class_name
            )
        ]

        if not positions:
            continue

        class_probs = test_probs[
            positions
        ]

        class_predictions = (
            test_predictions[
                positions
            ]
        )

        positive_rate = float(
            class_predictions.mean()
        )

        result = {
            "count":
                len(
                    positions
                ),

            "mean_probability":
                float(
                    class_probs.mean()
                ),

            "median_probability":
                float(
                    np.median(
                        class_probs
                    )
                ),

            "min_probability":
                float(
                    class_probs.min()
                ),

            "max_probability":
                float(
                    class_probs.max()
                ),

            "predicted_drowsy_rate":
                positive_rate,
        }

        class_results[
            class_name
        ] = result

        print(
            f"\n{class_name.upper()}"
        )

        for key, value in (
            result.items()
        ):
            print(
                f"{key:24s}: "
                f"{value}"
            )

    # ------------------------------------
    # PER-VIDEO ANALYSIS
    # ------------------------------------

    grouped = defaultdict(
        lambda: {
            "probabilities": [],
            "predictions": [],
            "labels": [],
            "class": None,
        }
    )

    for position, sample in enumerate(
        test_sample_meta
    ):

        video = sample[
            "video"
        ]

        grouped[
            video
        ][
            "probabilities"
        ].append(
            float(
                test_probs[
                    position
                ]
            )
        )

        grouped[
            video
        ][
            "predictions"
        ].append(
            int(
                test_predictions[
                    position
                ]
            )
        )

        grouped[
            video
        ][
            "labels"
        ].append(
            int(
                test_labels[
                    position
                ]
            )
        )

        grouped[
            video
        ][
            "class"
        ] = sample[
            "original_class"
        ]

    print(
        "\n============================"
    )
    print(
        "PER VIDEO"
    )
    print(
        "============================"
    )

    video_results = {}

    for video in sorted(
        grouped
    ):

        data = grouped[
            video
        ]

        probabilities = np.asarray(
            data[
                "probabilities"
            ],
            dtype=np.float32,
        )

        predictions = np.asarray(
            data[
                "predictions"
            ],
            dtype=np.int64,
        )

        result = {
            "class":
                data[
                    "class"
                ],

            "windows":
                int(
                    len(
                        probabilities
                    )
                ),

            "mean_probability":
                float(
                    probabilities.mean()
                ),

            "median_probability":
                float(
                    np.median(
                        probabilities
                    )
                ),

            "min_probability":
                float(
                    probabilities.min()
                ),

            "max_probability":
                float(
                    probabilities.max()
                ),

            "predicted_drowsy_rate":
                float(
                    predictions.mean()
                ),
        }

        video_results[
            video
        ] = result

        print(
            "\n",
            Path(
                video
            ).name,
        )

        print(
            "class:",
            result[
                "class"
            ],
        )

        print(
            "windows:",
            result[
                "windows"
            ],
        )

        print(
            "mean probability:",
            f"{result['mean_probability']:.4f}",
        )

        print(
            "median probability:",
            f"{result['median_probability']:.4f}",
        )

        print(
            "drowsy window rate:",
            f"{result['predicted_drowsy_rate']:.4f}",
        )

    output = {
        "checkpoint_threshold":
            checkpoint.get(
                "decision_threshold"
            ),

        "refined_threshold":
            best_threshold,

        "validation":
            val_metrics,

        "test":
            test_metrics,

        "class_results":
            class_results,

        "video_results":
            video_results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()