from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    WeightedRandomSampler,
)


BASE_DIR = Path(
    __file__
).resolve().parents[1]


DATA_DIR = (
    BASE_DIR
    / "datasets"
    / "processed"
)


X_PATH = (
    DATA_DIR
    / "X_transformer.npy"
)


Y_PATH = (
    DATA_DIR
    / "y_transformer.npy"
)


META_PATH = (
    DATA_DIR
    / "meta_transformer.json"
)


def load_transformer_data():

    X = np.load(
        X_PATH
    )

    y = np.load(
        Y_PATH
    )

    with open(
        META_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        meta = json.load(
            file
        )

    samples = meta[
        "samples"
    ]

    if len(X) != len(y):

        raise RuntimeError(
            "X/y size mismatch."
        )

    if len(X) != len(samples):

        raise RuntimeError(
            "Dataset/metadata size mismatch."
        )

    if X.shape[1:] != (
        60,
        14,
    ):

        raise RuntimeError(
            "Expected X shape "
            f"(N, 60, 14), got {X.shape}"
        )

    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(
            "X contains NaN or infinity."
        )

    return (
        X.astype(
            np.float32
        ),
        y.astype(
            np.float32
        ),
        meta,
    )


def split_by_video(
    samples,
    seed: int = 42,
):

    rng = np.random.default_rng(
        seed
    )

    videos_by_class = defaultdict(
        set
    )

    for sample in samples:

        videos_by_class[
            sample[
                "original_class"
            ]
        ].add(
            sample[
                "video"
            ]
        )

    split_videos = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    for class_name in sorted(
        videos_by_class
    ):

        videos = np.asarray(
            sorted(
                videos_by_class[
                    class_name
                ]
            ),
            dtype=object,
        )

        rng.shuffle(
            videos
        )

        n = len(
            videos
        )

        if n < 3:

            raise RuntimeError(
                f"{class_name} needs at "
                "least 3 videos for "
                "train/val/test splitting."
            )

        n_val = max(
            1,
            int(
                round(
                    0.15 * n
                )
            ),
        )

        n_test = max(
            1,
            int(
                round(
                    0.15 * n
                )
            ),
        )

        while (
            n_val
            + n_test
            >= n
        ):

            if n_val > 1:

                n_val -= 1

            elif n_test > 1:

                n_test -= 1

            else:

                raise RuntimeError(
                    "Unable to create "
                    "video split."
                )

        test_videos = videos[
            :n_test
        ]

        val_videos = videos[
            n_test:
            n_test + n_val
        ]

        train_videos = videos[
            n_test + n_val:
        ]

        split_videos[
            "train"
        ].update(
            train_videos.tolist()
        )

        split_videos[
            "val"
        ].update(
            val_videos.tolist()
        )

        split_videos[
            "test"
        ].update(
            test_videos.tolist()
        )

        print(
            f"{class_name:12s} "
            f"train={len(train_videos)} "
            f"val={len(val_videos)} "
            f"test={len(test_videos)}"
        )

    if (
        split_videos["train"]
        & split_videos["val"]
    ):

        raise RuntimeError(
            "Train/val video leakage."
        )

    if (
        split_videos["train"]
        & split_videos["test"]
    ):

        raise RuntimeError(
            "Train/test video leakage."
        )

    if (
        split_videos["val"]
        & split_videos["test"]
    ):

        raise RuntimeError(
            "Val/test video leakage."
        )

    train_indices = []

    val_indices = []

    test_indices = []

    for index, sample in enumerate(
        samples
    ):

        video = sample[
            "video"
        ]

        if video in split_videos[
            "train"
        ]:

            train_indices.append(
                index
            )

        elif video in split_videos[
            "val"
        ]:

            val_indices.append(
                index
            )

        elif video in split_videos[
            "test"
        ]:

            test_indices.append(
                index
            )

        else:

            raise RuntimeError(
                f"Video not assigned: "
                f"{video}"
            )

    return (
        np.asarray(
            train_indices,
            dtype=np.int64,
        ),
        np.asarray(
            val_indices,
            dtype=np.int64,
        ),
        np.asarray(
            test_indices,
            dtype=np.int64,
        ),
        split_videos,
    )


def compute_train_normalization(
    X: np.ndarray,
    train_indices: np.ndarray,
):

    train_values = X[
        train_indices
    ].reshape(
        -1,
        X.shape[-1],
    )

    mean = train_values.mean(
        axis=0,
        dtype=np.float64,
    )

    std = train_values.std(
        axis=0,
        dtype=np.float64,
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std,
    )

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


class TransformerSequenceDataset(
    Dataset
):

    def __init__(
        self,
        X,
        y,
        indices,
        mean,
        std,
    ):

        self.X = X
        self.y = y

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.mean = np.asarray(
            mean,
            dtype=np.float32,
        )

        self.std = np.asarray(
            std,
            dtype=np.float32,
        )

    def __len__(
        self,
    ):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        item,
    ):

        index = self.indices[
            item
        ]

        x = self.X[
            index
        ]

        x = (
            x
            - self.mean
        ) / self.std

        y = self.y[
            index
        ]

        return (
            torch.from_numpy(
                x.astype(
                    np.float32,
                    copy=False,
                )
            ),
            torch.tensor(
                y,
                dtype=torch.float32,
            ),
        )


def create_balanced_sampler(
    samples,
    train_indices,
):

    class_counts = Counter(
        samples[
            int(index)
        ][
            "original_class"
        ]

        for index
        in train_indices
    )

    print(
        "\nTraining original-class counts:"
    )

    for (
        class_name,
        count,
    ) in sorted(
        class_counts.items()
    ):

        print(
            f"{class_name:12s}: "
            f"{count}"
        )

    # We want:
    #
    # alert      -> 50% total probability
    #
    # drowsy     -> 25%
    # microsleep -> 25%
    #
    # This gives binary balance while also
    # preventing microsleep from dominating
    # ordinary drowsiness.

    target_mass = {
        "alert": 0.50,
        "drowsy": 0.25,
        "microsleep": 0.25,
    }

    weights = []

    for index in train_indices:

        class_name = samples[
            int(index)
        ][
            "original_class"
        ]

        if class_name not in target_mass:

            raise RuntimeError(
                "Unexpected class: "
                f"{class_name}"
            )

        count = class_counts[
            class_name
        ]

        sample_weight = (
            target_mass[
                class_name
            ]
            / count
        )

        weights.append(
            sample_weight
        )

    weights = torch.tensor(
        weights,
        dtype=torch.double,
    )

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(
            train_indices
        ),
        replacement=True,
    )


def summarize_split(
    name,
    indices,
    y,
    samples,
):

    binary = Counter(
        int(
            y[
                int(index)
            ]
        )

        for index in indices
    )

    original = Counter(
        samples[
            int(index)
        ][
            "original_class"
        ]

        for index in indices
    )

    videos = set(
        samples[
            int(index)
        ][
            "video"
        ]

        for index in indices
    )

    print(
        f"\n{name.upper()}"
    )

    print(
        f"samples: {len(indices)}"
    )

    print(
        f"videos : {len(videos)}"
    )

    print(
        "binary :",
        dict(
            sorted(
                binary.items()
            )
        ),
    )

    print(
        "classes:",
        dict(
            sorted(
                original.items()
            )
        ),
    )