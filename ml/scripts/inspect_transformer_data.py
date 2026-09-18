from __future__ import annotations

import json

from pathlib import Path

import numpy as np


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


print(
    "Loading dataset..."
)


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


feature_names = meta[
    "feature_names"
]


print("\n")
print(
    "================================"
)

print(
    "TRANSFORMER DATASET INSPECTION"
)

print(
    "================================"
)


print(
    "\nX shape:",
    X.shape,
)


print(
    "y shape:",
    y.shape,
)


print(
    "X dtype:",
    X.dtype,
)


print(
    "y dtype:",
    y.dtype,
)


print(
    "\nLabels:"
)


unique_labels, counts = (
    np.unique(
        y,
        return_counts=True,
    )
)


for (
    label,
    count,
) in zip(
    unique_labels,
    counts,
):

    print(
        f"Label {int(label)}: "
        f"{count} samples"
    )


print(
    "\nFinite X:",
    bool(
        np.isfinite(
            X
        ).all()
    ),
)


print(
    "Sequence length:",
    meta[
        "sequence_length"
    ],
)


print(
    "Target FPS:",
    meta[
        "target_fps"
    ],
)


print(
    "Nominal temporal span:",
    meta[
        "sequence_length"
    ]
    /
    meta[
        "target_fps"
    ],
    "seconds",
)


# ---------------------------------------
# Shape validation
# ---------------------------------------

if X.ndim != 3:

    raise SystemExit(
        "ERROR: X should have "
        "3 dimensions"
    )


if X.shape[1:] != (
    60,
    14,
):

    raise SystemExit(
        "ERROR: expected "
        f"(N, 60, 14), "
        f"got {X.shape}"
    )


# ---------------------------------------
# NaN / infinity
# ---------------------------------------

if not np.isfinite(
    X
).all():

    raise SystemExit(
        "ERROR: X contains "
        "NaN or infinity"
    )


# ---------------------------------------
# Binary labels
# ---------------------------------------

labels = set(
    np.unique(
        y
    ).tolist()
)


if not labels.issubset(
    {
        0.0,
        1.0,
    }
):

    raise SystemExit(
        "ERROR: labels are "
        f"not binary: {labels}"
    )


# ---------------------------------------
# Feature statistics
# ---------------------------------------

flat = X.reshape(
    -1,
    X.shape[-1],
)


print(
    "\n================================"
)

print(
    "PER-FEATURE STATISTICS"
)

print(
    "================================"
)


for (
    index,
    name,
) in enumerate(
    feature_names
):

    values = flat[
        :,
        index,
    ]

    print(
        f"{index + 1:02d}. "
        f"{name:12s} "
        f"min={values.min():10.4f} "
        f"max={values.max():10.4f} "
        f"mean={values.mean():10.4f} "
        f"std={values.std():10.4f}"
    )


print("\n")
print(
    "================================"
)

print(
    "STAGE 1 SUCCESS"
)

print(
    "================================"
)


print(
    "Transformer preprocessing "
    "output has the expected "
    "contract."
)