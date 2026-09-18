from pathlib import Path

import numpy as np

from app.services.transformer_model_service import (
    transformer_model_service,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


X_PATH = (
    ROOT
    / "ml"
    / "datasets"
    / "processed"
    / "X_transformer.npy"
)


Y_PATH = (
    ROOT
    / "ml"
    / "datasets"
    / "processed"
    / "y_transformer.npy"
)


def main():

    service = (
        transformer_model_service
    )

    print(
        "================================"
    )

    print(
        "TRANSFORMER BACKEND SMOKE TEST"
    )

    print(
        "================================"
    )

    print(
        "Device:",
        service.device,
    )

    print(
        "Checkpoint:",
        service.checkpoint_path,
    )

    print(
        "Loaded:",
        service.loaded,
    )

    print(
        "Load error:",
        service.load_error,
    )

    print(
        "Sequence length:",
        service.seq_len,
    )

    print(
        "Input dimension:",
        service.input_dim,
    )

    print(
        "Threshold:",
        service.decision_threshold,
    )


    if not service.loaded:

        raise SystemExit(
            "ERROR: Transformer "
            "checkpoint did not load."
        )


    X = np.load(
        X_PATH
    )

    y = np.load(
        Y_PATH
    )


    print(
        "\nDataset:",
        X.shape,
    )


    for target in (
        0,
        1,
    ):

        indices = np.where(
            y == target
        )[0]

        if len(indices) == 0:
            continue

        index = int(
            indices[0]
        )

        result = service.predict(
            X[
                index
            ]
        )

        print(
            "\nExpected label:",
            int(
                y[
                    index
                ]
            ),
        )

        print(
            "Dataset index:",
            index,
        )

        print(
            "Result:",
            result,
        )


if __name__ == "__main__":
    main()