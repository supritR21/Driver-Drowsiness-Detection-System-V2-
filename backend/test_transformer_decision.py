from app.services.transformer_decision_state import (
    TransformerDecisionStore,
)


def show(
    name,
    probabilities,
):

    print(
        "\n============================"
    )

    print(
        name
    )

    print(
        "============================"
    )

    store = (
        TransformerDecisionStore()
    )

    session_id = (
        "test-session"
    )

    threshold = 0.01

    for i, probability in enumerate(
        probabilities,
        start=1,
    ):

        result = store.update(
            session_id=session_id,
            raw_probability=probability,
            threshold=threshold,
        )

        print(
            f"{i:02d} "
            f"raw={probability:.4f} "
            f"ema="
            f"{result['ema_probability']:.4f} "
            f"vote_ratio="
            f"{result['vote_ratio']:.2f} "
            f"prediction="
            f"{result['prediction']:7s} "
            f"level="
            f"{result['level']}"
        )


def main():

    show(
        "ALERT STABLE",
        [
            0.0005,
            0.0006,
            0.0005,
            0.0007,
            0.0004,
        ],
    )

    show(
        "BECOMING DROWSY",
        [
            0.0005,
            0.0006,
            0.9980,
            0.9970,
            0.9990,
            0.9980,
        ],
    )

    show(
        "RECOVERY",
        [
            0.9980,
            0.9990,
            0.9980,
            0.0005,
            0.0006,
            0.0005,
            0.0004,
            0.0005,
        ],
    )


if __name__ == "__main__":
    main()