"""
AWSRE Attack Pipeline Test
"""

import numpy as np

from attacks.pipeline import (
    AttackPipeline,
    AttackResult,
    AttackStep,
    SUPPORTED_ATTACKS,
)


def run_test():
    print("=" * 72)
    print("AWSRE ATTACK PIPELINE TEST")
    print("=" * 72)

    rng = np.random.default_rng(
        seed=2026
    )

    grayscale = rng.integers(
        0,
        256,
        size=(256, 256),
        dtype=np.uint8,
    )

    color = rng.integers(
        0,
        256,
        size=(256, 256, 3),
        dtype=np.uint8,
    )

    pipeline = AttackPipeline(
        default_seed=42
    )

    cases = [
        ("none", None),
        ("jpeg", 70),
        ("gaussian_noise", 0.01),
        ("salt_pepper", 0.05),
        ("gaussian_blur", 3),
        ("rotation", 5),
        ("crop", 0.90),
        ("brightness", 20),
        ("brightness", -20),
        ("contrast", 20),
        ("contrast", -20),
        ("gamma", 1.2),
        ("sharpen", 1.0),
    ]

    for source in (
        grayscale,
        color,
    ):
        for attack_type, parameter in cases:
            result = pipeline.apply(
                source,
                attack_type,
                parameter,
            )

            assert isinstance(
                result,
                AttackResult,
            )

            assert result.image.shape == (
                source.shape
            )

            assert result.image.dtype == np.uint8
            assert result.runtime_seconds >= 0
            assert len(result.steps) == 1

    first_noise = pipeline.apply(
        grayscale,
        "gaussian_noise",
        0.01,
    )

    second_noise = pipeline.apply(
        grayscale,
        "gaussian_noise",
        0.01,
    )

    assert np.array_equal(
        first_noise.image,
        second_noise.image,
    )

    first_salt_pepper = pipeline.apply(
        grayscale,
        "salt_pepper",
        0.05,
    )

    second_salt_pepper = pipeline.apply(
        grayscale,
        "salt_pepper",
        0.05,
    )

    assert np.array_equal(
        first_salt_pepper.image,
        second_salt_pepper.image,
    )

    combined = pipeline.apply_combined(
        grayscale,
        [
            AttackStep(
                "jpeg",
                70,
            ),
            AttackStep(
                "gaussian_blur",
                3,
            ),
            AttackStep(
                "gaussian_noise",
                0.01,
            ),
        ],
    )

    assert isinstance(
        combined,
        AttackResult,
    )

    assert combined.attack_type == "combined"
    assert combined.image.shape == grayscale.shape
    assert len(combined.steps) == 3

    assert set(
        pipeline.supported_attacks()
    ) == set(
        SUPPORTED_ATTACKS
    )

    print(
        "\nIndividual cases:",
        len(cases),
    )

    print(
        "Image modes tested: grayscale and color"
    )

    print(
        "Combined steps:",
        len(combined.steps),
    )

    print(
        "Deterministic noise: verified"
    )

    print(
        "\n✅ ATTACK PIPELINE TEST PASSED"
    )


if __name__ == "__main__":
    run_test()
