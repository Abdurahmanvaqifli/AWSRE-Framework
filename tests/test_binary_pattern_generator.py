"""
AWSRE Binary Pattern Generator Test
"""

import numpy as np

from benchmark.watermark_generator.base_generator import (
    GeneratedWatermark,
    WatermarkType,
)
from benchmark.watermark_generator.binary_pattern import (
    BinaryPatternGenerator,
    BinaryPatternType,
    generate_binary_pattern,
)


def run_test():
    print("=" * 72)
    print("AWSRE BINARY PATTERN GENERATOR TEST")
    print("=" * 72)

    generator = BinaryPatternGenerator(
        default_seed=2026
    )

    generated_results = {}

    for pattern in BinaryPatternType:
        result = generator.generate(
            size=(32, 32),
            pattern=pattern,
            seed=2026,
            density=0.5,
            thickness=2,
        )

        assert isinstance(
            result,
            GeneratedWatermark,
        )

        assert result.image.shape == (
            32,
            32,
        )

        assert result.image.dtype == np.uint8

        assert set(
            np.unique(result.image)
        ).issubset({0, 1})

        assert (
            result.watermark_type
            == WatermarkType.BINARY_PATTERN.value
        )

        assert result.payload_bits == 1024
        assert len(result.file_hash) == 64

        generated_results[
            pattern.value
        ] = result

    # Reproducibility
    random_1 = generator.generate(
        size=(64, 64),
        pattern="random",
        seed=42,
        density=0.35,
    )

    random_2 = generator.generate(
        size=(64, 64),
        pattern="random",
        seed=42,
        density=0.35,
    )

    random_3 = generator.generate(
        size=(64, 64),
        pattern="random",
        seed=43,
        density=0.35,
    )

    assert np.array_equal(
        random_1.image,
        random_2.image,
    )

    assert (
        random_1.file_hash
        == random_2.file_hash
    )

    assert not np.array_equal(
        random_1.image,
        random_3.image,
    )

    # Density should be approximately requested value.
    assert abs(
        random_1.density - 0.35
    ) < 0.05

    # Inversion
    normal = generator.generate(
        size=(32, 32),
        pattern="checkerboard",
        invert=False,
    )

    inverted = generator.generate(
        size=(32, 32),
        pattern="checkerboard",
        invert=True,
    )

    assert np.array_equal(
        inverted.image,
        1 - normal.image,
    )

    # Functional wrapper
    wrapper_result = generate_binary_pattern(
        size=(32, 32),
        pattern="cross",
        seed=42,
        thickness=3,
    )

    assert isinstance(
        wrapper_result,
        GeneratedWatermark,
    )

    assert wrapper_result.image.shape == (
        32,
        32,
    )

    print(
        "\nPatterns tested:",
        len(generated_results),
    )

    print(
        "Random requested density: 0.35"
    )

    print(
        "Random actual density:",
        f"{random_1.density:.4f}",
    )

    print(
        "Reproducibility verified:",
        random_1.file_hash,
    )

    print(
        "\n✅ BINARY PATTERN GENERATOR TEST PASSED"
    )


if __name__ == "__main__":
    run_test()
