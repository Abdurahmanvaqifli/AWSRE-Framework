"""
AWSRE Logo Watermark Generator Test
"""

import cv2
import numpy as np

from benchmark.watermark_generator.base_generator import (
    GeneratedWatermark,
    WatermarkType,
)
from benchmark.watermark_generator.logo_generator import (
    LogoWatermarkGenerator,
    generate_logo_watermark,
)


def run_test():
    print("=" * 72)
    print("AWSRE LOGO WATERMARK GENERATOR TEST")
    print("=" * 72)

    source = np.full(
        (200, 300, 3),
        255,
        dtype=np.uint8,
    )

    cv2.rectangle(
        source,
        (40, 30),
        (260, 170),
        (0, 0, 0),
        thickness=8,
    )

    cv2.putText(
        source,
        "AWSRE",
        (55, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    generator = LogoWatermarkGenerator(
        default_seed=42
    )

    result = generator.generate(
        size=(64, 64),
        source=source,
    )

    assert isinstance(
        result,
        GeneratedWatermark,
    )

    assert result.image.shape == (
        64,
        64,
    )

    assert result.image.dtype == np.uint8

    assert set(
        np.unique(result.image)
    ).issubset({0, 1})

    assert (
        result.watermark_type
        == WatermarkType.LOGO.value
    )

    assert np.sum(result.image) > 0
    assert len(result.file_hash) == 64

    repeated = generator.generate(
        size=(64, 64),
        source=source,
    )

    assert np.array_equal(
        result.image,
        repeated.image,
    )

    wrapper = generate_logo_watermark(
        source=source,
        size=(32, 32),
    )

    assert wrapper.image.shape == (
        32,
        32,
    )

    print(
        "\nLogo density:",
        f"{result.density:.6f}",
    )

    print(
        "Source dimensions:",
        result.metadata["source_width"],
        "x",
        result.metadata["source_height"],
    )

    print(
        "\n✅ LOGO WATERMARK GENERATOR TEST PASSED"
    )


if __name__ == "__main__":
    run_test()
