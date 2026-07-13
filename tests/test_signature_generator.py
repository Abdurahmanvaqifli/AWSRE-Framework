"""
AWSRE Signature Watermark Generator Test
"""

import cv2
import numpy as np

from benchmark.watermark_generator.base_generator import (
    GeneratedWatermark,
    WatermarkType,
)
from benchmark.watermark_generator.signature_generator import (
    SignatureWatermarkGenerator,
    generate_signature_watermark,
)


def run_test():
    print("=" * 72)
    print("AWSRE SIGNATURE WATERMARK GENERATOR TEST")
    print("=" * 72)

    source = np.full(
        (120, 400),
        255,
        dtype=np.uint8,
    )

    points = np.array([
        [10, 85],
        [55, 35],
        [100, 90],
        [160, 25],
        [220, 80],
        [285, 40],
        [350, 85],
        [390, 55],
    ], dtype=np.int32)

    cv2.polylines(
        source,
        [points],
        False,
        0,
        4,
        cv2.LINE_AA,
    )

    generator = SignatureWatermarkGenerator(
        default_seed=42
    )

    result = generator.generate(
        size=(128, 64),
        source=source,
    )

    assert isinstance(
        result,
        GeneratedWatermark,
    )

    assert result.image.shape == (
        64,
        128,
    )

    assert result.image.dtype == np.uint8

    assert set(
        np.unique(result.image)
    ).issubset({0, 1})

    assert (
        result.watermark_type
        == WatermarkType.SIGNATURE.value
    )

    assert np.sum(result.image) > 0
    assert result.metadata[
        "image_category"
    ] == "signature"

    repeated = generator.generate(
        size=(128, 64),
        source=source,
    )

    assert np.array_equal(
        result.image,
        repeated.image,
    )

    wrapper = generate_signature_watermark(
        source=source,
        size=(64, 32),
    )

    assert wrapper.image.shape == (
        32,
        64,
    )

    print(
        "\nSignature density:",
        f"{result.density:.6f}",
    )

    print(
        "Threshold:",
        result.metadata["threshold_value"],
    )

    print(
        "\n✅ SIGNATURE WATERMARK GENERATOR TEST PASSED"
    )


if __name__ == "__main__":
    run_test()
