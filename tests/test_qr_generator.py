"""
AWSRE QR Code Watermark Generator Test
"""

import numpy as np

from benchmark.watermark_generator.base_generator import (
    GeneratedWatermark,
    WatermarkType,
)
from benchmark.watermark_generator.qr_generator import (
    QRErrorCorrection,
    QRCodeWatermarkGenerator,
    generate_qr_watermark,
)


def run_test():
    print("=" * 72)
    print("AWSRE QR CODE WATERMARK GENERATOR TEST")
    print("=" * 72)

    generator = QRCodeWatermarkGenerator(
        default_seed=2026
    )

    results = {}

    for correction in QRErrorCorrection:
        result = generator.generate(
            size=(128, 128),
            text="AWSRE-TEST-2026",
            error_correction=correction,
            border=2,
        )

        assert isinstance(
            result,
            GeneratedWatermark,
        )

        assert result.image.shape == (
            128,
            128,
        )

        assert result.image.dtype == np.uint8

        assert set(
            np.unique(result.image)
        ).issubset({0, 1})

        assert (
            result.watermark_type
            == WatermarkType.QR_CODE.value
        )

        assert result.payload_bits == (
            128 * 128
        )

        assert len(
            result.file_hash
        ) == 64

        assert result.metadata[
            "error_correction"
        ] == correction.value

        assert result.metadata[
            "module_count"
        ] > 0

        results[
            correction.value
        ] = result

    # Reproducible random payload.
    first = generator.generate(
        size=(128, 128),
        random_length=24,
        seed=42,
        error_correction="Q",
    )

    second = generator.generate(
        size=(128, 128),
        random_length=24,
        seed=42,
        error_correction="Q",
    )

    third = generator.generate(
        size=(128, 128),
        random_length=24,
        seed=43,
        error_correction="Q",
    )

    assert (
        first.metadata["encoded_text"]
        == second.metadata["encoded_text"]
    )

    assert np.array_equal(
        first.image,
        second.image,
    )

    assert (
        first.file_hash
        == second.file_hash
    )

    assert (
        first.metadata["encoded_text"]
        != third.metadata["encoded_text"]
    )

    # Inversion.
    normal = generator.generate(
        size=(64, 64),
        text="AWSRE",
        invert=False,
    )

    inverted = generator.generate(
        size=(64, 64),
        text="AWSRE",
        invert=True,
    )

    assert np.array_equal(
        inverted.image,
        1 - normal.image,
    )

    # Non-square output.
    rectangular = generator.generate(
        size=(128, 64),
        text="AWSRE-RECTANGLE",
        preserve_square=True,
    )

    assert rectangular.image.shape == (
        64,
        128,
    )

    # Functional wrapper.
    wrapper_result = generate_qr_watermark(
        size=(64, 64),
        text="AWSRE",
        seed=42,
        error_correction="M",
    )

    assert isinstance(
        wrapper_result,
        GeneratedWatermark,
    )

    assert wrapper_result.image.shape == (
        64,
        64,
    )

    print(
        "\nError-correction levels tested:",
        len(results),
    )

    print(
        "Random encoded text:",
        first.metadata["encoded_text"],
    )

    print(
        "Wrapper QR version:",
        wrapper_result.metadata["actual_version"],
    )

    print(
        "Wrapper density:",
        f"{wrapper_result.density:.6f}",
    )

    print(
        "\n✅ QR CODE WATERMARK GENERATOR TEST PASSED"
    )


if __name__ == "__main__":
    run_test()
