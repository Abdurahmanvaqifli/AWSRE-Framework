"""
AWSRE Text Watermark Generator Test
"""

import numpy as np

from benchmark.watermark_generator.base_generator import (
    GeneratedWatermark,
    WatermarkType,
)
from benchmark.watermark_generator.text_generator import (
    TextFont,
    TextWatermarkGenerator,
    generate_text_watermark,
)


def run_test():
    print("=" * 72)
    print("AWSRE TEXT WATERMARK GENERATOR TEST")
    print("=" * 72)

    generator = TextWatermarkGenerator(
        default_seed=2026
    )

    for font in TextFont:
        result = generator.generate(
            size=(128, 64),
            text="AWSRE",
            font=font,
            thickness=1,
            padding=3,
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
            == WatermarkType.TEXT.value
        )

        assert result.payload_bits == 8192
        assert np.sum(result.image) > 0
        assert len(result.file_hash) == 64

    # Deterministic random text.
    first = generator.generate(
        size=(128, 64),
        random_length=12,
        seed=42,
    )

    second = generator.generate(
        size=(128, 64),
        random_length=12,
        seed=42,
    )

    third = generator.generate(
        size=(128, 64),
        random_length=12,
        seed=43,
    )

    assert (
        first.metadata["text"]
        == second.metadata["text"]
    )

    assert np.array_equal(
        first.image,
        second.image,
    )

    assert (
        first.metadata["text"]
        != third.metadata["text"]
    )

    # Inversion.
    normal = generator.generate(
        size=(64, 64),
        text="A",
        invert=False,
    )

    inverted = generator.generate(
        size=(64, 64),
        text="A",
        invert=True,
    )

    assert np.array_equal(
        inverted.image,
        1 - normal.image,
    )

    # Functional wrapper.
    wrapper_result = generate_text_watermark(
        size=(64, 64),
        text="AWSRE",
        seed=42,
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
        "\nFonts tested:",
        len(TextFont),
    )

    print(
        "Generated random text:",
        first.metadata["text"],
    )

    print(
        "Fixed watermark density:",
        f"{wrapper_result.density:.6f}",
    )

    print(
        "\n✅ TEXT WATERMARK GENERATOR TEST PASSED"
    )


if __name__ == "__main__":
    run_test()
