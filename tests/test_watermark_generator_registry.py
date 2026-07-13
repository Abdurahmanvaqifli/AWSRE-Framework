"""
AWSRE Watermark Generator Registry Test
"""

import cv2
import numpy as np

from benchmark.watermark_generator.base_generator import (
    GeneratedWatermark,
    WatermarkType,
)
from benchmark.watermark_generator.binary_pattern import (
    BinaryPatternGenerator,
)
from benchmark.watermark_generator.logo_generator import (
    LogoWatermarkGenerator,
)
from benchmark.watermark_generator.qr_generator import (
    QRCodeWatermarkGenerator,
)
from benchmark.watermark_generator.registry import (
    create_generator,
    generate_watermark,
    generator_registry_size,
    get_generator_registry_info,
    is_generator_registered,
    list_registered_generators,
    load_builtin_generators,
)
from benchmark.watermark_generator.signature_generator import (
    SignatureWatermarkGenerator,
)
from benchmark.watermark_generator.text_generator import (
    TextWatermarkGenerator,
)


def run_test():
    print("=" * 72)
    print("AWSRE WATERMARK GENERATOR REGISTRY TEST")
    print("=" * 72)

    loaded = load_builtin_generators(
        strict=True
    )

    expected = {
        "Binary Pattern",
        "Text",
        "QR Code",
        "Logo",
        "Signature",
    }

    assert set(loaded) == expected
    assert generator_registry_size() == 5

    for watermark_type in WatermarkType:
        assert is_generator_registered(
            watermark_type
        )

    binary_generator = create_generator(
        "binary",
        default_seed=42,
    )

    text_generator = create_generator(
        WatermarkType.TEXT,
        default_seed=42,
    )

    qr_generator = create_generator(
        "qr",
        default_seed=42,
    )

    logo_generator = create_generator(
        "logo",
        default_seed=42,
    )

    signature_generator = create_generator(
        "signature",
        default_seed=42,
    )

    assert isinstance(
        binary_generator,
        BinaryPatternGenerator,
    )

    assert isinstance(
        text_generator,
        TextWatermarkGenerator,
    )

    assert isinstance(
        qr_generator,
        QRCodeWatermarkGenerator,
    )

    assert isinstance(
        logo_generator,
        LogoWatermarkGenerator,
    )

    assert isinstance(
        signature_generator,
        SignatureWatermarkGenerator,
    )

    binary_result = generate_watermark(
        "Binary Pattern",
        size=(32, 32),
        pattern="checkerboard",
        seed=42,
    )

    text_result = generate_watermark(
        "Text",
        size=(64, 64),
        text="AWSRE",
        seed=42,
    )

    qr_result = generate_watermark(
        "QR Code",
        size=(64, 64),
        text="AWSRE-2026",
        seed=42,
    )

    logo_source = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )

    cv2.rectangle(
        logo_source,
        (20, 20),
        (180, 80),
        0,
        5,
    )

    logo_result = generate_watermark(
        "Logo",
        size=(64, 64),
        source=logo_source,
        seed=42,
    )

    signature_source = np.full(
        (80, 250),
        255,
        dtype=np.uint8,
    )

    cv2.line(
        signature_source,
        (10, 60),
        (230, 20),
        0,
        3,
    )

    signature_result = generate_watermark(
        "Signature",
        size=(128, 64),
        source=signature_source,
        seed=42,
    )

    results = [
        binary_result,
        text_result,
        qr_result,
        logo_result,
        signature_result,
    ]

    for result in results:
        assert isinstance(
            result,
            GeneratedWatermark,
        )

        assert set(
            np.unique(result.image)
        ).issubset({0, 1})

        assert len(result.file_hash) == 64

    assert binary_result.image.shape == (
        32,
        32,
    )

    assert text_result.image.shape == (
        64,
        64,
    )

    assert qr_result.image.shape == (
        64,
        64,
    )

    assert logo_result.image.shape == (
        64,
        64,
    )

    assert signature_result.image.shape == (
        64,
        128,
    )

    info = get_generator_registry_info()

    assert len(info) == 5

    print(
        "\nRegistered generators:",
        list_registered_generators(),
    )

    print(
        "Generated watermark count:",
        len(results),
    )

    print(
        "\n✅ WATERMARK GENERATOR REGISTRY TEST PASSED"
    )


if __name__ == "__main__":
    run_test()
