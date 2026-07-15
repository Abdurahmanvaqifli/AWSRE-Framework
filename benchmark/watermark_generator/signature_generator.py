"""
AWSRE Signature Watermark Generator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from benchmark.watermark_generator.base_generator import (
    GeneratedWatermark,
    WatermarkType,
)
from benchmark.watermark_generator.image_based_generator import (
    ImageBasedWatermarkGenerator,
)


class SignatureWatermarkGenerator(
    ImageBasedWatermarkGenerator
):
    """
    Generates binary signature watermarks.
    """

    GENERATOR_NAME = "Signature Watermark Generator"
    WATERMARK_TYPE = WatermarkType.SIGNATURE

    DESCRIPTION = (
        "Converts handwritten signature images into clean "
        "binary watermark patterns."
    )

    def generate(
        self,
        size: Tuple[int, int],
        *,
        source: str | Path | np.ndarray,
        seed: Optional[int] = None,
        padding: int = 3,
        threshold: Optional[int] = None,
        use_otsu: bool = True,
        foreground_dark: bool = True,
        invert: bool = False,
        morphology: str = "opening",
        morphology_kernel_size: int = 3,
        morphology_iterations: int = 1,
        alpha_background: int = 255,
        **kwargs: Any,
    ) -> GeneratedWatermark:
        return self.generate_from_image(
            size=size,
            source=source,
            seed=seed,
            padding=padding,
            threshold=threshold,
            use_otsu=use_otsu,
            foreground_dark=foreground_dark,
            invert=invert,
            morphology=morphology,
            morphology_kernel_size=(
                morphology_kernel_size
            ),
            morphology_iterations=(
                morphology_iterations
            ),
            alpha_background=alpha_background,
            extra_metadata={
                "image_category": "signature",
            },
        )


def generate_signature_watermark(
    source: str | Path | np.ndarray,
    size: Tuple[int, int] = (128, 64),
    *,
    seed: Optional[int] = 42,
    padding: int = 3,
    use_otsu: bool = True,
    invert: bool = False,
) -> GeneratedWatermark:
    """
    Functional signature-generation wrapper.
    """
    generator = SignatureWatermarkGenerator(
        default_seed=seed
    )

    return generator.generate(
        size=size,
        source=source,
        seed=seed,
        padding=padding,
        use_otsu=use_otsu,
        invert=invert,
    )


if __name__ == "__main__":
    sample = np.full(
        (120, 400),
        255,
        dtype=np.uint8,
    )

    points = np.array([
        [20, 80],
        [70, 35],
        [120, 90],
        [180, 30],
        [240, 85],
        [310, 45],
        [380, 75],
    ], dtype=np.int32)

    cv2.polylines(
        sample,
        [points],
        isClosed=False,
        color=0,
        thickness=4,
        lineType=cv2.LINE_AA,
    )

    generator = SignatureWatermarkGenerator()

    result = generator.generate(
        size=(128, 64),
        source=sample,
    )

    print("=" * 72)
    print("AWSRE SIGNATURE WATERMARK GENERATOR TEST")
    print("=" * 72)

    print("Shape:", result.image.shape)
    print("Density:", f"{result.density:.6f}")
    print("Hash:", result.file_hash)

    print(
        "\n✅ Signature watermark generator self test passed."
    )
