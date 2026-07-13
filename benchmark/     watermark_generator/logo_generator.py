"""
AWSRE Logo Watermark Generator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from benchmark.watermark_generator.base_generator import (
    GeneratedWatermark,
    WatermarkType,
)
from benchmark.watermark_generator.image_based_generator import (
    ImageBasedWatermarkGenerator,
)


class LogoWatermarkGenerator(
    ImageBasedWatermarkGenerator
):
    """
    Generates binary logo watermarks from image files or arrays.
    """

    GENERATOR_NAME = "Logo Watermark Generator"
    WATERMARK_TYPE = WatermarkType.LOGO

    DESCRIPTION = (
        "Converts logos into normalized binary watermark patterns "
        "while preserving aspect ratio."
    )

    def generate(
        self,
        size: Tuple[int, int],
        *,
        source: str | Path | np.ndarray,
        seed: Optional[int] = None,
        padding: int = 2,
        threshold: Optional[int] = None,
        use_otsu: bool = True,
        foreground_dark: bool = True,
        invert: bool = False,
        morphology: str = "closing",
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
                "image_category": "logo",
            },
        )


def generate_logo_watermark(
    source: str | Path | np.ndarray,
    size: Tuple[int, int] = (64, 64),
    *,
    seed: Optional[int] = 42,
    padding: int = 2,
    use_otsu: bool = True,
    invert: bool = False,
) -> GeneratedWatermark:
    """
    Functional logo-generation wrapper.
    """
    generator = LogoWatermarkGenerator(
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
    import cv2

    sample = np.full(
        (200, 300),
        255,
        dtype=np.uint8,
    )

    cv2.circle(
        sample,
        (150, 100),
        65,
        0,
        thickness=-1,
    )

    cv2.putText(
        sample,
        "A",
        (115, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        255,
        4,
        cv2.LINE_AA,
    )

    generator = LogoWatermarkGenerator()

    result = generator.generate(
        size=(64, 64),
        source=sample,
    )

    print("=" * 72)
    print("AWSRE LOGO WATERMARK GENERATOR TEST")
    print("=" * 72)

    print("Shape:", result.image.shape)
    print("Density:", f"{result.density:.6f}")
    print("Hash:", result.file_hash)

    print(
        "\n✅ Logo watermark generator self test passed."
    )
