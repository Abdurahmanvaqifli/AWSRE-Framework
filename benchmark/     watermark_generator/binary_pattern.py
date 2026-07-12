"""
AWSRE Binary Pattern Watermark Generator

Generates deterministic binary watermark patterns for benchmark
experiments.

Supported patterns:

- random
- checkerboard
- diagonal
- border
- cross
- circle
- x_pattern
- horizontal_stripes
- vertical_stripes

Every generated watermark uses internal binary values 0 and 1.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from benchmark.watermark_generator.base_generator import (
    BaseWatermarkGenerator,
    GeneratedWatermark,
    GenerationTimer,
    WatermarkType,
)


# ============================================================
# PATTERN ENUM
# ============================================================

class BinaryPatternType(str, Enum):
    RANDOM = "random"
    CHECKERBOARD = "checkerboard"
    DIAGONAL = "diagonal"
    BORDER = "border"
    CROSS = "cross"
    CIRCLE = "circle"
    X_PATTERN = "x_pattern"
    HORIZONTAL_STRIPES = "horizontal_stripes"
    VERTICAL_STRIPES = "vertical_stripes"


# ============================================================
# GENERATOR
# ============================================================

class BinaryPatternGenerator(
    BaseWatermarkGenerator
):
    """
    Deterministic binary-pattern generator.
    """

    GENERATOR_NAME = "Binary Pattern Generator"
    WATERMARK_TYPE = WatermarkType.BINARY_PATTERN

    DESCRIPTION = (
        "Generates reproducible binary patterns for AWSRE "
        "benchmark experiments."
    )

    SUPPORTED_PATTERNS = tuple(
        item.value
        for item in BinaryPatternType
    )

    def generate(
        self,
        size: Tuple[int, int],
        *,
        pattern: str | BinaryPatternType = (
            BinaryPatternType.RANDOM
        ),
        seed: Optional[int] = None,
        density: float = 0.5,
        thickness: int = 1,
        invert: bool = False,
        **kwargs: Any,
    ) -> GeneratedWatermark:
        """
        Generate a binary watermark.

        Parameters
        ----------
        size:
            Watermark size as (width, height).
        pattern:
            Pattern name or BinaryPatternType.
        seed:
            Reproducibility seed.
        density:
            Foreground probability for random pattern.
        thickness:
            Thickness used by border, cross, circle and X patterns.
        invert:
            Exchange zero and one values.
        """
        width, height = self.validate_size(
            size
        )

        normalized_pattern = self.normalize_pattern(
            pattern
        )

        resolved_seed = self.resolve_seed(
            seed
        )

        density = float(density)

        if not 0.0 <= density <= 1.0:
            raise ValueError(
                "density must be between 0 and 1."
            )

        thickness = max(
            1,
            int(thickness),
        )

        rng = np.random.default_rng(
            resolved_seed
        )

        with GenerationTimer() as timer:
            watermark = self._generate_pattern(
                pattern=normalized_pattern,
                width=width,
                height=height,
                rng=rng,
                density=density,
                thickness=thickness,
            )

            if invert:
                watermark = (
                    1 - watermark
                ).astype(np.uint8)

        metadata: Dict[str, Any] = {
            "pattern": normalized_pattern,
            "requested_density": density,
            "actual_density": float(
                np.mean(watermark)
            ),
            "thickness": thickness,
            "invert": bool(invert),
            "supported_patterns": list(
                self.SUPPORTED_PATTERNS
            ),
        }

        return self.build_result(
            image=watermark,
            size=(width, height),
            seed=resolved_seed,
            generation_time_seconds=(
                timer.elapsed
            ),
            metadata=metadata,
        )

    # --------------------------------------------------------
    # PATTERN DISPATCH
    # --------------------------------------------------------

    def _generate_pattern(
        self,
        *,
        pattern: str,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        """
        Route generation to the selected pattern function.
        """
        generators = {
            BinaryPatternType.RANDOM.value:
                self._random_pattern,

            BinaryPatternType.CHECKERBOARD.value:
                self._checkerboard_pattern,

            BinaryPatternType.DIAGONAL.value:
                self._diagonal_pattern,

            BinaryPatternType.BORDER.value:
                self._border_pattern,

            BinaryPatternType.CROSS.value:
                self._cross_pattern,

            BinaryPatternType.CIRCLE.value:
                self._circle_pattern,

            BinaryPatternType.X_PATTERN.value:
                self._x_pattern,

            BinaryPatternType.HORIZONTAL_STRIPES.value:
                self._horizontal_stripes_pattern,

            BinaryPatternType.VERTICAL_STRIPES.value:
                self._vertical_stripes_pattern,
        }

        generator = generators[
            pattern
        ]

        watermark = generator(
            width=width,
            height=height,
            rng=rng,
            density=density,
            thickness=thickness,
        )

        watermark = np.asarray(
            watermark,
            dtype=np.uint8,
        )

        if watermark.shape != (
            height,
            width,
        ):
            raise RuntimeError(
                "Generated pattern has incorrect shape: "
                f"{watermark.shape}."
            )

        if not set(
            np.unique(watermark)
        ).issubset({0, 1}):
            raise RuntimeError(
                "Generated pattern contains non-binary values."
            )

        return watermark

    # --------------------------------------------------------
    # INDIVIDUAL PATTERNS
    # --------------------------------------------------------

    @staticmethod
    def _random_pattern(
        *,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        return (
            rng.random(
                (height, width)
            ) < density
        ).astype(np.uint8)

    @staticmethod
    def _checkerboard_pattern(
        *,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        rows, columns = np.indices(
            (height, width)
        )

        return (
            (rows + columns) % 2
        ).astype(np.uint8)

    @staticmethod
    def _diagonal_pattern(
        *,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        pattern = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        spacing = max(
            2,
            thickness * 3,
        )

        rows, columns = np.indices(
            (height, width)
        )

        pattern[
            (rows - columns) % spacing
            < thickness
        ] = 1

        return pattern

    @staticmethod
    def _border_pattern(
        *,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        pattern = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        effective_thickness = min(
            thickness,
            max(
                1,
                min(width, height) // 2,
            ),
        )

        pattern[
            :effective_thickness,
            :,
        ] = 1

        pattern[
            -effective_thickness:,
            :,
        ] = 1

        pattern[
            :,
            :effective_thickness,
        ] = 1

        pattern[
            :,
            -effective_thickness:,
        ] = 1

        return pattern

    @staticmethod
    def _cross_pattern(
        *,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        pattern = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        center_x = width // 2
        center_y = height // 2

        half = max(
            0,
            thickness // 2,
        )

        x_start = max(
            0,
            center_x - half,
        )

        x_end = min(
            width,
            center_x + half + 1,
        )

        y_start = max(
            0,
            center_y - half,
        )

        y_end = min(
            height,
            center_y + half + 1,
        )

        pattern[
            :,
            x_start:x_end,
        ] = 1

        pattern[
            y_start:y_end,
            :,
        ] = 1

        return pattern

    @staticmethod
    def _circle_pattern(
        *,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        pattern = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        center = (
            width // 2,
            height // 2,
        )

        radius = max(
            1,
            min(width, height) // 3,
        )

        cv2.circle(
            pattern,
            center,
            radius,
            color=1,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )

        return pattern

    @staticmethod
    def _x_pattern(
        *,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        pattern = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        cv2.line(
            pattern,
            (0, 0),
            (width - 1, height - 1),
            color=1,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )

        cv2.line(
            pattern,
            (width - 1, 0),
            (0, height - 1),
            color=1,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )

        return pattern

    @staticmethod
    def _horizontal_stripes_pattern(
        *,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        pattern = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        stripe_width = max(
            1,
            thickness,
        )

        step = stripe_width * 2

        for row in range(
            0,
            height,
            step,
        ):
            pattern[
                row:min(
                    row + stripe_width,
                    height,
                ),
                :,
            ] = 1

        return pattern

    @staticmethod
    def _vertical_stripes_pattern(
        *,
        width: int,
        height: int,
        rng: np.random.Generator,
        density: float,
        thickness: int,
    ) -> np.ndarray:
        pattern = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        stripe_width = max(
            1,
            thickness,
        )

        step = stripe_width * 2

        for column in range(
            0,
            width,
            step,
        ):
            pattern[
                :,
                column:min(
                    column + stripe_width,
                    width,
                ),
            ] = 1

        return pattern

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    @classmethod
    def normalize_pattern(
        cls,
        pattern: str | BinaryPatternType,
    ) -> str:
        """
        Normalize and validate pattern name.
        """
        if isinstance(
            pattern,
            BinaryPatternType,
        ):
            normalized = pattern.value

        elif isinstance(
            pattern,
            str,
        ):
            normalized = (
                pattern
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

        else:
            raise TypeError(
                "pattern must be a string or BinaryPatternType."
            )

        aliases = {
            "checker": "checkerboard",
            "checker_board": "checkerboard",
            "x": "x_pattern",
            "xpattern": "x_pattern",
            "horizontal": "horizontal_stripes",
            "vertical": "vertical_stripes",
            "random_binary": "random",
        }

        normalized = aliases.get(
            normalized,
            normalized,
        )

        if normalized not in cls.SUPPORTED_PATTERNS:
            raise ValueError(
                f"Unsupported binary pattern: {normalized}. "
                f"Supported: {', '.join(cls.SUPPORTED_PATTERNS)}."
            )

        return normalized


# ============================================================
# FUNCTIONAL WRAPPER
# ============================================================

def generate_binary_pattern(
    size: Tuple[int, int] = (32, 32),
    *,
    pattern: str | BinaryPatternType = "random",
    seed: Optional[int] = 42,
    density: float = 0.5,
    thickness: int = 1,
    invert: bool = False,
) -> GeneratedWatermark:
    """
    Convenience wrapper around BinaryPatternGenerator.
    """
    generator = BinaryPatternGenerator(
        default_seed=seed
    )

    return generator.generate(
        size=size,
        pattern=pattern,
        seed=seed,
        density=density,
        thickness=thickness,
        invert=invert,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    generator = BinaryPatternGenerator(
        default_seed=42
    )

    first = generator.generate(
        size=(32, 32),
        pattern="random",
        seed=42,
        density=0.5,
    )

    second = generator.generate(
        size=(32, 32),
        pattern="random",
        seed=42,
        density=0.5,
    )

    checkerboard = generator.generate(
        size=(32, 32),
        pattern="checkerboard",
    )

    assert np.array_equal(
        first.image,
        second.image,
    )

    assert first.file_hash == second.file_hash
    assert first.image.shape == (32, 32)
    assert checkerboard.image.shape == (32, 32)

    assert set(
        np.unique(first.image)
    ).issubset({0, 1})

    print("=" * 72)
    print("AWSRE BINARY PATTERN GENERATOR TEST")
    print("=" * 72)

    print(
        "Supported patterns:",
        generator.SUPPORTED_PATTERNS,
    )

    print(
        "Random density:",
        f"{first.density:.4f}",
    )

    print(
        "Checkerboard density:",
        f"{checkerboard.density:.4f}",
    )

    print(
        "Generation runtime:",
        f"{first.generation_time_seconds:.8f} s",
    )

    print(
        "Reproducible hash:",
        first.file_hash,
    )

    print(
        "\n✅ Binary pattern generator self test passed."
    )
