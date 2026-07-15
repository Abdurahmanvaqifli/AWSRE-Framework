"""
AWSRE Text Watermark Generator

Generates reproducible binary text watermarks for AWSRE-Bench.

Features:
- fixed or automatically generated text;
- multiple OpenCV fonts;
- automatic font scaling;
- centered or custom positioning;
- optional inversion;
- deterministic random text generation;
- binary output containing only 0 and 1.
"""

from __future__ import annotations

import string
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
# FONT ENUM
# ============================================================

class TextFont(str, Enum):
    SIMPLEX = "hershey_simplex"
    PLAIN = "hershey_plain"
    DUPLEX = "hershey_duplex"
    COMPLEX = "hershey_complex"
    TRIPLEX = "hershey_triplex"
    COMPLEX_SMALL = "hershey_complex_small"
    SCRIPT_SIMPLEX = "hershey_script_simplex"
    SCRIPT_COMPLEX = "hershey_script_complex"


FONT_MAP = {
    TextFont.SIMPLEX.value: cv2.FONT_HERSHEY_SIMPLEX,
    TextFont.PLAIN.value: cv2.FONT_HERSHEY_PLAIN,
    TextFont.DUPLEX.value: cv2.FONT_HERSHEY_DUPLEX,
    TextFont.COMPLEX.value: cv2.FONT_HERSHEY_COMPLEX,
    TextFont.TRIPLEX.value: cv2.FONT_HERSHEY_TRIPLEX,
    TextFont.COMPLEX_SMALL.value: cv2.FONT_HERSHEY_COMPLEX_SMALL,
    TextFont.SCRIPT_SIMPLEX.value: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    TextFont.SCRIPT_COMPLEX.value: cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}


# ============================================================
# TEXT GENERATOR
# ============================================================

class TextWatermarkGenerator(BaseWatermarkGenerator):
    """
    Binary text watermark generator.
    """

    GENERATOR_NAME = "Text Watermark Generator"
    WATERMARK_TYPE = WatermarkType.TEXT

    DESCRIPTION = (
        "Generates deterministic binary text watermarks with "
        "automatic font fitting and configurable rendering."
    )

    def generate(
        self,
        size: Tuple[int, int],
        *,
        text: Optional[str] = "AWSRE",
        seed: Optional[int] = None,
        random_length: Optional[int] = None,
        charset: str = string.ascii_uppercase + string.digits,
        font: str | TextFont = TextFont.SIMPLEX,
        font_scale: Optional[float] = None,
        thickness: int = 1,
        padding: int = 2,
        centered: bool = True,
        position: Optional[Tuple[int, int]] = None,
        invert: bool = False,
        line_type: int = cv2.LINE_AA,
        **kwargs: Any,
    ) -> GeneratedWatermark:
        """
        Generate a binary text watermark.

        Parameters
        ----------
        size:
            Output size as (width, height).
        text:
            Text to render. Ignored when random_length is supplied.
        seed:
            Seed used for deterministic random text generation.
        random_length:
            Generate random text with this number of characters.
        charset:
            Character set used for random text.
        font:
            OpenCV font name.
        font_scale:
            Explicit scale. When None, the largest fitting scale
            is calculated automatically.
        thickness:
            Text stroke thickness.
        padding:
            Minimum empty margin around text.
        centered:
            Center text automatically.
        position:
            Optional baseline position as (x, y). Used when
            centered=False.
        invert:
            Invert the final binary watermark.
        """
        width, height = self.validate_size(size)
        resolved_seed = self.resolve_seed(seed)

        rendered_text = self._resolve_text(
            text=text,
            random_length=random_length,
            charset=charset,
            seed=resolved_seed,
        )

        font_name = self.normalize_font(font)
        font_face = FONT_MAP[font_name]

        thickness = max(1, int(thickness))
        padding = max(0, int(padding))

        if padding * 2 >= width or padding * 2 >= height:
            raise ValueError(
                "Padding is too large for the requested watermark size."
            )

        with GenerationTimer() as timer:
            if font_scale is None:
                resolved_scale = self.find_fitting_font_scale(
                    text=rendered_text,
                    font_face=font_face,
                    thickness=thickness,
                    available_width=width - 2 * padding,
                    available_height=height - 2 * padding,
                )
            else:
                resolved_scale = float(font_scale)

                if resolved_scale <= 0:
                    raise ValueError(
                        "font_scale must be greater than zero."
                    )

            canvas = np.zeros(
                (height, width),
                dtype=np.uint8,
            )

            text_size, baseline = cv2.getTextSize(
                rendered_text,
                font_face,
                resolved_scale,
                thickness,
            )

            text_width, text_height = text_size

            if centered:
                x = max(
                    padding,
                    (width - text_width) // 2,
                )

                y = max(
                    text_height + padding,
                    (height + text_height) // 2,
                )
            else:
                if position is None:
                    x = padding
                    y = text_height + padding
                else:
                    x = int(position[0])
                    y = int(position[1])

            x = max(0, min(x, width - 1))
            y = max(text_height, min(y, height - 1))

            cv2.putText(
                canvas,
                rendered_text,
                (x, y),
                font_face,
                resolved_scale,
                color=255,
                thickness=thickness,
                lineType=line_type,
            )

            _, binary = cv2.threshold(
                canvas,
                127,
                1,
                cv2.THRESH_BINARY,
            )

            binary = binary.astype(np.uint8)

            if invert:
                binary = (
                    1 - binary
                ).astype(np.uint8)

        if np.sum(binary) == 0:
            raise RuntimeError(
                "Rendered text watermark is empty. "
                "Increase watermark size or reduce text length."
            )

        metadata: Dict[str, Any] = {
            "text": rendered_text,
            "text_length": len(rendered_text),
            "random_text": random_length is not None,
            "font": font_name,
            "font_scale": float(resolved_scale),
            "thickness": thickness,
            "padding": padding,
            "centered": bool(centered),
            "position": [int(x), int(y)],
            "invert": bool(invert),
            "text_width": int(text_width),
            "text_height": int(text_height),
            "baseline": int(baseline),
            "foreground_pixels": int(np.sum(binary)),
            "actual_density": float(np.mean(binary)),
        }

        return self.build_result(
            image=binary,
            size=(width, height),
            seed=resolved_seed,
            generation_time_seconds=timer.elapsed,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # TEXT PREPARATION
    # --------------------------------------------------------

    @staticmethod
    def _resolve_text(
        *,
        text: Optional[str],
        random_length: Optional[int],
        charset: str,
        seed: Optional[int],
    ) -> str:
        """
        Return supplied text or generate deterministic random text.
        """
        if random_length is not None:
            length = int(random_length)

            if length <= 0:
                raise ValueError(
                    "random_length must be greater than zero."
                )

            if not charset:
                raise ValueError(
                    "charset cannot be empty."
                )

            rng = np.random.default_rng(seed)

            characters = np.asarray(
                list(charset)
            )

            indexes = rng.integers(
                0,
                len(characters),
                size=length,
            )

            return "".join(
                characters[indexes]
            )

        normalized_text = (
            str(text).strip()
            if text is not None
            else ""
        )

        if not normalized_text:
            raise ValueError(
                "Text watermark cannot be empty."
            )

        return normalized_text

    # --------------------------------------------------------
    # FONT HELPERS
    # --------------------------------------------------------

    @staticmethod
    def normalize_font(
        font: str | TextFont,
    ) -> str:
        """
        Normalize and validate font name.
        """
        if isinstance(font, TextFont):
            normalized = font.value

        elif isinstance(font, str):
            normalized = (
                font.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

        else:
            raise TypeError(
                "font must be a string or TextFont."
            )

        aliases = {
            "simplex": TextFont.SIMPLEX.value,
            "plain": TextFont.PLAIN.value,
            "duplex": TextFont.DUPLEX.value,
            "complex": TextFont.COMPLEX.value,
            "triplex": TextFont.TRIPLEX.value,
            "script": TextFont.SCRIPT_SIMPLEX.value,
        }

        normalized = aliases.get(
            normalized,
            normalized,
        )

        if normalized not in FONT_MAP:
            raise ValueError(
                f"Unsupported font: {normalized}. "
                f"Supported fonts: {', '.join(FONT_MAP.keys())}."
            )

        return normalized

    @staticmethod
    def find_fitting_font_scale(
        *,
        text: str,
        font_face: int,
        thickness: int,
        available_width: int,
        available_height: int,
        minimum_scale: float = 0.05,
        maximum_scale: float = 10.0,
        iterations: int = 40,
    ) -> float:
        """
        Find the largest font scale fitting inside the canvas.
        """
        if available_width <= 0 or available_height <= 0:
            raise ValueError(
                "Available text area must be positive."
            )

        low = minimum_scale
        high = maximum_scale
        best = minimum_scale

        for _ in range(iterations):
            middle = (low + high) / 2.0

            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font_face,
                middle,
                thickness,
            )

            fits = (
                text_width <= available_width
                and text_height + baseline <= available_height
            )

            if fits:
                best = middle
                low = middle
            else:
                high = middle

        return float(best)


# ============================================================
# FUNCTIONAL WRAPPER
# ============================================================

def generate_text_watermark(
    size: Tuple[int, int] = (64, 64),
    *,
    text: str = "AWSRE",
    seed: Optional[int] = 42,
    random_length: Optional[int] = None,
    font: str | TextFont = TextFont.SIMPLEX,
    thickness: int = 1,
    padding: int = 2,
    invert: bool = False,
) -> GeneratedWatermark:
    """
    Convenience wrapper around TextWatermarkGenerator.
    """
    generator = TextWatermarkGenerator(
        default_seed=seed
    )

    return generator.generate(
        size=size,
        text=text,
        seed=seed,
        random_length=random_length,
        font=font,
        thickness=thickness,
        padding=padding,
        invert=invert,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    generator = TextWatermarkGenerator(
        default_seed=42
    )

    fixed = generator.generate(
        size=(64, 64),
        text="AWSRE",
        font="simplex",
        thickness=1,
    )

    random_1 = generator.generate(
        size=(128, 64),
        random_length=10,
        seed=42,
    )

    random_2 = generator.generate(
        size=(128, 64),
        random_length=10,
        seed=42,
    )

    assert fixed.image.shape == (64, 64)

    assert np.array_equal(
        random_1.image,
        random_2.image,
    )

    assert (
        random_1.metadata["text"]
        == random_2.metadata["text"]
    )

    print("=" * 72)
    print("AWSRE TEXT WATERMARK GENERATOR TEST")
    print("=" * 72)

    print(
        "Fixed text:",
        fixed.metadata["text"],
    )

    print(
        "Random text:",
        random_1.metadata["text"],
    )

    print(
        "Font scale:",
        fixed.metadata["font_scale"],
    )

    print(
        "Density:",
        f"{fixed.density:.6f}",
    )

    print(
        "Generation time:",
        f"{fixed.generation_time_seconds:.8f} s",
    )

    print(
        "\n✅ Text watermark generator self test passed."
    )
