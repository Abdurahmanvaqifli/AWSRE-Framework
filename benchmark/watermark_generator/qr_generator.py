"""
AWSRE QR Code Watermark Generator

Generates deterministic binary QR-code watermarks for AWSRE-Bench.

Features:
- configurable QR error correction;
- automatic or fixed QR version;
- deterministic text generation;
- nearest-neighbor resizing;
- binary 0/1 output;
- optional inversion;
- complete benchmark metadata.
"""

from __future__ import annotations

import string
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import qrcode
from qrcode.exceptions import DataOverflowError

from benchmark.watermark_generator.base_generator import (
    BaseWatermarkGenerator,
    GeneratedWatermark,
    GenerationTimer,
    WatermarkType,
)


# ============================================================
# ERROR-CORRECTION ENUM
# ============================================================

class QRErrorCorrection(str, Enum):
    LOW = "L"
    MEDIUM = "M"
    QUARTILE = "Q"
    HIGH = "H"


ERROR_CORRECTION_MAP = {
    QRErrorCorrection.LOW.value:
        qrcode.constants.ERROR_CORRECT_L,

    QRErrorCorrection.MEDIUM.value:
        qrcode.constants.ERROR_CORRECT_M,

    QRErrorCorrection.QUARTILE.value:
        qrcode.constants.ERROR_CORRECT_Q,

    QRErrorCorrection.HIGH.value:
        qrcode.constants.ERROR_CORRECT_H,
}


# ============================================================
# QR GENERATOR
# ============================================================

class QRCodeWatermarkGenerator(BaseWatermarkGenerator):
    """
    Binary QR-code watermark generator.
    """

    GENERATOR_NAME = "QR Code Watermark Generator"
    WATERMARK_TYPE = WatermarkType.QR_CODE

    DESCRIPTION = (
        "Generates reproducible binary QR-code watermarks "
        "with configurable error correction and version."
    )

    def generate(
        self,
        size: Tuple[int, int],
        *,
        text: Optional[str] = "AWSRE",
        seed: Optional[int] = None,
        random_length: Optional[int] = None,
        charset: str = string.ascii_uppercase + string.digits,
        version: Optional[int] = None,
        error_correction: str | QRErrorCorrection = (
            QRErrorCorrection.MEDIUM
        ),
        border: int = 2,
        box_size: int = 10,
        fit: bool = True,
        invert: bool = False,
        preserve_square: bool = True,
        **kwargs: Any,
    ) -> GeneratedWatermark:
        """
        Generate a binary QR-code watermark.

        Parameters
        ----------
        size:
            Output watermark size as (width, height).
        text:
            Content encoded inside the QR code.
        seed:
            Seed for deterministic random content.
        random_length:
            Generate random QR content when supplied.
        charset:
            Character set used for random content.
        version:
            QR version from 1 to 40, or None for automatic.
        error_correction:
            One of L, M, Q or H.
        border:
            Number of quiet-zone modules.
        box_size:
            Pixel size used during initial QR rendering.
        fit:
            Automatically increase QR version when necessary.
        invert:
            Exchange zero and one values.
        preserve_square:
            Center the QR inside a non-square output canvas
            instead of stretching it.
        """
        width, height = self.validate_size(
            size
        )

        resolved_seed = self.resolve_seed(
            seed
        )

        encoded_text = self._resolve_text(
            text=text,
            random_length=random_length,
            charset=charset,
            seed=resolved_seed,
        )

        normalized_error_correction = (
            self.normalize_error_correction(
                error_correction
            )
        )

        resolved_version = self.validate_version(
            version
        )

        border = max(
            0,
            int(border),
        )

        box_size = max(
            1,
            int(box_size),
        )

        with GenerationTimer() as timer:
            qr = qrcode.QRCode(
                version=resolved_version,
                error_correction=ERROR_CORRECTION_MAP[
                    normalized_error_correction
                ],
                box_size=box_size,
                border=border,
            )

            qr.add_data(
                encoded_text
            )

            try:
                qr.make(
                    fit=bool(fit)
                )
            except DataOverflowError as exc:
                raise ValueError(
                    "QR content does not fit the selected version "
                    "and error-correction configuration."
                ) from exc

            pil_image = qr.make_image(
                fill_color="black",
                back_color="white",
            ).convert("L")

            qr_uint8 = np.asarray(
                pil_image,
                dtype=np.uint8,
            )

            # QR black modules become 1 and white background becomes 0.
            qr_binary = (
                qr_uint8 < 128
            ).astype(np.uint8)

            original_module_image_size = (
                int(qr_binary.shape[1]),
                int(qr_binary.shape[0]),
            )

            resized = self._resize_qr(
                qr_binary,
                output_size=(width, height),
                preserve_square=bool(
                    preserve_square
                ),
            )

            if invert:
                resized = (
                    1 - resized
                ).astype(np.uint8)

        if resized.shape != (
            height,
            width,
        ):
            raise RuntimeError(
                "QR generator returned an incorrect output shape."
            )

        if not set(
            np.unique(resized)
        ).issubset({0, 1}):
            raise RuntimeError(
                "QR generator returned non-binary values."
            )

        metadata: Dict[str, Any] = {
            "encoded_text": encoded_text,
            "payload_length": len(encoded_text),
            "random_text": random_length is not None,
            "requested_version": resolved_version,
            "actual_version": int(qr.version),
            "error_correction": (
                normalized_error_correction
            ),
            "border": border,
            "box_size": box_size,
            "fit": bool(fit),
            "invert": bool(invert),
            "preserve_square": bool(
                preserve_square
            ),
            "module_count": int(
                qr.modules_count
            ),
            "source_qr_width": (
                original_module_image_size[0]
            ),
            "source_qr_height": (
                original_module_image_size[1]
            ),
            "foreground_pixels": int(
                np.sum(resized)
            ),
            "actual_density": float(
                np.mean(resized)
            ),
        }

        return self.build_result(
            image=resized,
            size=(width, height),
            seed=resolved_seed,
            generation_time_seconds=(
                timer.elapsed
            ),
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
        Return supplied content or deterministic random text.
        """
        if random_length is not None:
            length = int(
                random_length
            )

            if length <= 0:
                raise ValueError(
                    "random_length must be greater than zero."
                )

            if not charset:
                raise ValueError(
                    "charset cannot be empty."
                )

            rng = np.random.default_rng(
                seed
            )

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
            str(text)
            if text is not None
            else ""
        )

        if not normalized_text:
            raise ValueError(
                "QR content cannot be empty."
            )

        return normalized_text

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def normalize_error_correction(
        value: str | QRErrorCorrection,
    ) -> str:
        """
        Normalize QR error-correction level.
        """
        if isinstance(
            value,
            QRErrorCorrection,
        ):
            normalized = value.value

        elif isinstance(
            value,
            str,
        ):
            normalized = value.strip().upper()

        else:
            raise TypeError(
                "error_correction must be a string "
                "or QRErrorCorrection."
            )

        aliases = {
            "LOW": "L",
            "MEDIUM": "M",
            "QUARTILE": "Q",
            "HIGH": "H",
        }

        normalized = aliases.get(
            normalized,
            normalized,
        )

        if normalized not in ERROR_CORRECTION_MAP:
            raise ValueError(
                "Unsupported QR error correction: "
                f"{normalized}. Supported values: L, M, Q, H."
            )

        return normalized

    @staticmethod
    def validate_version(
        version: Optional[int],
    ) -> Optional[int]:
        """
        Validate QR version.
        """
        if version is None:
            return None

        normalized = int(
            version
        )

        if not 1 <= normalized <= 40:
            raise ValueError(
                "QR version must be between 1 and 40."
            )

        return normalized

    # --------------------------------------------------------
    # RESIZING
    # --------------------------------------------------------

    @staticmethod
    def _resize_qr(
        qr_binary: np.ndarray,
        *,
        output_size: Tuple[int, int],
        preserve_square: bool,
    ) -> np.ndarray:
        """
        Resize QR code using nearest-neighbor interpolation.
        """
        output_width = int(
            output_size[0]
        )

        output_height = int(
            output_size[1]
        )

        if not preserve_square:
            resized = cv2.resize(
                qr_binary,
                (
                    output_width,
                    output_height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            return (
                resized > 0
            ).astype(np.uint8)

        target_side = min(
            output_width,
            output_height,
        )

        square_qr = cv2.resize(
            qr_binary,
            (
                target_side,
                target_side,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        canvas = np.zeros(
            (
                output_height,
                output_width,
            ),
            dtype=np.uint8,
        )

        x_start = (
            output_width - target_side
        ) // 2

        y_start = (
            output_height - target_side
        ) // 2

        canvas[
            y_start:y_start + target_side,
            x_start:x_start + target_side,
        ] = square_qr

        return (
            canvas > 0
        ).astype(np.uint8)


# ============================================================
# FUNCTIONAL WRAPPER
# ============================================================

def generate_qr_watermark(
    size: Tuple[int, int] = (64, 64),
    *,
    text: str = "AWSRE",
    seed: Optional[int] = 42,
    version: Optional[int] = None,
    error_correction: str | QRErrorCorrection = "M",
    border: int = 2,
    invert: bool = False,
) -> GeneratedWatermark:
    """
    Convenience wrapper around QRCodeWatermarkGenerator.
    """
    generator = QRCodeWatermarkGenerator(
        default_seed=seed
    )

    return generator.generate(
        size=size,
        text=text,
        seed=seed,
        version=version,
        error_correction=error_correction,
        border=border,
        invert=invert,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    generator = QRCodeWatermarkGenerator(
        default_seed=42
    )

    fixed = generator.generate(
        size=(64, 64),
        text="AWSRE-2026",
        error_correction="M",
    )

    random_1 = generator.generate(
        size=(128, 128),
        random_length=20,
        seed=42,
        error_correction="H",
    )

    random_2 = generator.generate(
        size=(128, 128),
        random_length=20,
        seed=42,
        error_correction="H",
    )

    assert fixed.image.shape == (
        64,
        64,
    )

    assert np.array_equal(
        random_1.image,
        random_2.image,
    )

    assert (
        random_1.metadata["encoded_text"]
        == random_2.metadata["encoded_text"]
    )

    print("=" * 72)
    print("AWSRE QR WATERMARK GENERATOR TEST")
    print("=" * 72)

    print(
        "Encoded text:",
        fixed.metadata["encoded_text"],
    )

    print(
        "Actual version:",
        fixed.metadata["actual_version"],
    )

    print(
        "Module count:",
        fixed.metadata["module_count"],
    )

    print(
        "Error correction:",
        fixed.metadata["error_correction"],
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
        "\n✅ QR watermark generator self test passed."
    )
