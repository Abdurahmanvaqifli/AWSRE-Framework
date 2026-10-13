"""
AWSRE Blind DCT Watermarking Module (DCT-QIM)

Block-based DCT watermarking with blind extraction, using dithered
Quantization Index Modulation (QIM) on the same mid-frequency
coefficients and 8x8 block layout as watermarking/dct.py.

Why a separate method: the classic AWSRE DCT method decodes a bit from
the SIGN of (processed - original) coefficient differences, so it needs
the original host image at extraction time (non-blind). QIM instead
moves each coefficient onto one of two interleaved quantization lattices,
one per bit value. The decoder only has to ask which lattice the received
coefficient is closer to, so it needs the watermarked (or attacked) image
and the secret key, never the original host.

Embedding rule for one coefficient c, bit b, step S and key dither d:

    offset_b = d + b * S / 2
    c'       = S * round((c - offset_b) / S) + offset_b

so |c' - c| <= S / 2. Decoding sums, over the block's coefficients, the
distance to the bit-0 lattice minus the distance to the bit-1 lattice and
outputs 1 when the result is positive (closer to the bit-1 lattice).

The quantization step is S = QIM_STEP_PER_ALPHA * alpha. With the default
factor of 2, the largest change to any coefficient equals alpha, which is
exactly the change the classic DCT method applies, so the alpha slider
keeps a comparable meaning across both methods.

The dither d is drawn per block and per coefficient from a generator
seeded with `key`. Without the key, the decoder does not know where the
lattices are, which is what makes the key a secret.

Known limits (inherent to QIM, not specific to this implementation):
- Amplitude scaling (e.g. contrast changes) moves coefficients off the
  lattice and is the classic QIM weakness.
- Geometric attacks (crop with resize, rotation) desynchronize the 8x8
  block grid, as they do for every AWSRE method.

Registered under its own method name ("DCT-QIM"), so the existing DCT
method and every scientific measurement collected for it are unaffected.
It is not part of the trained Intelligent-Mode models.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from watermarking.base import (
    EmbeddingResult,
    ExtractionResult,
    Timer,
    build_metadata,
    ensure_grayscale,
    validate_host_image,
)
from watermarking.dct import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_COEFFICIENTS,
    DCTWatermarker,
    ensure_uint8_image,
    extract_block,
    normalize_binary_watermark,
    place_block,
    required_host_dimensions,
    validate_embedding_capacity,
)
from watermarking.registry import watermarker


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_QIM_KEY = 2026

# Quantization step S = QIM_STEP_PER_ALPHA * alpha. With 2.0 the largest
# per-coefficient change is alpha, matching the classic DCT method.
QIM_STEP_PER_ALPHA = 2.0


# ============================================================
# QIM HELPERS
# ============================================================

def qim_dither(
    key: int,
    watermark_shape: Tuple[int, int],
    coefficient_count: int,
    step: float,
) -> np.ndarray:
    """
    Key-dependent dither in [0, step), one value per watermark bit
    and per coefficient. The same key and shape always give the same
    dither, which is what lets the decoder find the lattices.
    """
    generator = np.random.default_rng(int(key))

    return generator.uniform(
        0.0,
        float(step),
        size=(
            int(watermark_shape[0]),
            int(watermark_shape[1]),
            int(coefficient_count),
        ),
    )


def qim_quantize(
    value: float,
    bit: int,
    step: float,
    dither: float,
) -> float:
    """
    Move a coefficient onto the lattice that encodes `bit`.
    """
    offset = dither + bit * step / 2.0

    return float(
        step * np.round((value - offset) / step) + offset
    )


def qim_distance(
    value: float,
    bit: int,
    step: float,
    dither: float,
) -> float:
    """
    Distance from a coefficient to the nearest point of the
    lattice that encodes `bit`.
    """
    return abs(
        value - qim_quantize(value, bit, step, dither)
    )


# ============================================================
# DCT-QIM WATERMARKER
# ============================================================

@watermarker
class DCTQIMWatermarker(DCTWatermarker):
    """
    Blind block-based DCT watermarker using dithered QIM.

    Block layout, coefficients and capacity are inherited from
    DCTWatermarker. embed() and extract() are replaced: extraction
    ignores the original image entirely.
    """

    METHOD = "DCT-QIM"

    DESCRIPTION = (
        "Blind block-based DCT watermarking: dithered quantization index "
        "modulation (QIM) on mid-frequency coefficients. Extraction needs "
        "only the watermarked image and the secret key."
    )

    BLIND_EXTRACTION = True

    DEFAULT_ALPHA = 20

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        block_size: int = DEFAULT_BLOCK_SIZE,
        coefficients: Iterable[
            Tuple[int, int]
        ] = DEFAULT_COEFFICIENTS,
        key: int = DEFAULT_QIM_KEY,
    ) -> None:
        super().__init__(
            alpha=alpha,
            block_size=block_size,
            coefficients=coefficients,
        )

        self.key = int(key)

        self.step = float(
            QIM_STEP_PER_ALPHA * self.alpha
        )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    def embed(
        self,
        host: np.ndarray,
        watermark: np.ndarray,
    ) -> EmbeddingResult:
        """
        Embed a binary watermark by quantizing each block's
        mid-frequency coefficients onto the bit's lattice.
        """
        self.validate(
            host,
            watermark,
        )

        host_gray = ensure_grayscale(
            ensure_uint8_image(host)
        )

        watermark_binary = normalize_binary_watermark(
            watermark
        )

        validate_embedding_capacity(
            host_gray,
            watermark_binary,
            self.block_size,
        )

        watermarked_float = host_gray.astype(
            np.float32
        )

        wm_height, wm_width = watermark_binary.shape

        dither = qim_dither(
            self.key,
            watermark_binary.shape,
            len(self.coefficients),
            self.step,
        )

        coefficient_changes = []

        with Timer() as timer:
            for bit_row in range(wm_height):
                for bit_column in range(wm_width):
                    block = extract_block(
                        watermarked_float,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    dct_block = cv2.dct(
                        block.astype(np.float32)
                    )

                    bit = int(
                        watermark_binary[
                            bit_row,
                            bit_column,
                        ]
                    )

                    for index, coefficient in enumerate(
                        self.coefficients
                    ):
                        original_value = float(
                            dct_block[coefficient]
                        )

                        quantized_value = qim_quantize(
                            original_value,
                            bit,
                            self.step,
                            dither[bit_row, bit_column, index],
                        )

                        dct_block[coefficient] = quantized_value

                        coefficient_changes.append(
                            quantized_value - original_value
                        )

                    place_block(
                        watermarked_float,
                        cv2.idct(dct_block),
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

        watermarked_uint8 = np.uint8(
            np.clip(
                np.rint(watermarked_float),
                0,
                255,
            )
        )

        changes = np.abs(
            np.asarray(
                coefficient_changes,
                dtype=np.float64,
            )
        )

        metadata: Dict[str, Any] = build_metadata(
            method=self.METHOD,
            alpha=self.alpha,
            host=host_gray,
            watermark=watermark_binary,
        )

        metadata.update({
            "block_size": self.block_size,
            "coefficients": [
                list(item)
                for item in self.coefficients
            ],
            "quantization_step": self.step,
            "embedded_bits": int(watermark_binary.size),
            "watermark_shape": list(
                watermark_binary.shape
            ),
            "host_shape": list(
                host_gray.shape
            ),
            "required_host_shape": list(
                required_host_dimensions(
                    watermark_binary.shape,
                    self.block_size,
                )
            ),
            "embedding_rate": float(
                watermark_binary.size
                / host_gray.size
            ),
            "mean_absolute_coefficient_change": float(
                changes.mean()
            ),
            "maximum_absolute_coefficient_change": float(
                changes.max()
            ),
            "non_blind_extraction": False,
            "blind_extraction": True,
        })

        return EmbeddingResult(
            watermarked_image=watermarked_uint8,
            runtime=timer.elapsed,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # EXTRACTION
    # --------------------------------------------------------

    def extract(
        self,
        original: Optional[np.ndarray],
        watermarked: np.ndarray,
        watermark_shape: Tuple[int, int],
    ) -> ExtractionResult:
        """
        Blind extraction.

        Parameters
        ----------
        original:
            Ignored. Accepted (and may be None) only so this method
            keeps the same call signature as every other AWSRE method.
        watermarked:
            Watermarked or attacked watermarked image.
        watermark_shape:
            Watermark shape as (height, width).
        """
        return self.extract_blind(
            watermarked,
            watermark_shape,
        )

    def extract_blind(
        self,
        watermarked: np.ndarray,
        watermark_shape: Tuple[int, int],
    ) -> ExtractionResult:
        """
        Decode the watermark from the watermarked image alone.
        """
        validate_host_image(watermarked)

        watermarked_gray = ensure_grayscale(
            ensure_uint8_image(watermarked)
        )

        normalized_shape = (
            int(watermark_shape[0]),
            int(watermark_shape[1]),
        )

        if normalized_shape[0] <= 0 or normalized_shape[1] <= 0:
            raise ValueError(
                "Watermark dimensions must be positive."
            )

        required_height, required_width = required_host_dimensions(
            normalized_shape,
            self.block_size,
        )

        if (
            watermarked_gray.shape[0] < required_height
            or watermarked_gray.shape[1] < required_width
        ):
            raise ValueError(
                "Image is too small for the requested extraction shape."
            )

        watermarked_float = watermarked_gray.astype(
            np.float32
        )

        dither = qim_dither(
            self.key,
            normalized_shape,
            len(self.coefficients),
            self.step,
        )

        wm_height, wm_width = normalized_shape

        extracted = np.zeros(
            normalized_shape,
            dtype=np.uint8,
        )

        scores = np.zeros(
            normalized_shape,
            dtype=np.float64,
        )

        with Timer() as timer:
            for bit_row in range(wm_height):
                for bit_column in range(wm_width):
                    dct_block = cv2.dct(
                        extract_block(
                            watermarked_float,
                            bit_row,
                            bit_column,
                            self.block_size,
                        ).astype(np.float32)
                    )

                    score = 0.0

                    for index, coefficient in enumerate(
                        self.coefficients
                    ):
                        value = float(
                            dct_block[coefficient]
                        )

                        block_dither = dither[
                            bit_row,
                            bit_column,
                            index,
                        ]

                        score += (
                            qim_distance(value, 0, self.step, block_dither)
                            - qim_distance(value, 1, self.step, block_dither)
                        )

                    scores[bit_row, bit_column] = score

                    extracted[bit_row, bit_column] = (
                        1
                        if score > 0
                        else 0
                    )

        # Each coefficient's score lies in [-S/2, S/2]; dividing by the
        # largest possible total gives a confidence in [0, 1] per bit.
        max_score = (
            len(self.coefficients)
            * self.step
            / 2.0
        )

        metadata = {
            "method": self.METHOD,
            "alpha": self.alpha,
            "block_size": self.block_size,
            "coefficients": [
                list(item)
                for item in self.coefficients
            ],
            "quantization_step": self.step,
            "watermark_shape": list(
                normalized_shape
            ),
            "extracted_bits": int(
                extracted.size
            ),
            "mean_decision_confidence": float(
                np.mean(np.abs(scores)) / max_score
            ),
            "minimum_decision_confidence": float(
                np.min(np.abs(scores)) / max_score
            ),
            "non_blind_extraction": False,
            "blind_extraction": True,
        }

        return ExtractionResult(
            extracted_watermark=extracted,
            runtime=timer.elapsed,
            metadata=metadata,
        )

    def info(self) -> Dict[str, Any]:
        """
        Return algorithm metadata.
        """
        information = super().info()

        information.update({
            "quantization_step": self.step,
            "extraction_type": "Blind",
        })

        return information


# ============================================================
# FUNCTIONAL WRAPPERS
# ============================================================

def embed_dct_qim(
    host: np.ndarray,
    watermark: np.ndarray,
    alpha: float = DCTQIMWatermarker.DEFAULT_ALPHA,
    key: int = DEFAULT_QIM_KEY,
) -> EmbeddingResult:
    """
    Functional wrapper around DCTQIMWatermarker.embed().
    """
    return DCTQIMWatermarker(
        alpha=alpha,
        key=key,
    ).embed(
        host,
        watermark,
    )


def extract_dct_qim(
    watermarked: np.ndarray,
    watermark_shape: Tuple[int, int],
    alpha: float = DCTQIMWatermarker.DEFAULT_ALPHA,
    key: int = DEFAULT_QIM_KEY,
) -> ExtractionResult:
    """
    Functional wrapper around DCTQIMWatermarker.extract_blind().
    """
    return DCTQIMWatermarker(
        alpha=alpha,
        key=key,
    ).extract_blind(
        watermarked,
        watermark_shape,
    )
