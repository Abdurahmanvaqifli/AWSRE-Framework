"""
AWSRE Improved DCT Watermarking Module

This module implements block-based invisible watermark embedding
and non-blind watermark extraction using multiple mid-frequency
DCT coefficients.

The implementation is reusable by:

- AWSRE-Bench
- Streamlit web platform
- FastAPI backend
- Desktop application
- Research experiments
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np

from watermarking.base import (
    BaseWatermarker,
    EmbeddingResult,
    ExtractionResult,
    Timer,
    build_metadata,
    ensure_grayscale,
    normalize_alpha,
    validate_host_image,
    validate_watermark,
)
from watermarking.registry import watermarker


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_BLOCK_SIZE = 8

DEFAULT_COEFFICIENTS: Tuple[Tuple[int, int], ...] = (
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image safely to uint8.
    """
    array = np.asarray(image)

    if array.size == 0:
        raise ValueError("Image array is empty.")

    if array.dtype == np.uint8:
        return array.copy()

    return np.uint8(
        np.clip(array, 0, 255)
    )


def normalize_binary_watermark(
    watermark: np.ndarray,
) -> np.ndarray:
    """
    Convert a watermark to a binary 0/1 uint8 matrix.

    Supported inputs:
    - 0/1
    - 0/255
    - grayscale
    - floating-point arrays
    """
    validate_watermark(watermark)

    wm = ensure_grayscale(
        np.asarray(watermark)
    ).astype(np.float32)

    if np.max(wm) <= 1.0:
        binary = wm >= 0.5
    else:
        binary = wm >= 127.5

    return binary.astype(np.uint8)


def validate_coefficients(
    coefficients: Iterable[Tuple[int, int]],
    block_size: int,
) -> Tuple[Tuple[int, int], ...]:
    """
    Validate DCT coefficient coordinates.
    """
    validated = []

    for coefficient in coefficients:
        if (
            not isinstance(coefficient, tuple)
            or len(coefficient) != 2
        ):
            raise ValueError(
                "Every coefficient must be a tuple: (row, column)."
            )

        row, column = coefficient

        if not (
            0 <= int(row) < block_size
            and 0 <= int(column) < block_size
        ):
            raise ValueError(
                f"Coefficient {coefficient} is outside "
                f"a {block_size}x{block_size} block."
            )

        validated.append(
            (int(row), int(column))
        )

    if not validated:
        raise ValueError(
            "At least one DCT coefficient is required."
        )

    return tuple(validated)


def required_host_dimensions(
    watermark_shape: Tuple[int, int],
    block_size: int,
) -> Tuple[int, int]:
    """
    Calculate minimum host dimensions required to embed
    every watermark bit into one image block.
    """
    wm_height, wm_width = watermark_shape

    required_height = wm_height * block_size
    required_width = wm_width * block_size

    return required_height, required_width


def validate_embedding_capacity(
    host: np.ndarray,
    watermark: np.ndarray,
    block_size: int,
) -> None:
    """
    Verify that the host has enough non-overlapping blocks
    for the supplied watermark.
    """
    host_height, host_width = host.shape[:2]

    required_height, required_width = required_host_dimensions(
        watermark.shape,
        block_size,
    )

    if (
        host_height < required_height
        or host_width < required_width
    ):
        raise ValueError(
            "Host image is too small for this watermark. "
            f"Host: {host_width}x{host_height}; "
            f"required: {required_width}x{required_height}; "
            f"watermark: {watermark.shape[1]}x{watermark.shape[0]}; "
            f"block size: {block_size}."
        )


def validate_extraction_inputs(
    original: np.ndarray,
    watermarked: np.ndarray,
    watermark_shape: Tuple[int, int],
    block_size: int,
) -> None:
    """
    Validate original and processed images before extraction.
    """
    validate_host_image(original)
    validate_host_image(watermarked)

    if original.shape != watermarked.shape:
        raise ValueError(
            "Original and watermarked image shapes must match. "
            f"Received {original.shape} and {watermarked.shape}."
        )

    if len(watermark_shape) != 2:
        raise ValueError(
            "watermark_shape must be (height, width)."
        )

    wm_height = int(watermark_shape[0])
    wm_width = int(watermark_shape[1])

    if wm_height <= 0 or wm_width <= 0:
        raise ValueError(
            "Watermark dimensions must be positive."
        )

    required_height, required_width = required_host_dimensions(
        (wm_height, wm_width),
        block_size,
    )

    host_height, host_width = original.shape[:2]

    if (
        host_height < required_height
        or host_width < required_width
    ):
        raise ValueError(
            "Images are too small for the requested extraction shape."
        )


def block_coordinates(
    bit_row: int,
    bit_column: int,
    block_size: int,
) -> Tuple[int, int, int, int]:
    """
    Return block boundaries for one watermark bit.
    """
    y_start = bit_row * block_size
    x_start = bit_column * block_size

    y_end = y_start + block_size
    x_end = x_start + block_size

    return y_start, y_end, x_start, x_end


def extract_block(
    image: np.ndarray,
    bit_row: int,
    bit_column: int,
    block_size: int,
) -> np.ndarray:
    """
    Extract one non-overlapping image block.
    """
    y_start, y_end, x_start, x_end = block_coordinates(
        bit_row,
        bit_column,
        block_size,
    )

    block = image[
        y_start:y_end,
        x_start:x_end,
    ]

    expected_shape = (
        block_size,
        block_size,
    )

    if block.shape != expected_shape:
        raise ValueError(
            f"Invalid block shape: {block.shape}; "
            f"expected: {expected_shape}."
        )

    return block


def place_block(
    image: np.ndarray,
    block: np.ndarray,
    bit_row: int,
    bit_column: int,
    block_size: int,
) -> None:
    """
    Place a reconstructed block back into an image.
    """
    y_start, y_end, x_start, x_end = block_coordinates(
        bit_row,
        bit_column,
        block_size,
    )

    image[
        y_start:y_end,
        x_start:x_end,
    ] = block


# ============================================================
# DCT WATERMARKER
# ============================================================

@watermarker
class DCTWatermarker(BaseWatermarker):
    """
    Improved block-based DCT watermarker.

    One watermark bit is embedded into multiple mid-frequency
    coefficients of one 8x8 host-image block.

    Extraction is non-blind because it compares coefficients
    from the original and processed images.
    """

    METHOD = "DCT"

    DESCRIPTION = (
        "Block-based invisible watermarking using multiple "
        "mid-frequency DCT coefficients."
    )

    SUPPORTS_RGB = False
    SUPPORTS_GRAYSCALE = True

    DEFAULT_ALPHA = 10

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        block_size: int = DEFAULT_BLOCK_SIZE,
        coefficients: Iterable[
            Tuple[int, int]
        ] = DEFAULT_COEFFICIENTS,
    ) -> None:
        super().__init__(
            alpha=normalize_alpha(alpha)
        )

        self.block_size = int(block_size)

        if self.block_size <= 0:
            raise ValueError(
                "block_size must be positive."
            )

        self.coefficients = validate_coefficients(
            coefficients,
            self.block_size,
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
        Embed a binary watermark into the host image.

        Parameters
        ----------
        host:
            Host image. RGB input is converted to grayscale.
        watermark:
            Binary or grayscale watermark.

        Returns
        -------
        EmbeddingResult
            Watermarked image, runtime and metadata.
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

        host_float = host_gray.astype(
            np.float32
        )

        watermarked_float = host_float.copy()

        embedded_bits = 0
        positive_bits = 0
        negative_bits = 0

        with Timer() as timer:
            wm_height, wm_width = (
                watermark_binary.shape
            )

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

                    direction = (
                        1.0
                        if bit == 1
                        else -1.0
                    )

                    for coefficient in self.coefficients:
                        dct_block[
                            coefficient
                        ] += (
                            direction
                            * self.alpha
                        )

                    reconstructed = cv2.idct(
                        dct_block
                    )

                    place_block(
                        watermarked_float,
                        reconstructed,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    embedded_bits += 1

                    if bit == 1:
                        positive_bits += 1
                    else:
                        negative_bits += 1

        watermarked_uint8 = np.uint8(
            np.clip(
                np.rint(watermarked_float),
                0,
                255,
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
            "embedded_bits": embedded_bits,
            "positive_bits": positive_bits,
            "negative_bits": negative_bits,
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
                embedded_bits
                / host_gray.size
            ),
            "non_blind_extraction": True,
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
        original: np.ndarray,
        watermarked: np.ndarray,
        watermark_shape: Tuple[int, int],
    ) -> ExtractionResult:
        """
        Extract the embedded watermark.

        Parameters
        ----------
        original:
            Original host image.
        watermarked:
            Watermarked or attacked watermarked image.
        watermark_shape:
            Watermark shape as (height, width).

        Returns
        -------
        ExtractionResult
            Extracted binary watermark with values 0 and 1.
        """
        original_gray = ensure_grayscale(
            ensure_uint8_image(original)
        )

        watermarked_gray = ensure_grayscale(
            ensure_uint8_image(watermarked)
        )

        normalized_shape = (
            int(watermark_shape[0]),
            int(watermark_shape[1]),
        )

        validate_extraction_inputs(
            original_gray,
            watermarked_gray,
            normalized_shape,
            self.block_size,
        )

        original_float = original_gray.astype(
            np.float32
        )

        watermarked_float = watermarked_gray.astype(
            np.float32
        )

        wm_height, wm_width = normalized_shape

        extracted = np.zeros(
            normalized_shape,
            dtype=np.uint8,
        )

        coefficient_differences = []

        with Timer() as timer:
            for bit_row in range(wm_height):
                for bit_column in range(wm_width):
                    original_block = extract_block(
                        original_float,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    processed_block = extract_block(
                        watermarked_float,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    original_dct = cv2.dct(
                        original_block.astype(
                            np.float32
                        )
                    )

                    processed_dct = cv2.dct(
                        processed_block.astype(
                            np.float32
                        )
                    )

                    differences = [
                        float(
                            processed_dct[coefficient]
                            - original_dct[coefficient]
                        )
                        for coefficient
                        in self.coefficients
                    ]

                    average_difference = float(
                        np.mean(differences)
                    )

                    extracted[
                        bit_row,
                        bit_column,
                    ] = (
                        1
                        if average_difference > 0
                        else 0
                    )

                    coefficient_differences.append(
                        average_difference
                    )

        differences_array = np.asarray(
            coefficient_differences,
            dtype=np.float64,
        )

        metadata = {
            "method": self.METHOD,
            "alpha": self.alpha,
            "block_size": self.block_size,
            "coefficients": [
                list(item)
                for item in self.coefficients
            ],
            "watermark_shape": list(
                normalized_shape
            ),
            "extracted_bits": int(
                extracted.size
            ),
            "mean_coefficient_difference": float(
                np.mean(differences_array)
            ),
            "mean_absolute_coefficient_difference": float(
                np.mean(
                    np.abs(
                        differences_array
                    )
                )
            ),
            "minimum_coefficient_difference": float(
                np.min(differences_array)
            ),
            "maximum_coefficient_difference": float(
                np.max(differences_array)
            ),
            "non_blind_extraction": True,
        }

        return ExtractionResult(
            extracted_watermark=extracted,
            runtime=timer.elapsed,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    def maximum_watermark_shape(
        self,
        host: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Return maximum watermark dimensions supported by host.
        """
        host_gray = ensure_grayscale(
            ensure_uint8_image(host)
        )

        host_height, host_width = (
            host_gray.shape
        )

        max_height = (
            host_height
            // self.block_size
        )

        max_width = (
            host_width
            // self.block_size
        )

        return max_height, max_width

    def validate(
        self,
        host: np.ndarray,
        watermark: np.ndarray,
    ) -> None:
        """
        Perform generic and DCT-specific validation.
        """
        super().validate(
            host,
            watermark,
        )

        host_gray = ensure_grayscale(
            np.asarray(host)
        )

        watermark_binary = normalize_binary_watermark(
            watermark
        )

        validate_embedding_capacity(
            host_gray,
            watermark_binary,
            self.block_size,
        )

    def info(self) -> Dict[str, Any]:
        """
        Return algorithm metadata.
        """
        information = super().info()

        information.update({
            "block_size": self.block_size,
            "coefficients": [
                list(item)
                for item in self.coefficients
            ],
            "extraction_type": "Non-blind",
            "watermark_representation": "Binary",
        })

        return information


# ============================================================
# FUNCTIONAL WRAPPERS
# ============================================================

def embed_dct(
    host: np.ndarray,
    watermark: np.ndarray,
    alpha: float = DCTWatermarker.DEFAULT_ALPHA,
) -> EmbeddingResult:
    """
    Functional wrapper around DCTWatermarker.embed().
    """
    algorithm = DCTWatermarker(
        alpha=alpha
    )

    return algorithm.embed(
        host,
        watermark,
    )


def extract_dct(
    original: np.ndarray,
    watermarked: np.ndarray,
    watermark_shape: Tuple[int, int],
    alpha: float = DCTWatermarker.DEFAULT_ALPHA,
) -> ExtractionResult:
    """
    Functional wrapper around DCTWatermarker.extract().
    """
    algorithm = DCTWatermarker(
        alpha=alpha
    )

    return algorithm.extract(
        original,
        watermarked,
        watermark_shape,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    from watermarking.metrics import (
        calculate_ber,
        calculate_correlation,
        calculate_psnr,
        calculate_ssim,
    )

    rng = np.random.default_rng(
        seed=42
    )

    test_host = rng.integers(
        0,
        256,
        size=(512, 512),
        dtype=np.uint8,
    )

    test_watermark = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    test_watermark[
        ::2,
        ::2,
    ] = 1

    test_watermark[
        1::2,
        1::2,
    ] = 1

    dct = DCTWatermarker(
        alpha=20
    )

    embedding_result = dct.embed(
        test_host,
        test_watermark,
    )

    extraction_result = dct.extract(
        test_host,
        embedding_result.watermarked_image,
        test_watermark.shape,
    )

    extracted = (
        extraction_result.extracted_watermark
    )

    ber = calculate_ber(
        test_watermark,
        extracted,
    )

    correlation = calculate_correlation(
        test_watermark,
        extracted,
    )

    psnr = calculate_psnr(
        test_host,
        embedding_result.watermarked_image,
    )

    ssim = calculate_ssim(
        test_host,
        embedding_result.watermarked_image,
    )

    print("=" * 68)
    print("AWSRE DCT WATERMARKER SELF TEST")
    print("=" * 68)

    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.6f}")
    print(f"BER: {ber:.8f}")
    print(f"Correlation: {correlation:.6f}")
    print(
        "Embedding runtime:",
        f"{embedding_result.runtime:.6f} s",
    )
    print(
        "Extraction runtime:",
        f"{extraction_result.runtime:.6f} s",
    )

    if ber > 0.02:
        raise RuntimeError(
            "DCT self test failed: BER is too high."
        )

    print("\n✅ DCT self test passed.")
