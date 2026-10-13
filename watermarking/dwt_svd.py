"""
AWSRE DWT-SVD Watermarking Module

Hybrid block-based invisible binary watermarking. A single-level 2-D DWT
is applied to the whole host image; the selected sub-band (HL by default,
matching watermarking/dwt.py's choice to avoid the visually sensitive LL
band) is then partitioned into non-overlapping blocks, and each watermark
bit is embedded by modulating selected singular values of that block's SVD
- the same block-SVD modulation rule used in watermarking/dct_svd.py,
applied in the wavelet domain instead of the DCT domain. Extraction is
non-blind and compares the original and processed sub-bands.

This file was previously an empty placeholder (referenced by
core/strategy_library.py and watermarking/registry.py's
BUILTIN_WATERMARKER_MODULES, but never implemented).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pywt

from watermarking.base import (
    BaseWatermarker,
    EmbeddingResult,
    ExtractionResult,
    Timer,
    build_metadata,
    ensure_grayscale,
    normalize_alpha,
)
from watermarking.dct import (
    DEFAULT_BLOCK_SIZE,
    ensure_uint8_image,
    extract_block,
    normalize_binary_watermark,
    place_block,
    required_host_dimensions,
    validate_embedding_capacity,
    validate_extraction_inputs,
)
from watermarking.registry import watermarker


DEFAULT_WAVELET = "haar"
DEFAULT_LEVEL = 1

SUPPORTED_SUBBANDS: Tuple[str, ...] = ("LL", "LH", "HL", "HH")
DEFAULT_SUBBAND = "HL"

DEFAULT_SINGULAR_INDICES: Tuple[int, ...] = (0, 1)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_subband(subband: str) -> str:
    normalized = str(subband).strip().upper()

    if normalized not in SUPPORTED_SUBBANDS:
        raise ValueError(
            f"Unsupported DWT subband: {subband}. "
            f"Supported: {', '.join(SUPPORTED_SUBBANDS)}."
        )

    return normalized


def validate_wavelet(wavelet: str) -> str:
    normalized = str(wavelet).strip()

    if not normalized:
        raise ValueError("wavelet cannot be empty.")

    try:
        pywt.Wavelet(normalized)
    except Exception as exc:
        raise ValueError(f"Unsupported wavelet: {normalized}.") from exc

    return normalized


def validate_singular_indices(
    singular_indices: Iterable[int],
    block_size: int,
) -> Tuple[int, ...]:
    values = tuple(int(index) for index in singular_indices)

    if not values:
        raise ValueError("At least one singular-value index is required.")

    if len(set(values)) != len(values):
        raise ValueError("singular_indices must not contain duplicates.")

    for index in values:
        if index < 0 or index >= block_size:
            raise ValueError(
                f"Singular-value index {index} is outside the valid "
                f"range [0, {block_size - 1}]."
            )

    return values


# ============================================================
# DWT SUBBAND HELPERS
# ============================================================

def decompose(image: np.ndarray, wavelet: str):
    """Perform level-1 two-dimensional DWT."""
    return pywt.dwt2(image.astype(np.float64), wavelet, mode="symmetric")


def select_subband(coefficients, subband: str) -> np.ndarray:
    ll, (lh, hl, hh) = coefficients
    bands = {"LL": ll, "LH": lh, "HL": hl, "HH": hh}
    return bands[subband]


def replace_subband(coefficients, subband: str, replacement: np.ndarray):
    ll, (lh, hl, hh) = coefficients

    if subband == "LL":
        ll = replacement
    elif subband == "LH":
        lh = replacement
    elif subband == "HL":
        hl = replacement
    else:
        hh = replacement

    return (ll, (lh, hl, hh))


def decompose_svd_block(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return U, S and Vt for the SVD of one wavelet sub-band block."""
    return np.linalg.svd(block.astype(np.float64), full_matrices=False)


def reconstruct_block(
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
) -> np.ndarray:
    return (u * singular_values) @ vt


# ============================================================
# DWT-SVD WATERMARKER
# ============================================================

@watermarker
class DWTSVDWatermarker(BaseWatermarker):
    """Hybrid block-based non-blind DWT-SVD binary watermarker."""

    METHOD = "DWT-SVD"

    DESCRIPTION = (
        "Hybrid invisible watermarking: level-1 DWT sub-band selection "
        "followed by block-wise singular-value modulation."
    )

    SUPPORTS_RGB = False
    SUPPORTS_GRAYSCALE = True

    DEFAULT_ALPHA = 20.0

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        wavelet: str = DEFAULT_WAVELET,
        level: int = DEFAULT_LEVEL,
        subband: str = DEFAULT_SUBBAND,
        block_size: int = DEFAULT_BLOCK_SIZE,
        singular_indices: Iterable[int] = DEFAULT_SINGULAR_INDICES,
    ) -> None:
        super().__init__(alpha=normalize_alpha(alpha))

        self.wavelet = validate_wavelet(wavelet)

        if int(level) != 1:
            raise ValueError(
                "The current AWSRE DWT-SVD implementation supports level=1 only."
            )
        self.level = 1

        self.subband = validate_subband(subband)

        self.block_size = int(block_size)
        if self.block_size <= 1:
            raise ValueError("block_size must be greater than 1.")

        self.singular_indices = validate_singular_indices(
            singular_indices,
            self.block_size,
        )

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    def maximum_watermark_shape(self, host: np.ndarray) -> Tuple[int, int]:
        """Return maximum binary-watermark shape supported by the host."""
        host_gray = ensure_grayscale(ensure_uint8_image(host))

        coefficients = decompose(host_gray, self.wavelet)
        selected_band = select_subband(coefficients, self.subband)

        return (
            selected_band.shape[0] // self.block_size,
            selected_band.shape[1] // self.block_size,
        )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    def embed(
        self,
        host: np.ndarray,
        watermark: np.ndarray,
    ) -> EmbeddingResult:
        """Embed a binary watermark and return image, runtime and metadata."""
        self.validate(host, watermark)

        host_gray = ensure_grayscale(ensure_uint8_image(host))
        watermark_binary = normalize_binary_watermark(watermark)

        positive_bits = 0
        negative_bits = 0

        with Timer() as timer:
            coefficients = decompose(host_gray, self.wavelet)
            selected_band = select_subband(coefficients, self.subband).copy()

            wm_height, wm_width = watermark_binary.shape

            for bit_row in range(wm_height):
                for bit_column in range(wm_width):
                    block = extract_block(
                        selected_band,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    u, singular_values, vt = decompose_svd_block(block)
                    modified = singular_values.astype(np.float64).copy()

                    bit = int(watermark_binary[bit_row, bit_column])
                    direction = 1.0 if bit == 1 else -1.0

                    for index in self.singular_indices:
                        modified[index] = max(
                            float(modified[index] + direction * self.alpha),
                            np.finfo(np.float64).eps,
                        )

                    reconstructed = reconstruct_block(u, modified, vt)

                    place_block(
                        selected_band,
                        reconstructed,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    if bit == 1:
                        positive_bits += 1
                    else:
                        negative_bits += 1

            modified_coefficients = replace_subband(
                coefficients,
                self.subband,
                selected_band,
            )

            reconstructed_image = pywt.idwt2(
                modified_coefficients,
                self.wavelet,
                mode="symmetric",
            )

            # DWT reconstruction may be one pixel larger for odd-sized input.
            reconstructed_image = reconstructed_image[
                :host_gray.shape[0],
                :host_gray.shape[1],
            ]

            watermarked_uint8 = np.uint8(
                np.clip(np.rint(reconstructed_image), 0, 255)
            )

        metadata: Dict[str, Any] = build_metadata(
            method=self.METHOD,
            alpha=self.alpha,
            host=host_gray,
            watermark=watermark_binary,
        )

        metadata.update({
            "wavelet": self.wavelet,
            "level": self.level,
            "subband": self.subband,
            "block_size": self.block_size,
            "singular_indices": list(self.singular_indices),
            "embedded_bits": int(watermark_binary.size),
            "positive_bits": positive_bits,
            "negative_bits": negative_bits,
            "watermark_shape": list(watermark_binary.shape),
            "host_shape": list(host_gray.shape),
            "selected_band_shape": list(selected_band.shape),
            "embedding_rate": float(watermark_binary.size / host_gray.size),
            "transform_pipeline": ["DWT", "SVD"],
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
        """Extract a binary watermark by singular-value difference voting."""
        original_gray = ensure_grayscale(ensure_uint8_image(original))
        watermarked_gray = ensure_grayscale(ensure_uint8_image(watermarked))

        normalized_shape = (int(watermark_shape[0]), int(watermark_shape[1]))

        extracted = np.zeros(normalized_shape, dtype=np.uint8)
        mean_differences = []

        with Timer() as timer:
            original_coefficients = decompose(original_gray, self.wavelet)
            watermarked_coefficients = decompose(watermarked_gray, self.wavelet)

            original_band = select_subband(original_coefficients, self.subband)
            watermarked_band = select_subband(watermarked_coefficients, self.subband)

            validate_extraction_inputs(
                original_band,
                watermarked_band,
                normalized_shape,
                self.block_size,
            )

            wm_height, wm_width = normalized_shape

            for bit_row in range(wm_height):
                for bit_column in range(wm_width):
                    original_block = extract_block(
                        original_band,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )
                    processed_block = extract_block(
                        watermarked_band,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    _, original_s, _ = decompose_svd_block(original_block)
                    _, processed_s, _ = decompose_svd_block(processed_block)

                    differences = [
                        float(processed_s[index] - original_s[index])
                        for index in self.singular_indices
                    ]
                    average_difference = float(np.mean(differences))

                    extracted[bit_row, bit_column] = (
                        1 if average_difference > 0.0 else 0
                    )
                    mean_differences.append(average_difference)

        differences_array = np.asarray(mean_differences, dtype=np.float64)

        metadata = {
            "method": self.METHOD,
            "alpha": self.alpha,
            "wavelet": self.wavelet,
            "level": self.level,
            "subband": self.subband,
            "block_size": self.block_size,
            "singular_indices": list(self.singular_indices),
            "watermark_shape": list(normalized_shape),
            "extracted_bits": int(extracted.size),
            "mean_singular_difference": float(np.mean(differences_array)),
            "mean_absolute_singular_difference": float(
                np.mean(np.abs(differences_array))
            ),
            "transform_pipeline": ["DWT", "SVD"],
            "non_blind_extraction": True,
        }

        return ExtractionResult(
            extracted_watermark=extracted,
            runtime=timer.elapsed,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # VALIDATION / INFO
    # --------------------------------------------------------

    def validate(self, host: np.ndarray, watermark: np.ndarray) -> None:
        """Perform generic and DWT-SVD-specific validation."""
        super().validate(host, watermark)

        host_gray = ensure_grayscale(ensure_uint8_image(np.asarray(host)))
        watermark_binary = normalize_binary_watermark(watermark)

        coefficients = decompose(host_gray, self.wavelet)
        selected_band = select_subband(coefficients, self.subband)

        # validate_embedding_capacity() only inspects .shape, so the
        # DWT sub-band can stand in for "host" here - embedding
        # capacity is bounded by the sub-band's size, not the full
        # host image's size (level-1 DWT halves each dimension).
        validate_embedding_capacity(
            selected_band,
            watermark_binary,
            self.block_size,
        )

    def info(self) -> Dict[str, Any]:
        information = super().info()
        information.update({
            "wavelet": self.wavelet,
            "level": self.level,
            "subband": self.subband,
            "block_size": self.block_size,
            "singular_indices": list(self.singular_indices),
            "extraction_type": "Non-blind",
            "watermark_representation": "Binary",
            "transform_pipeline": ["DWT", "SVD"],
        })
        return information


def embed_dwt_svd(
    host: np.ndarray,
    watermark: np.ndarray,
    alpha: float = DWTSVDWatermarker.DEFAULT_ALPHA,
) -> EmbeddingResult:
    """Functional wrapper around DWTSVDWatermarker.embed()."""
    return DWTSVDWatermarker(alpha=alpha).embed(host, watermark)


def extract_dwt_svd(
    original: np.ndarray,
    watermarked: np.ndarray,
    watermark_shape: Tuple[int, int],
    alpha: float = DWTSVDWatermarker.DEFAULT_ALPHA,
) -> ExtractionResult:
    """Functional wrapper around DWTSVDWatermarker.extract()."""
    return DWTSVDWatermarker(alpha=alpha).extract(
        original,
        watermarked,
        watermark_shape,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    host = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
    watermark = rng.integers(0, 2, size=(16, 16), dtype=np.uint8)

    algorithm = DWTSVDWatermarker(alpha=20)

    embedded = algorithm.embed(host, watermark)
    extracted = algorithm.extract(
        host,
        embedded.watermarked_image,
        watermark.shape,
    )

    ber = float(np.mean(watermark != extracted.extracted_watermark))

    assert embedded.watermarked_image.shape == host.shape
    assert extracted.extracted_watermark.shape == watermark.shape

    print("DWT-SVD self-test BER:", ber)
    print("DWT-SVD watermarker self test passed.")
