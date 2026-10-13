"""
AWSRE DWT Watermarking Module

Level-1, non-blind DWT watermark embedding and extraction.

Improved version:
- distributes watermark bits across the selected DWT subband
- avoids concentrating the watermark in the top-left region
- uses a deterministic embedding pattern
- defaults to the HL subband instead of LL
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pywt

from watermarking.base import (
    BaseWatermarker,
    EmbeddingResult,
    ExtractionResult,
    Timer,
    ensure_grayscale,
    normalize_alpha,
    validate_host_image,
    validate_watermark,
)
from watermarking.registry import watermarker


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_SUBBANDS = (
    "LL",
    "LH",
    "HL",
    "HH",
)

DEFAULT_WAVELET = "haar"

# Important change:
# LL contains most of the visible low-frequency image energy.
# HL is less visually sensitive for invisible watermarking.
DEFAULT_SUBBAND = "HL"

DEFAULT_LEVEL = 1
DEFAULT_ALPHA = 10.0

# Fixed seed guarantees that embedding and extraction select
# exactly the same DWT coefficient positions.
DEFAULT_SEED = 2026


# ============================================================
# BASIC HELPERS
# ============================================================

def _ensure_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert image data safely to uint8.
    """
    array = np.asarray(
        image
    )

    if array.size == 0:
        raise ValueError(
            "Image array is empty."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "Image contains NaN or infinite values."
        )

    if array.dtype == np.uint8:
        return array.copy()

    return np.uint8(
        np.clip(
            np.rint(array),
            0,
            255,
        )
    )


def _normalize_watermark(
    watermark: np.ndarray,
) -> np.ndarray:
    """
    Convert watermark to a binary array containing 0 and 1.
    """
    validate_watermark(
        watermark
    )

    array = ensure_grayscale(
        np.asarray(
            watermark
        )
    ).astype(
        np.float64
    )

    if array.size == 0:
        raise ValueError(
            "Watermark image is empty."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "Watermark contains NaN or infinite values."
        )

    threshold = (
        0.5
        if float(
            np.max(array)
        ) <= 1.0
        else 127.5
    )

    return (
        array >= threshold
    ).astype(
        np.uint8
    )


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def _validate_subband(
    subband: str,
) -> str:
    """
    Validate DWT subband name.
    """
    normalized = (
        str(subband)
        .strip()
        .upper()
    )

    if normalized not in SUPPORTED_SUBBANDS:
        raise ValueError(
            f"Unsupported DWT subband: {subband}. "
            f"Supported: {', '.join(SUPPORTED_SUBBANDS)}."
        )

    return normalized


def _validate_level(
    level: int,
) -> int:
    """
    AWSRE currently supports one-level DWT.
    """
    normalized = int(
        level
    )

    if normalized != 1:
        raise ValueError(
            "The current AWSRE DWT implementation "
            "supports level=1 only."
        )

    return normalized


def _validate_wavelet(
    wavelet: str,
) -> str:
    """
    Validate PyWavelets wavelet name.
    """
    normalized = str(
        wavelet
    ).strip()

    if not normalized:
        raise ValueError(
            "wavelet cannot be empty."
        )

    try:
        pywt.Wavelet(
            normalized
        )

    except Exception as exc:
        raise ValueError(
            f"Unsupported wavelet: {normalized}."
        ) from exc

    return normalized


# ============================================================
# DWT HELPERS
# ============================================================

def _decompose(
    image: np.ndarray,
    wavelet: str,
):
    """
    Perform level-1 two-dimensional DWT.
    """
    return pywt.dwt2(
        image.astype(
            np.float64
        ),
        wavelet,
        mode="symmetric",
    )


def _select_subband(
    coefficients,
    subband: str,
) -> np.ndarray:
    """
    Return the requested DWT subband.
    """
    ll, (
        lh,
        hl,
        hh,
    ) = coefficients

    bands = {
        "LL": ll,
        "LH": lh,
        "HL": hl,
        "HH": hh,
    }

    return bands[
        subband
    ]


def _replace_subband(
    coefficients,
    subband: str,
    replacement: np.ndarray,
):
    """
    Replace one DWT subband and preserve the others.
    """
    ll, (
        lh,
        hl,
        hh,
    ) = coefficients

    if subband == "LL":
        ll = replacement

    elif subband == "LH":
        lh = replacement

    elif subband == "HL":
        hl = replacement

    else:
        hh = replacement

    return (
        ll,
        (
            lh,
            hl,
            hh,
        ),
    )


# ============================================================
# DISTRIBUTED EMBEDDING
# ============================================================

def _distributed_indices(
    band_shape: Tuple[int, int],
    number_of_bits: int,
    seed: int,
) -> np.ndarray:
    """
    Select coefficient positions distributed across the entire
    DWT subband.

    The same seed guarantees identical positions during
    embedding and extraction.
    """
    height = int(
        band_shape[0]
    )

    width = int(
        band_shape[1]
    )

    total_coefficients = (
        height * width
    )

    number_of_bits = int(
        number_of_bits
    )

    if number_of_bits <= 0:
        raise ValueError(
            "The watermark must contain at least one bit."
        )

    if number_of_bits > total_coefficients:
        raise ValueError(
            "Watermark requires more coefficients than "
            "the selected DWT subband contains. "
            f"Required: {number_of_bits}; "
            f"available: {total_coefficients}."
        )

    rng = np.random.default_rng(
        int(seed)
    )

    indices = rng.choice(
        total_coefficients,
        size=number_of_bits,
        replace=False,
    )

    return np.asarray(
        indices,
        dtype=np.int64,
    )


def _validate_capacity(
    selected_band: np.ndarray,
    watermark: np.ndarray,
) -> None:
    """
    Check whether the DWT subband contains enough coefficients.
    """
    required_bits = int(
        watermark.size
    )

    available_coefficients = int(
        selected_band.size
    )

    if required_bits > available_coefficients:
        raise ValueError(
            "Watermark exceeds DWT embedding capacity. "
            f"Watermark bits: {required_bits}; "
            f"available coefficients: "
            f"{available_coefficients}."
        )


# ============================================================
# EXTRACTION VALIDATION
# ============================================================

def _validate_extraction(
    original: np.ndarray,
    processed: np.ndarray,
    watermark_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Validate extraction inputs.
    """
    validate_host_image(
        original
    )

    validate_host_image(
        processed
    )

    if original.shape != processed.shape:
        raise ValueError(
            "Original and processed images must have "
            "identical shapes."
        )

    if len(
        watermark_shape
    ) != 2:
        raise ValueError(
            "watermark_shape must be (height, width)."
        )

    height = int(
        watermark_shape[0]
    )

    width = int(
        watermark_shape[1]
    )

    if (
        height <= 0
        or width <= 0
    ):
        raise ValueError(
            "Watermark dimensions must be positive."
        )

    return (
        height,
        width,
    )


# ============================================================
# METADATA
# ============================================================

def _embedding_metadata(
    *,
    alpha: float,
    wavelet: str,
    level: int,
    subband: str,
    watermark_shape: Tuple[int, int],
    host_shape: Tuple[int, ...],
    seed: int,
) -> Dict[str, Any]:
    """
    Build metadata describing the embedding process.
    """
    return {
        "method": "DWT",
        "family": "wavelet_transform",
        "alpha": float(
            alpha
        ),
        "wavelet": wavelet,
        "level": int(
            level
        ),
        "subband": subband,
        "blind": False,
        "supports_rgb": False,
        "supports_grayscale": True,
        "watermark_height": int(
            watermark_shape[0]
        ),
        "watermark_width": int(
            watermark_shape[1]
        ),
        "watermark_bits": int(
            watermark_shape[0]
            * watermark_shape[1]
        ),
        "host_shape": list(
            host_shape
        ),
        "distribution": (
            "deterministic distributed coefficient embedding"
        ),
        "seed": int(
            seed
        ),
        "embedding_rule": (
            "selected distributed coefficient += alpha "
            "for bit 1 and -= alpha for bit 0"
        ),
    }


# ============================================================
# DWT WATERMARKER
# ============================================================

@watermarker
class DWTWatermarker(
    BaseWatermarker
):
    """
    Non-blind level-1 distributed DWT watermarker.
    """

    METHOD = "DWT"

    DESCRIPTION = (
        "Non-blind level-1 DWT watermarking with "
        "distributed coefficient embedding for improved "
        "visual imperceptibility."
    )

    SUPPORTS_RGB = False
    SUPPORTS_GRAYSCALE = True

    DEFAULT_ALPHA = DEFAULT_ALPHA

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        wavelet: str = DEFAULT_WAVELET,
        level: int = DEFAULT_LEVEL,
        subband: str = DEFAULT_SUBBAND,
        seed: int = DEFAULT_SEED,
    ) -> None:

        super().__init__(
            alpha=normalize_alpha(
                alpha
            )
        )

        self.wavelet = (
            _validate_wavelet(
                wavelet
            )
        )

        self.level = (
            _validate_level(
                level
            )
        )

        self.subband = (
            _validate_subband(
                subband
            )
        )

        self.seed = int(
            seed
        )

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    def maximum_watermark_shape(
        self,
        host: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Return the largest square binary watermark that can be
        embedded using one coefficient per watermark bit.
        """
        validate_host_image(
            host
        )

        host_gray = _ensure_uint8(
            ensure_grayscale(
                np.asarray(
                    host
                )
            )
        )

        coefficients = _decompose(
            host_gray,
            self.wavelet,
        )

        selected_band = _select_subband(
            coefficients,
            self.subband,
        )

        capacity = int(
            selected_band.size
        )

        side = int(
            np.floor(
                np.sqrt(
                    capacity
                )
            )
        )

        side = max(
            side,
            1,
        )

        return (
            side,
            side,
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
        Embed a binary watermark across distributed DWT
        coefficient locations.
        """
        validate_host_image(
            host
        )

        validate_watermark(
            watermark
        )

        host_gray = _ensure_uint8(
            ensure_grayscale(
                np.asarray(
                    host
                )
            )
        )

        watermark_binary = (
            _normalize_watermark(
                watermark
            )
        )

        with Timer() as timer:

            coefficients = _decompose(
                host_gray,
                self.wavelet,
            )

            selected = _select_subband(
                coefficients,
                self.subband,
            ).copy()

            _validate_capacity(
                selected,
                watermark_binary,
            )

            watermark_flat = (
                watermark_binary
                .astype(
                    np.float64
                )
                .reshape(
                    -1
                )
            )

            # Convert:
            # bit 0 -> -1
            # bit 1 -> +1
            signs = (
                watermark_flat
                * 2.0
                - 1.0
            )

            indices = _distributed_indices(
                band_shape=selected.shape,
                number_of_bits=(
                    watermark_binary.size
                ),
                seed=self.seed,
            )

            selected_flat = (
                selected.reshape(
                    -1
                )
            )

            selected_flat[
                indices
            ] += (
                self.alpha
                * signs
            )

            selected = (
                selected_flat.reshape(
                    selected.shape
                )
            )

            modified_coefficients = (
                _replace_subband(
                    coefficients,
                    self.subband,
                    selected,
                )
            )

            reconstructed = pywt.idwt2(
                modified_coefficients,
                self.wavelet,
                mode="symmetric",
            )

            # DWT reconstruction may be one pixel larger
            # for odd-sized input images.
            reconstructed = reconstructed[
                :host_gray.shape[0],
                :host_gray.shape[1],
            ]

            watermarked = (
                _ensure_uint8(
                    reconstructed
                )
            )

        metadata = _embedding_metadata(
            alpha=self.alpha,
            wavelet=self.wavelet,
            level=self.level,
            subband=self.subband,
            watermark_shape=(
                watermark_binary.shape
            ),
            host_shape=(
                host_gray.shape
            ),
            seed=self.seed,
        )

        metadata[
            "selected_band_shape"
        ] = list(
            selected.shape
        )

        metadata[
            "embedded_coefficients"
        ] = int(
            watermark_binary.size
        )

        return EmbeddingResult(
            watermarked_image=watermarked,
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
        Extract the watermark using the original host image.

        This is a non-blind extraction method.
        """
        (
            wm_height,
            wm_width,
        ) = _validate_extraction(
            original,
            watermarked,
            watermark_shape,
        )

        original_gray = _ensure_uint8(
            ensure_grayscale(
                np.asarray(
                    original
                )
            )
        )

        processed_gray = _ensure_uint8(
            ensure_grayscale(
                np.asarray(
                    watermarked
                )
            )
        )

        with Timer() as timer:

            original_coefficients = (
                _decompose(
                    original_gray,
                    self.wavelet,
                )
            )

            processed_coefficients = (
                _decompose(
                    processed_gray,
                    self.wavelet,
                )
            )

            original_band = (
                _select_subband(
                    original_coefficients,
                    self.subband,
                )
            )

            processed_band = (
                _select_subband(
                    processed_coefficients,
                    self.subband,
                )
            )

            number_of_bits = int(
                wm_height
                * wm_width
            )

            if (
                number_of_bits
                > original_band.size
            ):
                raise ValueError(
                    "Requested watermark shape exceeds "
                    "the selected DWT band's capacity."
                )

            indices = _distributed_indices(
                band_shape=(
                    original_band.shape
                ),
                number_of_bits=(
                    number_of_bits
                ),
                seed=self.seed,
            )

            original_flat = (
                original_band.reshape(
                    -1
                )
            )

            processed_flat = (
                processed_band.reshape(
                    -1
                )
            )

            difference = (
                processed_flat[
                    indices
                ]
                - original_flat[
                    indices
                ]
            )

            extracted_flat = (
                difference >= 0.0
            ).astype(
                np.uint8
            )

            extracted = (
                extracted_flat.reshape(
                    wm_height,
                    wm_width,
                )
            )

        metadata = {
            "method": self.METHOD,
            "alpha": float(
                self.alpha
            ),
            "wavelet": self.wavelet,
            "level": int(
                self.level
            ),
            "subband": self.subband,
            "blind": False,
            "seed": int(
                self.seed
            ),
            "distribution": (
                "deterministic distributed coefficient embedding"
            ),
            "watermark_height": int(
                wm_height
            ),
            "watermark_width": int(
                wm_width
            ),
            "watermark_bits": int(
                number_of_bits
            ),
            "selected_band_shape": list(
                original_band.shape
            ),
        }

        return ExtractionResult(
            extracted_watermark=extracted,
            runtime=timer.elapsed,
            metadata=metadata,
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    rng = np.random.default_rng(
        seed=42
    )

    host = rng.integers(
        0,
        256,
        size=(
            512,
            512,
        ),
        dtype=np.uint8,
    )

    watermark = rng.integers(
        0,
        2,
        size=(
            32,
            32,
        ),
        dtype=np.uint8,
    )

    algorithm = DWTWatermarker(
        alpha=20,
        wavelet="haar",
        subband="HL",
        seed=2026,
    )

    embedded = algorithm.embed(
        host,
        watermark,
    )

    extracted = algorithm.extract(
        host,
        embedded.watermarked_image,
        watermark.shape,
    )

    ber = float(
        np.mean(
            watermark
            != extracted.extracted_watermark
        )
    )

    assert (
        embedded.watermarked_image.shape
        == host.shape
    )

    assert (
        extracted.extracted_watermark.shape
        == watermark.shape
    )

    print(
        "DWT self-test BER:",
        ber,
    )

    print(
        "✅ DWT distributed watermarker self test passed."
    )