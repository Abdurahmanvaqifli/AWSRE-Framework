"""
AWSRE DWT Watermarking Module

Level-1, non-blind DWT watermark embedding and extraction.
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


SUPPORTED_SUBBANDS = ("LL", "LH", "HL", "HH")
DEFAULT_WAVELET = "haar"
DEFAULT_SUBBAND = "LL"
DEFAULT_LEVEL = 1
DEFAULT_ALPHA = 10.0


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)

    if array.size == 0:
        raise ValueError("Image array is empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError("Image contains NaN or infinite values.")

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
    validate_watermark(watermark)

    array = ensure_grayscale(
        np.asarray(watermark)
    ).astype(np.float64)

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Watermark contains NaN or infinite values."
        )

    threshold = (
        0.5
        if float(np.max(array)) <= 1.0
        else 127.5
    )

    return (
        array >= threshold
    ).astype(np.uint8)


def _validate_subband(subband: str) -> str:
    normalized = str(subband).strip().upper()

    if normalized not in SUPPORTED_SUBBANDS:
        raise ValueError(
            f"Unsupported DWT subband: {subband}. "
            f"Supported: {', '.join(SUPPORTED_SUBBANDS)}."
        )

    return normalized


def _validate_level(level: int) -> int:
    normalized = int(level)

    if normalized != 1:
        raise ValueError(
            "The initial AWSRE DWT implementation supports "
            "level=1 only."
        )

    return normalized


def _validate_wavelet(wavelet: str) -> str:
    normalized = str(wavelet).strip()

    if not normalized:
        raise ValueError("wavelet cannot be empty.")

    try:
        pywt.Wavelet(normalized)
    except Exception as exc:
        raise ValueError(
            f"Unsupported wavelet: {normalized}."
        ) from exc

    return normalized


def _decompose(
    image: np.ndarray,
    wavelet: str,
):
    return pywt.dwt2(
        image.astype(np.float64),
        wavelet,
        mode="symmetric",
    )


def _select_subband(
    coefficients,
    subband: str,
) -> np.ndarray:
    ll, (lh, hl, hh) = coefficients

    return {
        "LL": ll,
        "LH": lh,
        "HL": hl,
        "HH": hh,
    }[subband]


def _replace_subband(
    coefficients,
    subband: str,
    replacement: np.ndarray,
):
    ll, (lh, hl, hh) = coefficients

    if subband == "LL":
        ll = replacement
    elif subband == "LH":
        lh = replacement
    elif subband == "HL":
        hl = replacement
    else:
        hh = replacement

    return ll, (lh, hl, hh)


def _validate_capacity(
    selected_band: np.ndarray,
    watermark: np.ndarray,
) -> None:
    wm_height, wm_width = watermark.shape
    band_height, band_width = selected_band.shape

    if (
        wm_height > band_height
        or wm_width > band_width
    ):
        raise ValueError(
            "Watermark is larger than the selected DWT band. "
            f"Watermark: {wm_width}x{wm_height}; "
            f"band: {band_width}x{band_height}."
        )


def _validate_extraction(
    original: np.ndarray,
    processed: np.ndarray,
    watermark_shape: Tuple[int, int],
) -> Tuple[int, int]:
    validate_host_image(original)
    validate_host_image(processed)

    if original.shape != processed.shape:
        raise ValueError(
            "Original and processed images must have "
            "identical shapes."
        )

    if len(watermark_shape) != 2:
        raise ValueError(
            "watermark_shape must be (height, width)."
        )

    height = int(watermark_shape[0])
    width = int(watermark_shape[1])

    if height <= 0 or width <= 0:
        raise ValueError(
            "Watermark dimensions must be positive."
        )

    return height, width


def _embedding_metadata(
    *,
    alpha: float,
    wavelet: str,
    level: int,
    subband: str,
    watermark_shape: Tuple[int, int],
    host_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    return {
        "method": "DWT",
        "family": "wavelet_transform",
        "alpha": float(alpha),
        "wavelet": wavelet,
        "level": int(level),
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
        "host_shape": list(host_shape),
        "embedding_rule": (
            "selected_band += alpha for bit 1; "
            "selected_band -= alpha for bit 0"
        ),
    }


@watermarker
class DWTWatermarker(BaseWatermarker):
    """Non-blind level-1 DWT watermarker."""

    METHOD = "DWT"

    DESCRIPTION = (
        "Non-blind level-1 wavelet-domain watermarking with "
        "configurable wavelet, subband and embedding strength."
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
    ) -> None:
        super().__init__(
            alpha=normalize_alpha(alpha)
        )

        self.wavelet = _validate_wavelet(
            wavelet
        )

        self.level = _validate_level(
            level
        )

        self.subband = _validate_subband(
            subband
        )

    def embed(
        self,
        host: np.ndarray,
        watermark: np.ndarray,
    ) -> EmbeddingResult:
        validate_host_image(host)
        validate_watermark(watermark)

        host_gray = _ensure_uint8(
            ensure_grayscale(
                np.asarray(host)
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

            wm_height, wm_width = (
                watermark_binary.shape
            )

            signs = (
                watermark_binary
                .astype(np.float64)
                * 2.0
                - 1.0
            )

            selected[
                :wm_height,
                :wm_width,
            ] += self.alpha * signs

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

            reconstructed = reconstructed[
                :host_gray.shape[0],
                :host_gray.shape[1],
            ]

            watermarked = _ensure_uint8(
                reconstructed
            )

        metadata = _embedding_metadata(
            alpha=self.alpha,
            wavelet=self.wavelet,
            level=self.level,
            subband=self.subband,
            watermark_shape=(
                watermark_binary.shape
            ),
            host_shape=host_gray.shape,
        )

        metadata["selected_band_shape"] = list(
            selected.shape
        )

        return EmbeddingResult(
            watermarked_image=watermarked,
            runtime=timer.elapsed,
            metadata=metadata,
        )

    def extract(
        self,
        original: np.ndarray,
        watermarked: np.ndarray,
        watermark_shape: Tuple[int, int],
    ) -> ExtractionResult:
        wm_height, wm_width = (
            _validate_extraction(
                original,
                watermarked,
                watermark_shape,
            )
        )

        original_gray = _ensure_uint8(
            ensure_grayscale(
                np.asarray(original)
            )
        )

        processed_gray = _ensure_uint8(
            ensure_grayscale(
                np.asarray(watermarked)
            )
        )

        with Timer() as timer:
            original_coefficients = _decompose(
                original_gray,
                self.wavelet,
            )

            processed_coefficients = _decompose(
                processed_gray,
                self.wavelet,
            )

            original_band = _select_subband(
                original_coefficients,
                self.subband,
            )

            processed_band = _select_subband(
                processed_coefficients,
                self.subband,
            )

            if (
                wm_height > original_band.shape[0]
                or wm_width > original_band.shape[1]
            ):
                raise ValueError(
                    "Requested watermark shape exceeds "
                    "the selected DWT band."
                )

            difference = (
                processed_band[
                    :wm_height,
                    :wm_width,
                ]
                - original_band[
                    :wm_height,
                    :wm_width,
                ]
            )

            extracted = (
                difference >= 0.0
            ).astype(np.uint8)

        metadata = {
            "method": self.METHOD,
            "alpha": float(self.alpha),
            "wavelet": self.wavelet,
            "level": int(self.level),
            "subband": self.subband,
            "blind": False,
            "watermark_height": wm_height,
            "watermark_width": wm_width,
            "selected_band_shape": list(
                original_band.shape
            ),
        }

        return ExtractionResult(
            extracted_watermark=extracted,
            runtime=timer.elapsed,
            metadata=metadata,
        )


if __name__ == "__main__":
    rng = np.random.default_rng(
        seed=42
    )

    host = rng.integers(
        0,
        256,
        size=(512, 512),
        dtype=np.uint8,
    )

    watermark = rng.integers(
        0,
        2,
        size=(32, 32),
        dtype=np.uint8,
    )

    algorithm = DWTWatermarker(
        alpha=10,
        wavelet="haar",
        subband="LL",
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

    assert embedded.watermarked_image.shape == host.shape
    assert extracted.extracted_watermark.shape == watermark.shape
    assert ber == 0.0

    print("✅ DWT watermarker self test passed.")
