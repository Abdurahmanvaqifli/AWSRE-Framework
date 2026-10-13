"""
AWSRE DCT-SVD Watermarking Module

Block-based invisible binary watermarking that applies a 2-D DCT to
non-overlapping host-image blocks and embeds each watermark bit by
modulating selected singular values of the DCT coefficient matrix.
Extraction is non-blind and compares the original and processed blocks.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np

from watermarking.adaptive_strength import compute_block_strength_map
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


DEFAULT_SINGULAR_INDICES: Tuple[int, ...] = (0, 1)


def validate_singular_indices(
    singular_indices: Iterable[int],
    block_size: int,
) -> Tuple[int, ...]:
    """Validate and normalize singular-value indices."""
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


def decompose_dct_block(
    block: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return U, S and Vt for the DCT transform of one image block."""
    dct_block = cv2.dct(block.astype(np.float32))
    return np.linalg.svd(dct_block, full_matrices=False)


def reconstruct_spatial_block(
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
) -> np.ndarray:
    """Reconstruct a spatial-domain block from a modified DCT SVD."""
    reconstructed_dct = (u * singular_values) @ vt
    return cv2.idct(reconstructed_dct.astype(np.float32))


@watermarker
class DCTSVDWatermarker(BaseWatermarker):
    """Block-based non-blind DCT-SVD binary watermarker."""

    METHOD = "DCT-SVD"

    DESCRIPTION = (
        "Block-based invisible watermarking using DCT-domain "
        "singular-value modulation."
    )

    SUPPORTS_RGB = False
    SUPPORTS_GRAYSCALE = True

    DEFAULT_ALPHA = 20

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        block_size: int = DEFAULT_BLOCK_SIZE,
        singular_indices: Iterable[int] = DEFAULT_SINGULAR_INDICES,
    ) -> None:
        super().__init__(alpha=normalize_alpha(alpha))

        self.block_size = int(block_size)

        if self.block_size <= 1:
            raise ValueError("block_size must be greater than 1.")

        self.singular_indices = validate_singular_indices(
            singular_indices,
            self.block_size,
        )

    def embed(
        self,
        host: np.ndarray,
        watermark: np.ndarray,
    ) -> EmbeddingResult:
        """Embed a binary watermark and return image, runtime and metadata."""
        self.validate(host, watermark)

        host_gray = ensure_grayscale(ensure_uint8_image(host))
        watermark_binary = normalize_binary_watermark(watermark)

        host_float = host_gray.astype(np.float32)
        watermarked_float = host_float.copy()

        positive_bits = 0
        negative_bits = 0

        with Timer() as timer:
            wm_height, wm_width = watermark_binary.shape

            for bit_row in range(wm_height):
                for bit_column in range(wm_width):
                    block = extract_block(
                        watermarked_float,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    u, singular_values, vt = decompose_dct_block(block)
                    modified = singular_values.astype(np.float32).copy()

                    bit = int(watermark_binary[bit_row, bit_column])
                    direction = 1.0 if bit == 1 else -1.0

                    for index in self.singular_indices:
                        modified[index] = max(
                            float(modified[index] + direction * self.alpha),
                            np.finfo(np.float32).eps,
                        )

                    reconstructed = reconstruct_spatial_block(
                        u,
                        modified,
                        vt,
                    )

                    place_block(
                        watermarked_float,
                        reconstructed,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    if bit == 1:
                        positive_bits += 1
                    else:
                        negative_bits += 1

        watermarked_uint8 = np.uint8(
            np.clip(np.rint(watermarked_float), 0, 255)
        )

        metadata: Dict[str, Any] = build_metadata(
            method=self.METHOD,
            alpha=self.alpha,
            host=host_gray,
            watermark=watermark_binary,
        )

        metadata.update({
            "block_size": self.block_size,
            "singular_indices": list(self.singular_indices),
            "embedded_bits": int(watermark_binary.size),
            "positive_bits": positive_bits,
            "negative_bits": negative_bits,
            "watermark_shape": list(watermark_binary.shape),
            "host_shape": list(host_gray.shape),
            "required_host_shape": list(
                required_host_dimensions(
                    watermark_binary.shape,
                    self.block_size,
                )
            ),
            "embedding_rate": float(watermark_binary.size / host_gray.size),
            "transform_pipeline": ["DCT", "SVD"],
            "non_blind_extraction": True,
        })

        return EmbeddingResult(
            watermarked_image=watermarked_uint8,
            runtime=timer.elapsed,
            metadata=metadata,
        )

    def extract(
        self,
        original: np.ndarray,
        watermarked: np.ndarray,
        watermark_shape: Tuple[int, int],
    ) -> ExtractionResult:
        """Extract a binary watermark by singular-value difference voting."""
        original_gray = ensure_grayscale(ensure_uint8_image(original))
        watermarked_gray = ensure_grayscale(ensure_uint8_image(watermarked))

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

        original_float = original_gray.astype(np.float32)
        watermarked_float = watermarked_gray.astype(np.float32)
        extracted = np.zeros(normalized_shape, dtype=np.uint8)
        mean_differences = []

        with Timer() as timer:
            wm_height, wm_width = normalized_shape

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

                    _, original_s, _ = decompose_dct_block(original_block)
                    _, processed_s, _ = decompose_dct_block(processed_block)

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
            "block_size": self.block_size,
            "singular_indices": list(self.singular_indices),
            "watermark_shape": list(normalized_shape),
            "extracted_bits": int(extracted.size),
            "mean_singular_difference": float(np.mean(differences_array)),
            "mean_absolute_singular_difference": float(
                np.mean(np.abs(differences_array))
            ),
            "minimum_singular_difference": float(np.min(differences_array)),
            "maximum_singular_difference": float(np.max(differences_array)),
            "transform_pipeline": ["DCT", "SVD"],
            "non_blind_extraction": True,
        }

        return ExtractionResult(
            extracted_watermark=extracted,
            runtime=timer.elapsed,
            metadata=metadata,
        )

    def maximum_watermark_shape(
        self,
        host: np.ndarray,
    ) -> Tuple[int, int]:
        """Return maximum binary-watermark shape supported by the host."""
        host_gray = ensure_grayscale(ensure_uint8_image(host))
        return (
            host_gray.shape[0] // self.block_size,
            host_gray.shape[1] // self.block_size,
        )

    def validate(
        self,
        host: np.ndarray,
        watermark: np.ndarray,
    ) -> None:
        """Perform generic and DCT-SVD-specific validation."""
        super().validate(host, watermark)
        host_gray = ensure_grayscale(np.asarray(host))
        watermark_binary = normalize_binary_watermark(watermark)
        validate_embedding_capacity(
            host_gray,
            watermark_binary,
            self.block_size,
        )

    def info(self) -> Dict[str, Any]:
        """Return algorithm configuration metadata."""
        information = super().info()
        information.update({
            "block_size": self.block_size,
            "singular_indices": list(self.singular_indices),
            "extraction_type": "Non-blind",
            "watermark_representation": "Binary",
            "transform_pipeline": ["DCT", "SVD"],
        })
        return information


@watermarker
class DCTSVDAdaptiveWatermarker(DCTSVDWatermarker):
    """
    Content-adaptive variant of DCTSVDWatermarker -- the same core
    contribution as DCTAdaptiveWatermarker (see watermarking/dct.py and
    watermarking/adaptive_strength.py), applied to singular-value
    modulation instead of raw DCT coefficients: each block's singular-
    value perturbation strength is scaled by that block's own local
    texture/edge energy (HVS masking) instead of using one fixed alpha
    everywhere.

    extract() is inherited from DCTSVDWatermarker completely unchanged --
    it decodes a bit from the SIGN of the singular-value difference, never
    its magnitude, so a per-block alpha decodes exactly as reliably as the
    classic fixed-alpha scheme. Only embed() is overridden here.

    Registered under a NEW method name ("DCT-SVD-Adaptive"), so the
    classic DCT-SVD method and every scientific measurement already
    collected for it are completely unaffected by this class's existence.
    """

    METHOD = "DCT-SVD-Adaptive"

    DESCRIPTION = (
        "Content-adaptive DCT-SVD watermarking: per-block singular-value "
        "modulation strength scaled by local texture/edge energy (HVS "
        "masking), instead of one fixed strength for the whole image."
    )

    def embed(
        self,
        host: np.ndarray,
        watermark: np.ndarray,
    ) -> EmbeddingResult:
        """Same embedding rule as DCTSVDWatermarker.embed(), except the
        additive perturbation for each block's singular values is
        `direction * self.alpha * strength_map[bit_row, bit_column]`
        instead of a flat `direction * self.alpha`."""
        self.validate(host, watermark)

        host_gray = ensure_grayscale(ensure_uint8_image(host))
        watermark_binary = normalize_binary_watermark(watermark)
        validate_embedding_capacity(host_gray, watermark_binary, self.block_size)

        host_float = host_gray.astype(np.float32)
        watermarked_float = host_float.copy()

        wm_height, wm_width = watermark_binary.shape

        strength_map = compute_block_strength_map(
            host_gray, self.block_size, wm_height, wm_width
        )

        positive_bits = 0
        negative_bits = 0

        with Timer() as timer:
            for bit_row in range(wm_height):
                for bit_column in range(wm_width):
                    block = extract_block(
                        watermarked_float, bit_row, bit_column, self.block_size
                    )

                    u, singular_values, vt = decompose_dct_block(block)
                    modified = singular_values.astype(np.float32).copy()

                    bit = int(watermark_binary[bit_row, bit_column])
                    direction = 1.0 if bit == 1 else -1.0
                    block_alpha = self.alpha * float(
                        strength_map[bit_row, bit_column]
                    )

                    for index in self.singular_indices:
                        modified[index] = max(
                            float(modified[index] + direction * block_alpha),
                            np.finfo(np.float32).eps,
                        )

                    reconstructed = reconstruct_spatial_block(u, modified, vt)

                    place_block(
                        watermarked_float,
                        reconstructed,
                        bit_row,
                        bit_column,
                        self.block_size,
                    )

                    if bit == 1:
                        positive_bits += 1
                    else:
                        negative_bits += 1

        watermarked_uint8 = np.uint8(
            np.clip(np.rint(watermarked_float), 0, 255)
        )

        metadata: Dict[str, Any] = build_metadata(
            method=self.METHOD,
            alpha=self.alpha,
            host=host_gray,
            watermark=watermark_binary,
        )

        metadata.update({
            "block_size": self.block_size,
            "singular_indices": list(self.singular_indices),
            "embedded_bits": int(watermark_binary.size),
            "positive_bits": positive_bits,
            "negative_bits": negative_bits,
            "watermark_shape": list(watermark_binary.shape),
            "host_shape": list(host_gray.shape),
            "required_host_shape": list(
                required_host_dimensions(watermark_binary.shape, self.block_size)
            ),
            "embedding_rate": float(watermark_binary.size / host_gray.size),
            "transform_pipeline": ["DCT", "SVD"],
            "non_blind_extraction": True,
            "adaptive_strength": True,
            "strength_multiplier_min": float(strength_map.min()),
            "strength_multiplier_max": float(strength_map.max()),
            "strength_multiplier_mean": float(strength_map.mean()),
        })

        return EmbeddingResult(
            watermarked_image=watermarked_uint8,
            runtime=timer.elapsed,
            metadata=metadata,
        )


def embed_dct_svd(
    host: np.ndarray,
    watermark: np.ndarray,
    alpha: float = DCTSVDWatermarker.DEFAULT_ALPHA,
) -> EmbeddingResult:
    """Functional wrapper around DCTSVDWatermarker.embed()."""
    return DCTSVDWatermarker(alpha=alpha).embed(host, watermark)


def extract_dct_svd(
    original: np.ndarray,
    watermarked: np.ndarray,
    watermark_shape: Tuple[int, int],
    alpha: float = DCTSVDWatermarker.DEFAULT_ALPHA,
) -> ExtractionResult:
    """Functional wrapper around DCTSVDWatermarker.extract()."""
    return DCTSVDWatermarker(alpha=alpha).extract(
        original,
        watermarked,
        watermark_shape,
    )
