"""
AWSRE Block-SVD Watermarking Module

Block-based invisible binary watermarking that applies SVD directly to
non-overlapping host-image pixel blocks (no transform domain step) and
embeds each watermark bit by modulating the dominant singular value.
Extraction is non-blind and compares the original and processed blocks.

This is the "plain" spatial-domain sibling of DCT-SVD (watermarking.dct_svd,
DCT-domain SVD) and DWT-SVD (watermarking.dwt_svd, wavelet-domain SVD):
together the three algorithms let AWSRE compare how the same singular-value
modulation strategy behaves in three different domains.

Rewritten to match the shared BaseWatermarker contract used by every other
AWSRE algorithm (embed()/extract() returning EmbeddingResult/ExtractionResult,
registration via the @watermarker class decorator). The previous version of
this file called `register_watermarker("block_svd")` as if it were a
decorator factory; the real registry.register_watermarker() takes a class as
its first argument, so that call raised TypeError at import time -- which is
why this module was commented out of BUILTIN_WATERMARKER_MODULES.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

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


DEFAULT_SINGULAR_INDICES: Tuple[int, ...] = (0,)


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


def decompose_block(
    block: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return U, S and Vt for the raw pixel-domain SVD of one image block."""
    return np.linalg.svd(block.astype(np.float32), full_matrices=False)


def reconstruct_spatial_block(
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
) -> np.ndarray:
    """Reconstruct a spatial-domain block from a modified SVD."""
    return (u * singular_values) @ vt


@watermarker
class BlockSVDWatermarker(BaseWatermarker):
    """Block-based non-blind spatial-domain SVD binary watermarker."""

    METHOD = "BLOCK-SVD"

    DESCRIPTION = (
        "Block-based invisible watermarking using pixel-domain "
        "singular-value modulation (no DCT/DWT transform step)."
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

                    u, singular_values, vt = decompose_block(block)
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
            "transform_pipeline": ["SVD"],
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

                    _, original_s, _ = decompose_block(original_block)
                    _, processed_s, _ = decompose_block(processed_block)

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
            "transform_pipeline": ["SVD"],
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
        """Perform generic and Block-SVD-specific validation."""
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
            "transform_pipeline": ["SVD"],
        })
        return information


def embed_block_svd(
    host: np.ndarray,
    watermark: np.ndarray,
    alpha: float = BlockSVDWatermarker.DEFAULT_ALPHA,
) -> EmbeddingResult:
    """Functional wrapper around BlockSVDWatermarker.embed()."""
    return BlockSVDWatermarker(alpha=alpha).embed(host, watermark)


def extract_block_svd(
    original: np.ndarray,
    watermarked: np.ndarray,
    watermark_shape: Tuple[int, int],
    alpha: float = BlockSVDWatermarker.DEFAULT_ALPHA,
) -> ExtractionResult:
    """Functional wrapper around BlockSVDWatermarker.extract()."""
    return BlockSVDWatermarker(alpha=alpha).extract(
        original,
        watermarked,
        watermark_shape,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    host_image = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
    watermark_bits = rng.integers(0, 2, size=(16, 16), dtype=np.uint8)

    algorithm = BlockSVDWatermarker(alpha=20.0)
    embed_result = algorithm.embed(host_image, watermark_bits)

    assert embed_result.watermarked_image.shape == host_image.shape

    extract_result = algorithm.extract(
        host_image,
        embed_result.watermarked_image,
        watermark_bits.shape,
    )

    ber = float(
        np.mean(extract_result.extracted_watermark != watermark_bits)
    )

    print(f"Block-SVD self-test BER: {ber}")
    assert ber < 0.05, "Block-SVD self-test BER too high"
    print("Block-SVD watermarker self test passed.")
