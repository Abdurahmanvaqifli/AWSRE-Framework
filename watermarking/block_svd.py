"""
Block-SVD Invisible Image Watermarking
======================================

A non-blind block-based SVD watermarking algorithm.

Embedding:
    1. Split the host image into fixed-size blocks.
    2. Apply SVD to every selected block.
    3. Modify the largest singular value according to the watermark bit.
    4. Reconstruct and merge the blocks.

Extraction:
    1. Compare singular values of original and watermarked blocks.
    2. Recover watermark bits from the direction of the modification.

The implementation supports:
    - Grayscale images
    - RGB/BGR images
    - Binary 1-D and 2-D watermarks
    - Configurable block size and embedding strength
    - Metadata-based extraction
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np


ArrayLike = Union[np.ndarray, list, tuple]


# ---------------------------------------------------------------------
# Optional framework integration
# ---------------------------------------------------------------------

try:
    from watermarking.base import BaseWatermarker
except (ImportError, ModuleNotFoundError):

    class BaseWatermarker:
        """Fallback base class when the framework base class is unavailable."""

        pass


def _identity_register(*args: Any, **kwargs: Any):
    """Fallback registry decorator."""

    def decorator(cls):
        return cls

    if args and isinstance(args[0], type):
        return args[0]

    return decorator


try:
    from watermarking.registry import register_watermarker
except (ImportError, ModuleNotFoundError):
    try:
        from watermarking.registry import register
        register_watermarker = register
    except (ImportError, ModuleNotFoundError):
        register_watermarker = _identity_register


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

@dataclass
class BlockSVDMetadata:
    """Metadata required for reproducible watermark extraction."""

    algorithm: str
    watermark_shape: Tuple[int, ...]
    watermark_length: int
    host_shape: Tuple[int, ...]
    block_size: int
    alpha: float
    channel: str
    usable_blocks: int
    embedded_bits: int
    repeated: bool
    embedding_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        result = asdict(self)
        result["watermark_shape"] = list(self.watermark_shape)
        result["host_shape"] = list(self.host_shape)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlockSVDMetadata":
        """Create metadata from a dictionary."""
        return cls(
            algorithm=str(data["algorithm"]),
            watermark_shape=tuple(data["watermark_shape"]),
            watermark_length=int(data["watermark_length"]),
            host_shape=tuple(data["host_shape"]),
            block_size=int(data["block_size"]),
            alpha=float(data["alpha"]),
            channel=str(data["channel"]),
            usable_blocks=int(data["usable_blocks"]),
            embedded_bits=int(data["embedded_bits"]),
            repeated=bool(data["repeated"]),
            embedding_time_seconds=float(
                data.get("embedding_time_seconds", 0.0)
            ),
        )


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_image(image: np.ndarray, name: str = "image") -> np.ndarray:
    """Validate and normalize an input image."""

    if image is None:
        raise ValueError(f"{name} cannot be None.")

    array = np.asarray(image)

    if array.ndim not in (2, 3):
        raise ValueError(
            f"{name} must be a 2-D grayscale or 3-D color image. "
            f"Received shape: {array.shape}"
        )

    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"{name} must contain 1, 3, or 4 channels. "
            f"Received shape: {array.shape}"
        )

    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"{name} is too small: {array.shape}")

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numerical values.")

    if not np.all(np.isfinite(array.astype(np.float64))):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array.copy()


def _validate_parameters(block_size: int, alpha: float) -> None:
    """Validate algorithm parameters."""

    if not isinstance(block_size, int):
        raise TypeError("block_size must be an integer.")

    if block_size < 2:
        raise ValueError("block_size must be at least 2.")

    if not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise TypeError("alpha must be numerical.")

    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")

    if alpha <= 0:
        raise ValueError("alpha must be greater than zero.")


def _prepare_watermark(
    watermark: ArrayLike,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Convert a watermark into a one-dimensional binary vector."""

    if watermark is None:
        raise ValueError("watermark cannot be None.")

    array = np.asarray(watermark)

    if array.size == 0:
        raise ValueError("watermark cannot be empty.")

    if array.ndim not in (1, 2):
        raise ValueError(
            "watermark must be a one-dimensional bit vector "
            "or a two-dimensional binary image."
        )

    original_shape = tuple(array.shape)

    if array.dtype == np.bool_:
        bits = array.astype(np.uint8).reshape(-1)
        return bits, original_shape

    numeric = array.astype(np.float64)

    if not np.all(np.isfinite(numeric)):
        raise ValueError("watermark contains NaN or infinite values.")

    unique_values = np.unique(numeric)

    if np.all(np.isin(unique_values, [0, 1])):
        bits = numeric.astype(np.uint8).reshape(-1)
    else:
        # Supports ordinary grayscale logo images.
        threshold = float(np.mean(numeric))
        bits = (numeric >= threshold).astype(np.uint8).reshape(-1)

    return bits, original_shape


# ---------------------------------------------------------------------
# Image-channel helpers
# ---------------------------------------------------------------------

def _extract_embedding_channel(
    image: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract the channel used for watermark embedding.

    Grayscale images:
        Embed directly into the grayscale plane.

    Color images:
        Embed into the Y channel of the YCrCb representation.
    """

    if image.ndim == 2:
        return image.astype(np.float64), {
            "mode": "grayscale",
            "original": image,
        }

    if image.shape[2] == 1:
        return image[:, :, 0].astype(np.float64), {
            "mode": "single_channel",
            "original": image,
        }

    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha_channel = image[:, :, 3].copy()

        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

        return ycrcb[:, :, 0].astype(np.float64), {
            "mode": "bgra",
            "ycrcb": ycrcb,
            "alpha_channel": alpha_channel,
        }

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    return ycrcb[:, :, 0].astype(np.float64), {
        "mode": "bgr",
        "ycrcb": ycrcb,
    }


def _restore_embedding_channel(
    modified_channel: np.ndarray,
    context: Dict[str, Any],
) -> np.ndarray:
    """Restore the modified embedding channel to the original image format."""

    channel_uint8 = np.clip(
        np.rint(modified_channel),
        0,
        255,
    ).astype(np.uint8)

    mode = context["mode"]

    if mode == "grayscale":
        return channel_uint8

    if mode == "single_channel":
        return channel_uint8[:, :, np.newaxis]

    ycrcb = context["ycrcb"].copy()
    ycrcb[:, :, 0] = channel_uint8

    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if mode == "bgra":
        return np.dstack((bgr, context["alpha_channel"]))

    return bgr


# ---------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------

def _block_positions(
    height: int,
    width: int,
    block_size: int,
):
    """Yield top-left positions of complete non-overlapping blocks."""

    usable_height = height - (height % block_size)
    usable_width = width - (width % block_size)

    for row in range(0, usable_height, block_size):
        for column in range(0, usable_width, block_size):
            yield row, column


def _count_blocks(
    height: int,
    width: int,
    block_size: int,
) -> int:
    """Return the number of complete blocks in an image."""

    return (height // block_size) * (width // block_size)


def _singular_values(block: np.ndarray) -> np.ndarray:
    """Calculate singular values of a block."""

    return np.linalg.svd(
        block.astype(np.float64),
        full_matrices=False,
        compute_uv=False,
    )


# ---------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------
@register_watermarker("block_svd")
class BlockSVDWatermarker(BaseWatermarker):
    """
    Non-blind Block-SVD watermarking algorithm.

    Parameters
    ----------
    block_size:
        Width and height of each non-overlapping image block.

    alpha:
        Singular-value modification strength.

    repeat_watermark:
        When True, repeat the watermark over all available blocks.
        Repetition allows majority-vote extraction.

    use_majority_vote:
        When True and repetition is enabled, combine repeated recovered
        copies using majority voting.
    """

    name = "block_svd"
    algorithm_name = "Block-SVD"
    requires_original = True
    is_blind = False

    def __init__(
        self,
        block_size: int = 8,
        alpha: float = 20.0,
        repeat_watermark: bool = True,
        use_majority_vote: bool = True,
    ) -> None:
        _validate_parameters(block_size, alpha)

        self.block_size = block_size
        self.alpha = float(alpha)
        self.repeat_watermark = bool(repeat_watermark)
        self.use_majority_vote = bool(use_majority_vote)

        self.last_metadata: Optional[BlockSVDMetadata] = None
        self.last_embedding_time: Optional[float] = None
        self.last_extraction_time: Optional[float] = None

    @property
    def identifier(self) -> str:
        """Registry-friendly algorithm identifier."""
        return self.name

    def capacity(self, image: np.ndarray) -> int:
        """Return embedding capacity in bits."""

        validated = _validate_image(image)
        height, width = validated.shape[:2]

        return _count_blocks(
            height=height,
            width=width,
            block_size=self.block_size,
        )

    def embed(
        self,
        image: np.ndarray,
        watermark: ArrayLike,
        *,
        alpha: Optional[float] = None,
        repeat_watermark: Optional[bool] = None,
        return_metadata: bool = True,
    ):
        """
        Embed a binary watermark into an image.

        Returns
        -------
        When return_metadata=True:
            (watermarked_image, metadata_dictionary)

        When return_metadata=False:
            watermarked_image
        """

        started_at = perf_counter()

        host = _validate_image(image, "image")
        bits, watermark_shape = _prepare_watermark(watermark)

        effective_alpha = self.alpha if alpha is None else float(alpha)
        effective_repeat = (
            self.repeat_watermark
            if repeat_watermark is None
            else bool(repeat_watermark)
        )

        _validate_parameters(self.block_size, effective_alpha)

        height, width = host.shape[:2]

        available_blocks = _count_blocks(
            height,
            width,
            self.block_size,
        )

        if available_blocks == 0:
            raise ValueError(
                "The image does not contain a complete embedding block. "
                f"Image shape={host.shape}, block_size={self.block_size}."
            )

        if bits.size > available_blocks:
            raise ValueError(
                "Watermark exceeds Block-SVD capacity. "
                f"Watermark bits={bits.size}, "
                f"available blocks={available_blocks}. "
                "Use a smaller watermark or a smaller block size."
            )

        channel, context = _extract_embedding_channel(host)
        modified_channel = channel.copy()

        embedded_bits = (
            available_blocks if effective_repeat else int(bits.size)
        )

        for index, (row, column) in enumerate(
            _block_positions(height, width, self.block_size)
        ):
            if index >= embedded_bits:
                break

            bit = int(bits[index % bits.size])

            block = modified_channel[
                row : row + self.block_size,
                column : column + self.block_size,
            ]

            u_matrix, singular_values, vh_matrix = np.linalg.svd(
                block,
                full_matrices=False,
            )

            # Bit 1 increases the dominant singular value.
            # Bit 0 decreases the dominant singular value.
            direction = 1.0 if bit == 1 else -1.0

            new_s0 = singular_values[0] + (
                direction * effective_alpha
            )

            # Singular values cannot be negative.
            singular_values[0] = max(new_s0, 0.0)

            reconstructed = (
                u_matrix
                @ np.diag(singular_values)
                @ vh_matrix
            )

            modified_channel[
                row : row + self.block_size,
                column : column + self.block_size,
            ] = reconstructed

        watermarked = _restore_embedding_channel(
            modified_channel,
            context,
        )

        elapsed = perf_counter() - started_at
        self.last_embedding_time = elapsed

        channel_name = "Y" if host.ndim == 3 else "grayscale"

        metadata = BlockSVDMetadata(
            algorithm=self.algorithm_name,
            watermark_shape=watermark_shape,
            watermark_length=int(bits.size),
            host_shape=tuple(host.shape),
            block_size=self.block_size,
            alpha=effective_alpha,
            channel=channel_name,
            usable_blocks=available_blocks,
            embedded_bits=embedded_bits,
            repeated=effective_repeat,
            embedding_time_seconds=elapsed,
        )

        self.last_metadata = metadata

        if return_metadata:
            return watermarked, metadata.to_dict()

        return watermarked

    def extract(
        self,
        original_image: np.ndarray,
        watermarked_image: np.ndarray,
        watermark_shape: Optional[Tuple[int, ...]] = None,
        *,
        metadata: Optional[
            Union[BlockSVDMetadata, Dict[str, Any]]
        ] = None,
        watermark_length: Optional[int] = None,
        block_size: Optional[int] = None,
        use_majority_vote: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Extract a watermark using the original and watermarked images.

        The algorithm is non-blind; therefore, original_image is required.
        """

        started_at = perf_counter()

        original = _validate_image(
            original_image,
            "original_image",
        )
        watermarked = _validate_image(
            watermarked_image,
            "watermarked_image",
        )

        if original.shape != watermarked.shape:
            raise ValueError(
                "Original and watermarked images must have the same shape. "
                f"Original={original.shape}, "
                f"watermarked={watermarked.shape}."
            )

        parsed_metadata: Optional[BlockSVDMetadata] = None

        if metadata is not None:
            if isinstance(metadata, BlockSVDMetadata):
                parsed_metadata = metadata
            elif isinstance(metadata, dict):
                parsed_metadata = BlockSVDMetadata.from_dict(metadata)
            else:
                raise TypeError(
                    "metadata must be a BlockSVDMetadata object "
                    "or a dictionary."
                )

        effective_block_size = (
            parsed_metadata.block_size
            if parsed_metadata is not None
            else (
                self.block_size
                if block_size is None
                else int(block_size)
            )
        )

        if watermark_shape is None:
            if parsed_metadata is not None:
                watermark_shape = parsed_metadata.watermark_shape
            elif self.last_metadata is not None:
                watermark_shape = self.last_metadata.watermark_shape

        if watermark_length is None:
            if parsed_metadata is not None:
                watermark_length = parsed_metadata.watermark_length
            elif watermark_shape is not None:
                watermark_length = int(np.prod(watermark_shape))
            elif self.last_metadata is not None:
                watermark_length = self.last_metadata.watermark_length

        if watermark_length is None:
            raise ValueError(
                "watermark_length, watermark_shape, or metadata "
                "must be provided."
            )

        watermark_length = int(watermark_length)

        if watermark_length <= 0:
            raise ValueError("watermark_length must be positive.")

        majority_vote = (
            self.use_majority_vote
            if use_majority_vote is None
            else bool(use_majority_vote)
        )

        original_channel, _ = _extract_embedding_channel(original)
        watermarked_channel, _ = _extract_embedding_channel(watermarked)

        height, width = original.shape[:2]

        available_blocks = _count_blocks(
            height,
            width,
            effective_block_size,
        )

        if watermark_length > available_blocks:
            raise ValueError(
                "Requested watermark length exceeds available blocks. "
                f"Requested={watermark_length}, "
                f"available={available_blocks}."
            )

        recovered_sequence = []

        for row, column in _block_positions(
            height,
            width,
            effective_block_size,
        ):
            original_block = original_channel[
                row : row + effective_block_size,
                column : column + effective_block_size,
            ]

            watermarked_block = watermarked_channel[
                row : row + effective_block_size,
                column : column + effective_block_size,
            ]

            original_s = _singular_values(original_block)
            watermarked_s = _singular_values(watermarked_block)

            difference = watermarked_s[0] - original_s[0]

            recovered_bit = 1 if difference >= 0 else 0
            recovered_sequence.append(recovered_bit)

        recovered_sequence_array = np.asarray(
            recovered_sequence,
            dtype=np.uint8,
        )

        repeated = (
            parsed_metadata.repeated
            if parsed_metadata is not None
            else recovered_sequence_array.size > watermark_length
        )

        if (
            repeated
            and majority_vote
            and recovered_sequence_array.size > watermark_length
        ):
            final_bits = np.zeros(
                watermark_length,
                dtype=np.uint8,
            )

            for bit_index in range(watermark_length):
                observations = recovered_sequence_array[
                    bit_index::watermark_length
                ]

                final_bits[bit_index] = (
                    1
                    if np.mean(observations) >= 0.5
                    else 0
                )
        else:
            final_bits = recovered_sequence_array[:watermark_length]

        if final_bits.size < watermark_length:
            raise RuntimeError(
                "Not enough blocks were decoded to reconstruct "
                "the complete watermark."
            )

        self.last_extraction_time = perf_counter() - started_at

        if watermark_shape is not None:
            expected_size = int(np.prod(watermark_shape))

            if expected_size != watermark_length:
                raise ValueError(
                    "watermark_shape does not match watermark_length. "
                    f"Shape size={expected_size}, "
                    f"length={watermark_length}."
                )

            return final_bits.reshape(watermark_shape)

        return final_bits

    def embed_watermark(
        self,
        image: np.ndarray,
        watermark: ArrayLike,
        **kwargs: Any,
    ):
        """Compatibility alias for embed()."""
        return self.embed(image, watermark, **kwargs)

    def extract_watermark(
        self,
        original_image: np.ndarray,
        watermarked_image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compatibility alias for extract()."""
        return self.extract(
            original_image,
            watermarked_image,
            **kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the current algorithm configuration."""
        return {
            "name": self.name,
            "algorithm": self.algorithm_name,
            "block_size": self.block_size,
            "alpha": self.alpha,
            "repeat_watermark": self.repeat_watermark,
            "use_majority_vote": self.use_majority_vote,
            "requires_original": self.requires_original,
            "is_blind": self.is_blind,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"block_size={self.block_size}, "
            f"alpha={self.alpha}, "
            f"repeat_watermark={self.repeat_watermark}, "
            f"use_majority_vote={self.use_majority_vote}"
            f")"
        )


# ---------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------

def embed_block_svd(
    image: np.ndarray,
    watermark: ArrayLike,
    *,
    block_size: int = 8,
    alpha: float = 20.0,
    repeat_watermark: bool = True,
    return_metadata: bool = True,
):
    """Functional wrapper for Block-SVD embedding."""

    watermarker = BlockSVDWatermarker(
        block_size=block_size,
        alpha=alpha,
        repeat_watermark=repeat_watermark,
    )

    return watermarker.embed(
        image,
        watermark,
        return_metadata=return_metadata,
    )


def extract_block_svd(
    original_image: np.ndarray,
    watermarked_image: np.ndarray,
    *,
    watermark_shape: Optional[Tuple[int, ...]] = None,
    metadata: Optional[
        Union[BlockSVDMetadata, Dict[str, Any]]
    ] = None,
    watermark_length: Optional[int] = None,
    block_size: int = 8,
    use_majority_vote: bool = True,
) -> np.ndarray:
    """Functional wrapper for Block-SVD extraction."""

    watermarker = BlockSVDWatermarker(
        block_size=block_size,
        use_majority_vote=use_majority_vote,
    )

    return watermarker.extract(
        original_image,
        watermarked_image,
        watermark_shape=watermark_shape,
        metadata=metadata,
        watermark_length=watermark_length,
    )


# Common compatibility aliases.
embed = embed_block_svd
extract = extract_block_svd


__all__ = [
    "BlockSVDMetadata",
    "BlockSVDWatermarker",
    "embed_block_svd",
    "extract_block_svd",
    "embed",
    "extract",
]
