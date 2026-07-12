"""
AWSRE Watermark Generator Base Module

Defines the common interface and result models used by every
watermark generator in AWSRE-Bench.

Supported and future generators:

- Binary Pattern
- Text
- QR Code
- Logo
- Signature
- Medical Identifier
- Provenance Identifier

This module contains no Streamlit code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import hashlib
import time
import uuid

import cv2
import numpy as np


# ============================================================
# ENUMS
# ============================================================

class WatermarkType(str, Enum):
    """
    Supported AWSRE watermark categories.
    """

    BINARY_PATTERN = "Binary Pattern"
    TEXT = "Text"
    QR_CODE = "QR Code"
    LOGO = "Logo"
    SIGNATURE = "Signature"


# ============================================================
# GENERAL HELPERS
# ============================================================

def generate_watermark_id() -> str:
    """
    Generate a unique watermark identifier.
    """
    short_uuid = uuid.uuid4().hex[:12].upper()

    return f"WM-{short_uuid}"


def calculate_array_hash(array: np.ndarray) -> str:
    """
    Calculate SHA-256 hash of a NumPy array.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(
            "Hash input must be a NumPy array."
        )

    if array.size == 0:
        raise ValueError(
            "Cannot hash an empty array."
        )

    contiguous = np.ascontiguousarray(array)

    sha256 = hashlib.sha256()
    sha256.update(contiguous.tobytes())

    return sha256.hexdigest()


def validate_watermark_size(
    size: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Validate and normalize watermark size.

    Parameters
    ----------
    size:
        Width and height as (width, height).
    """
    if (
        not isinstance(size, tuple)
        or len(size) != 2
    ):
        raise TypeError(
            "Watermark size must be a tuple: (width, height)."
        )

    width = int(size[0])
    height = int(size[1])

    if width <= 0 or height <= 0:
        raise ValueError(
            "Watermark width and height must be positive."
        )

    return width, height


def normalize_binary_watermark(
    watermark: np.ndarray,
) -> np.ndarray:
    """
    Convert watermark data into binary uint8 values 0 and 1.

    Supports:
    - binary 0/1 arrays;
    - grayscale 0/255 arrays;
    - floating-point arrays;
    - three-channel arrays.
    """
    if watermark is None:
        raise ValueError(
            "Watermark cannot be None."
        )

    if not isinstance(watermark, np.ndarray):
        raise TypeError(
            "Watermark must be a NumPy array."
        )

    if watermark.size == 0:
        raise ValueError(
            "Watermark cannot be empty."
        )

    if watermark.ndim == 3:
        if watermark.shape[2] != 3:
            raise ValueError(
                "Three-channel watermark must have exactly 3 channels."
            )

        watermark = cv2.cvtColor(
            watermark,
            cv2.COLOR_BGR2GRAY,
        )

    if watermark.ndim != 2:
        raise ValueError(
            "Watermark must be two-dimensional after preprocessing."
        )

    watermark_float = watermark.astype(
        np.float32
    )

    if not np.all(
        np.isfinite(watermark_float)
    ):
        raise ValueError(
            "Watermark contains NaN or infinite values."
        )

    if float(np.max(watermark_float)) <= 1.0:
        binary = watermark_float >= 0.5
    else:
        binary = watermark_float >= 127.5

    return binary.astype(
        np.uint8
    )


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class GeneratedWatermark:
    """
    Result returned by every watermark generator.
    """

    image: np.ndarray

    watermark_type: str
    generator_name: str

    width: int
    height: int

    watermark_id: str = field(
        default_factory=generate_watermark_id
    )

    seed: Optional[int] = None
    generation_time_seconds: float = 0.0

    file_hash: str = ""
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.image = normalize_binary_watermark(
            self.image
        )

        actual_height, actual_width = (
            self.image.shape
        )

        self.width = int(
            self.width
        )

        self.height = int(
            self.height
        )

        if (
            self.width != actual_width
            or self.height != actual_height
        ):
            raise ValueError(
                "Generated watermark dimensions do not match metadata. "
                f"Image: {actual_width}x{actual_height}; "
                f"metadata: {self.width}x{self.height}."
            )

        self.generation_time_seconds = max(
            0.0,
            float(self.generation_time_seconds),
        )

        if not self.file_hash:
            self.file_hash = calculate_array_hash(
                self.image
            )

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Return shape as (height, width).
        """
        return self.image.shape

    @property
    def size(self) -> Tuple[int, int]:
        """
        Return dimensions as (width, height).
        """
        return self.width, self.height

    @property
    def payload_bits(self) -> int:
        """
        Number of binary payload elements.
        """
        return int(
            self.image.size
        )

    @property
    def density(self) -> float:
        """
        Ratio of foreground bits equal to one.
        """
        return float(
            np.mean(self.image)
        )

    def metadata_dict(self) -> Dict[str, Any]:
        """
        Return serializable metadata without the NumPy image.
        """
        return {
            "watermark_id": self.watermark_id,
            "watermark_type": self.watermark_type,
            "generator_name": self.generator_name,
            "width": self.width,
            "height": self.height,
            "payload_bits": self.payload_bits,
            "density": self.density,
            "seed": self.seed,
            "generation_time_seconds": (
                self.generation_time_seconds
            ),
            "file_hash": self.file_hash,
            "metadata": self.metadata,
        }

    def to_uint8_image(self) -> np.ndarray:
        """
        Convert internal 0/1 binary representation to 0/255.
        """
        return (
            self.image * 255
        ).astype(np.uint8)


# ============================================================
# BASE GENERATOR
# ============================================================

class BaseWatermarkGenerator(ABC):
    """
    Abstract parent class for all AWSRE watermark generators.
    """

    GENERATOR_NAME = "Base Generator"
    WATERMARK_TYPE = WatermarkType.BINARY_PATTERN

    DESCRIPTION = ""

    def __init__(
        self,
        *,
        default_seed: Optional[int] = 42,
    ) -> None:
        self.default_seed = (
            None
            if default_seed is None
            else int(default_seed)
        )

    @abstractmethod
    def generate(
        self,
        size: Tuple[int, int],
        **kwargs: Any,
    ) -> GeneratedWatermark:
        """
        Generate one watermark.
        """

    def validate_size(
        self,
        size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Validate requested watermark dimensions.
        """
        return validate_watermark_size(
            size
        )

    def resolve_seed(
        self,
        seed: Optional[int],
    ) -> Optional[int]:
        """
        Use explicitly supplied seed or the generator default.
        """
        if seed is None:
            return self.default_seed

        return int(seed)

    def build_result(
        self,
        *,
        image: np.ndarray,
        size: Tuple[int, int],
        seed: Optional[int],
        generation_time_seconds: float,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> GeneratedWatermark:
        """
        Create a validated GeneratedWatermark object.
        """
        width, height = self.validate_size(
            size
        )

        return GeneratedWatermark(
            image=image,
            watermark_type=self.WATERMARK_TYPE.value,
            generator_name=self.GENERATOR_NAME,
            width=width,
            height=height,
            seed=seed,
            generation_time_seconds=(
                generation_time_seconds
            ),
            metadata=metadata or {},
        )

    def info(self) -> Dict[str, Any]:
        """
        Return generator metadata.
        """
        return {
            "generator_name": self.GENERATOR_NAME,
            "watermark_type": self.WATERMARK_TYPE.value,
            "description": self.DESCRIPTION,
            "default_seed": self.default_seed,
        }


# ============================================================
# TIMER
# ============================================================

class GenerationTimer:
    """
    Lightweight context manager for generation runtime.
    """

    def __enter__(self) -> "GenerationTimer":
        self.started_at = time.perf_counter()
        self.elapsed = 0.0

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.finished_at = time.perf_counter()
        self.elapsed = (
            self.finished_at
            - self.started_at
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    sample = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    sample[::2, ::2] = 1
    sample[1::2, 1::2] = 1

    result = GeneratedWatermark(
        image=sample,
        watermark_type=(
            WatermarkType.BINARY_PATTERN.value
        ),
        generator_name="Base Self Test",
        width=32,
        height=32,
        seed=42,
    )

    assert result.shape == (32, 32)
    assert result.payload_bits == 1024
    assert set(
        np.unique(result.image)
    ).issubset({0, 1})

    print("=" * 72)
    print("AWSRE WATERMARK GENERATOR BASE TEST")
    print("=" * 72)

    print(
        "Watermark ID:",
        result.watermark_id,
    )

    print(
        "Shape:",
        result.shape,
    )

    print(
        "Payload:",
        result.payload_bits,
    )

    print(
        "Density:",
        f"{result.density:.4f}",
    )

    print(
        "SHA-256:",
        result.file_hash,
    )

    print(
        "\n✅ Watermark generator base test passed."
    )
