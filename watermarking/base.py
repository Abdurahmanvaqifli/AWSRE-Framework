"""
AWSRE Watermarking Base Module

Every watermarking algorithm in AWSRE inherits from this class.

Supported algorithms:

- DCT
- DWT
- DCT-SVD
- DWT-SVD
- Block-SVD

Future:

- DFT
- DCT-DWT
- Autoencoder
- CNN
- Transformer

No Streamlit code exists here.
"""

from __future__ import annotations

import time
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


# ==========================================================
# IMAGE UTILITIES
# ==========================================================

def load_image(
    image_path: str | Path,
    grayscale: bool = True,
) -> np.ndarray:
    """
    Load image from disk.
    """

    image_path = str(image_path)

    flag = (
        cv2.IMREAD_GRAYSCALE
        if grayscale
        else cv2.IMREAD_COLOR
    )

    image = cv2.imread(image_path, flag)

    if image is None:
        raise FileNotFoundError(
            f"Cannot load image: {image_path}"
        )

    return image


def save_image(
    image: np.ndarray,
    output_path: str | Path,
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        image,
    )


def resize_watermark(
    watermark: np.ndarray,
    size: Tuple[int, int],
):

    return cv2.resize(
        watermark,
        size,
        interpolation=cv2.INTER_AREA,
    )


def ensure_grayscale(
    image: np.ndarray,
):

    if image.ndim == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def split_rgb(
    image: np.ndarray,
):

    if image.ndim != 3:

        raise ValueError(
            "RGB image required."
        )

    b, g, r = cv2.split(image)

    return r, g, b


def merge_rgb(
    r,
    g,
    b,
):

    return cv2.merge(
        [b, g, r]
    )


# ==========================================================
# VALIDATION
# ==========================================================

def validate_host_image(
    image: np.ndarray,
):

    if image is None:

        raise ValueError(
            "Host image is None."
        )

    if image.size == 0:

        raise ValueError(
            "Empty host image."
        )


def validate_watermark(
    watermark: np.ndarray,
):

    if watermark is None:

        raise ValueError(
            "Watermark is None."
        )

    if watermark.size == 0:

        raise ValueError(
            "Empty watermark."
        )


def normalize_alpha(
    alpha,
):

    alpha = float(alpha)

    if alpha <= 0:

        raise ValueError(
            "Alpha must be positive."
        )

    return alpha


# ==========================================================
# CAPACITY
# ==========================================================

def estimate_capacity(
    host: np.ndarray,
    watermark: np.ndarray,
):

    host_pixels = host.shape[0] * host.shape[1]

    wm_pixels = (
        watermark.shape[0]
        * watermark.shape[1]
    )

    return {

        "host_pixels": host_pixels,

        "watermark_pixels": wm_pixels,

        "capacity_ratio":
            wm_pixels / host_pixels,

    }


# ==========================================================
# HASH
# ==========================================================

def image_hash(
    image: np.ndarray,
):

    return hashlib.sha256(
        image.tobytes()
    ).hexdigest()


# ==========================================================
# TIMER
# ==========================================================

class Timer:

    def __enter__(self):

        self.start = time.perf_counter()

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):

        self.end = time.perf_counter()

        self.elapsed = (
            self.end
            - self.start
        )


# ==========================================================
# BASE WATERMARKER
# ==========================================================

class BaseWatermarker(ABC):

    """
    Parent class for all AWSRE
    watermarking algorithms.
    """

    METHOD = "BASE"

    DESCRIPTION = ""

    SUPPORTS_RGB = False

    SUPPORTS_GRAYSCALE = True

    DEFAULT_ALPHA = 10

    def __init__(
        self,
        alpha: float = 10,
    ):

        self.alpha = normalize_alpha(alpha)

    @abstractmethod
    def embed(
        self,
        host: np.ndarray,
        watermark: np.ndarray,
    ):
        """
        Embed watermark.
        """

        pass

    @abstractmethod
    def extract(
        self,
        original: np.ndarray,
        watermarked: np.ndarray,
        watermark_shape,
    ):
        """
        Extract watermark.
        """

        pass

    def validate(
        self,
        host,
        watermark,
    ):

        validate_host_image(host)

        validate_watermark(watermark)

    def info(self):

        return {

            "method":
                self.METHOD,

            "description":
                self.DESCRIPTION,

            "supports_rgb":
                self.SUPPORTS_RGB,

            "supports_grayscale":
                self.SUPPORTS_GRAYSCALE,

            "default_alpha":
                self.DEFAULT_ALPHA,

        }
     # ==========================================================
# EMBEDDING RESULT
# ==========================================================

class EmbeddingResult:
    """
    Returned by every embed() function.
    """

    def __init__(
        self,
        watermarked_image: np.ndarray,
        runtime: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):

        self.watermarked_image = watermarked_image

        self.runtime = runtime

        self.metadata = metadata or {}

    def to_dict(self):

        return {

            "runtime": self.runtime,

            "metadata": self.metadata,

        }


# ==========================================================
# EXTRACTION RESULT
# ==========================================================

class ExtractionResult:

    """
    Returned by every extract() function.
    """

    def __init__(

        self,

        extracted_watermark: np.ndarray,

        runtime: float,

        metadata: Optional[Dict[str, Any]] = None,

    ):

        self.extracted_watermark = extracted_watermark

        self.runtime = runtime

        self.metadata = metadata or {}

    def to_dict(self):

        return {

            "runtime": self.runtime,

            "metadata": self.metadata,

        }


# ==========================================================
# RGB HELPERS
# ==========================================================

def embed_single_channel(
    channel: np.ndarray,
    callback,
):
    """
    Apply embedding to a single RGB channel.
    """

    return callback(channel)


def embed_rgb_channels(
    image: np.ndarray,
    callback,
):

    r, g, b = split_rgb(image)

    r = callback(r)

    g = callback(g)

    b = callback(b)

    return merge_rgb(
        r,
        g,
        b,
    )


# ==========================================================
# COMMON PREPROCESSING
# ==========================================================

def prepare_inputs(
    host: np.ndarray,
    watermark: np.ndarray,
):

    validate_host_image(host)

    validate_watermark(watermark)

    host = ensure_grayscale(host)

    watermark = ensure_grayscale(watermark)

    return host, watermark


# ==========================================================
# BENCHMARK METADATA
# ==========================================================

def build_metadata(

    method,

    alpha,

    host,

    watermark,

):

    return {

        "method": method,

        "alpha": alpha,

        "host_hash": image_hash(host),

        "watermark_hash": image_hash(
            watermark
        ),

        "capacity": estimate_capacity(
            host,
            watermark,
        ),

    }


# ==========================================================
# SAVE DEBUG IMAGES
# ==========================================================

def save_debug_images(

    folder,

    host,

    watermark,

    watermarked=None,

    extracted=None,

):

    folder = Path(folder)

    folder.mkdir(

        parents=True,

        exist_ok=True,

    )

    save_image(

        host,

        folder / "host.png",

    )

    save_image(

        watermark,

        folder / "watermark.png",

    )

    if watermarked is not None:

        save_image(

            watermarked,

            folder / "watermarked.png",

        )

    if extracted is not None:

        save_image(

            extracted,

            folder / "extracted.png",

        )


# ==========================================================
# LOGGER PLACEHOLDER
# ==========================================================

def benchmark_hook(
    metadata,
):

    """
    Placeholder.

    AWSRE Benchmark
    will override this.
    """

    return metadata


# ==========================================================
# VERSION
# ==========================================================

FRAMEWORK_NAME = "AWSRE Watermarking Framework"

FRAMEWORK_VERSION = "1.0"


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(FRAMEWORK_NAME)

    print("=" * 60)

    base = BaseWatermarker

    print()

    print("Framework Version:")

    print(FRAMEWORK_VERSION)

    print()

    print("Base class loaded successfully.")

    print()

    print("Ready for:")

    print("- DCT")

    print("- DWT")

    print("- DCT-SVD")

    print("- DWT-SVD")

    print("- Block-SVD")
