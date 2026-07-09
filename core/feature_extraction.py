"""
AWSRE Feature Extraction Module

This module extracts measurable features from:
1. Host images
2. Watermarks

It contains no Streamlit code.
It can be reused in Streamlit, FastAPI, desktop apps, and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class HostFeatures:
    brightness: float
    contrast: float
    entropy: float
    edge_density: float
    texture_variance: float
    frequency_energy: float
    resolution: int
    width: int
    height: int
    aspect_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatermarkFeatures:
    width: int
    height: int
    payload_bits: int
    density: float
    entropy: float
    edge_complexity: float
    connected_components: int
    structural_complexity: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# IMAGE LOADING / PREPROCESSING
# ============================================================

def load_image_as_rgb(file, target_size: Tuple[int, int] | None = None) -> np.ndarray:
    """
    Load an image file as RGB numpy array.

    Parameters
    ----------
    file:
        Uploaded file path, file-like object, or PIL-compatible object.
    target_size:
        Optional resize size as (width, height).

    Returns
    -------
    np.ndarray
        RGB image array.
    """
    image = Image.open(file).convert("RGB")
    image_np = np.array(image)

    if target_size is not None:
        image_np = cv2.resize(image_np, target_size)

    return image_np


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    """
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize any numeric image array into uint8 range [0, 255].
    """
    image = np.asarray(image, dtype=np.float32)

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


# ============================================================
# HOST IMAGE FEATURE FUNCTIONS
# ============================================================

def compute_brightness(gray: np.ndarray) -> float:
    """
    Mean intensity of grayscale image.
    """
    return float(np.mean(gray))


def compute_contrast(gray: np.ndarray) -> float:
    """
    Standard deviation of grayscale intensity.
    """
    return float(np.std(gray))


def compute_entropy(gray: np.ndarray) -> float:
    """
    Shannon entropy of image intensity distribution.
    """
    return float(shannon_entropy(gray))


def compute_edge_density(gray: np.ndarray) -> float:
    """
    Ratio of edge pixels to all pixels using Canny edge detection.
    """
    edges = cv2.Canny(gray, 100, 200)
    return float(np.sum(edges > 0) / edges.size)


def compute_texture_variance(gray: np.ndarray) -> float:
    """
    Texture variance based on Laplacian variance.
    Higher value means stronger local texture/detail.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_frequency_energy(gray: np.ndarray) -> float:
    """
    Frequency-domain energy based on FFT magnitude spectrum.
    """
    gray_float = np.float32(gray)

    fft = np.fft.fft2(gray_float)
    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)

    return float(np.mean(log_magnitude))


def compute_resolution(gray: np.ndarray) -> int:
    """
    Total pixel count.
    """
    height, width = gray.shape
    return int(height * width)


def compute_aspect_ratio(gray: np.ndarray) -> float:
    """
    Width / height ratio.
    """
    height, width = gray.shape
    return float(width / height)


def extract_host_features(image_rgb: np.ndarray) -> HostFeatures:
    """
    Extract all host image features required by AWSRE.

    Parameters
    ----------
    image_rgb:
        RGB host image.

    Returns
    -------
    HostFeatures
        Structured host feature object.
    """
    gray = rgb_to_gray(image_rgb)
    height, width = gray.shape

    return HostFeatures(
        brightness=round(compute_brightness(gray), 4),
        contrast=round(compute_contrast(gray), 4),
        entropy=round(compute_entropy(gray), 4),
        edge_density=round(compute_edge_density(gray), 6),
        texture_variance=round(compute_texture_variance(gray), 4),
        frequency_energy=round(compute_frequency_energy(gray), 4),
        resolution=compute_resolution(gray),
        width=int(width),
        height=int(height),
        aspect_ratio=round(compute_aspect_ratio(gray), 4),
    )


# ============================================================
# WATERMARK CREATION / PREPROCESSING
# ============================================================

def load_watermark_as_binary(
    file,
    target_size: Tuple[int, int] = (64, 64),
    threshold: int = 127
) -> np.ndarray:
    """
    Load watermark image and convert to binary matrix.

    Returns
    -------
    np.ndarray
        Binary watermark with values 0 and 1.
    """
    watermark = Image.open(file).convert("L")
    watermark_np = np.array(watermark)

    watermark_np = cv2.resize(
        watermark_np,
        target_size,
        interpolation=cv2.INTER_NEAREST
    )

    _, binary = cv2.threshold(
        watermark_np,
        threshold,
        1,
        cv2.THRESH_BINARY
    )

    return binary.astype(np.uint8)


def create_text_watermark(
    text: str,
    target_size: Tuple[int, int] = (64, 64)
) -> np.ndarray:
    """
    Create binary text watermark.

    Parameters
    ----------
    text:
        Text to render as watermark.
    target_size:
        Output watermark size.

    Returns
    -------
    np.ndarray
        Binary watermark.
    """
    width, height = target_size
    watermark = np.zeros((height, width), dtype=np.uint8)

    text = text.strip() if text else "AWSRE"

    cv2.putText(
        watermark,
        text[:8],
        (4, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        255,
        1,
        cv2.LINE_AA
    )

    _, binary = cv2.threshold(
        watermark,
        127,
        1,
        cv2.THRESH_BINARY
    )

    return binary.astype(np.uint8)


def create_binary_pattern_watermark(
    target_size: Tuple[int, int] = (64, 64)
) -> np.ndarray:
    """
    Create a default binary checker-like watermark pattern.
    """
    width, height = target_size
    pattern = np.zeros((height, width), dtype=np.uint8)

    pattern[::2, ::2] = 1
    pattern[1::2, 1::2] = 1

    return pattern


def ensure_binary_watermark(watermark: np.ndarray) -> np.ndarray:
    """
    Ensure watermark has binary values 0 and 1.
    """
    watermark = np.asarray(watermark)

    if watermark.max() > 1:
        watermark = watermark / 255.0

    binary = np.where(watermark > 0.5, 1, 0)
    return binary.astype(np.uint8)


# ============================================================
# WATERMARK FEATURE FUNCTIONS
# ============================================================

def compute_payload_bits(binary_wm: np.ndarray) -> int:
    """
    Payload capacity in bits assuming one bit per watermark pixel.
    """
    return int(binary_wm.size)


def compute_watermark_density(binary_wm: np.ndarray) -> float:
    """
    Ratio of active watermark pixels.
    """
    return float(np.sum(binary_wm > 0) / binary_wm.size)


def compute_watermark_entropy(binary_wm: np.ndarray) -> float:
    """
    Entropy of binary watermark.
    """
    return float(shannon_entropy(binary_wm))


def compute_edge_complexity(binary_wm: np.ndarray) -> float:
    """
    Edge density of watermark.
    """
    wm_uint8 = (binary_wm * 255).astype(np.uint8)

    edges = cv2.Canny(wm_uint8, 50, 150)

    return float(np.sum(edges > 0) / edges.size)


def compute_connected_components(binary_wm: np.ndarray) -> int:
    """
    Count connected foreground components in watermark.
    """
    binary_wm = binary_wm.astype(np.uint8)

    num_labels, _ = cv2.connectedComponents(binary_wm)

    return int(max(0, num_labels - 1))


def compute_structural_complexity(binary_wm: np.ndarray) -> float:
    """
    Combined structural complexity measure.

    It combines:
    - watermark entropy
    - edge complexity
    - connected component complexity
    """
    entropy = compute_watermark_entropy(binary_wm)
    edge_complexity = compute_edge_complexity(binary_wm)
    components = compute_connected_components(binary_wm)

    normalized_components = min(components / 50.0, 1.0)

    complexity = (
        0.4 * entropy +
        0.3 * edge_complexity +
        0.3 * normalized_components
    )

    return float(complexity)


def extract_watermark_features(binary_wm: np.ndarray) -> WatermarkFeatures:
    """
    Extract all watermark features required by AWSRE.

    Parameters
    ----------
    binary_wm:
        Binary watermark matrix.

    Returns
    -------
    WatermarkFeatures
        Structured watermark feature object.
    """
    binary_wm = ensure_binary_watermark(binary_wm)

    height, width = binary_wm.shape

    return WatermarkFeatures(
        width=int(width),
        height=int(height),
        payload_bits=compute_payload_bits(binary_wm),
        density=round(compute_watermark_density(binary_wm), 6),
        entropy=round(compute_watermark_entropy(binary_wm), 4),
        edge_complexity=round(compute_edge_complexity(binary_wm), 6),
        connected_components=compute_connected_components(binary_wm),
        structural_complexity=round(compute_structural_complexity(binary_wm), 6),
    )


# ============================================================
# FEATURE INTERPRETATION HELPERS
# ============================================================

def classify_brightness(value: float) -> str:
    if value < 85:
        return "Low"
    if value < 170:
        return "Medium"
    return "High"


def classify_contrast(value: float) -> str:
    if value < 35:
        return "Low"
    if value < 70:
        return "Medium"
    return "High"


def classify_entropy(value: float) -> str:
    if value < 4:
        return "Low"
    if value < 6.5:
        return "Medium"
    return "High"


def classify_edge_density(value: float) -> str:
    if value < 0.03:
        return "Low"
    if value < 0.10:
        return "Medium"
    return "High"


def classify_texture_variance(value: float) -> str:
    if value < 100:
        return "Low"
    if value < 800:
        return "Medium"
    return "High"


def classify_watermark_density(value: float) -> str:
    if value < 0.20:
        return "Sparse"
    if value < 0.55:
        return "Balanced"
    return "Dense"


def classify_watermark_complexity(value: float) -> str:
    if value < 0.25:
        return "Low"
    if value < 0.55:
        return "Medium"
    return "High"


def interpret_host_features(features: HostFeatures) -> Dict[str, str]:
    """
    Convert numeric host features into qualitative labels.
    """
    return {
        "brightness_level": classify_brightness(features.brightness),
        "contrast_level": classify_contrast(features.contrast),
        "entropy_level": classify_entropy(features.entropy),
        "edge_density_level": classify_edge_density(features.edge_density),
        "texture_level": classify_texture_variance(features.texture_variance),
    }


def interpret_watermark_features(features: WatermarkFeatures) -> Dict[str, str]:
    """
    Convert numeric watermark features into qualitative labels.
    """
    return {
        "density_level": classify_watermark_density(features.density),
        "complexity_level": classify_watermark_complexity(
            features.structural_complexity
        ),
    }


# ============================================================
# FULL PIPELINE HELPERS
# ============================================================

def analyze_host_image(image_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Complete host image analysis pipeline.

    Returns numeric features and qualitative interpretation.
    """
    features = extract_host_features(image_rgb)
    interpretation = interpret_host_features(features)

    return {
        "features": features.to_dict(),
        "interpretation": interpretation
    }


def analyze_watermark(binary_wm: np.ndarray) -> Dict[str, Any]:
    """
    Complete watermark analysis pipeline.

    Returns numeric features and qualitative interpretation.
    """
    features = extract_watermark_features(binary_wm)
    interpretation = interpret_watermark_features(features)

    return {
        "features": features.to_dict(),
        "interpretation": interpretation
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    dummy_image = np.random.randint(
        0,
        255,
        size=(512, 512, 3),
        dtype=np.uint8
    )

    dummy_watermark = create_binary_pattern_watermark()

    host_result = analyze_host_image(dummy_image)
    watermark_result = analyze_watermark(dummy_watermark)

    print("Host Features:")
    print(host_result)

    print("\nWatermark Features:")
    print(watermark_result)
