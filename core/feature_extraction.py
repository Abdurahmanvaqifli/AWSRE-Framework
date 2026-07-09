"""
AWSRE Feature Extraction Module

This module extracts advanced measurable features from:
1. Host images
2. Watermarks

No Streamlit code is used here.
This module is reusable for:
- Streamlit
- FastAPI
- PyQt desktop app
- research experiments
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
    high_frequency_ratio: float
    dynamic_range: float
    noise_level: float
    sharpness: float
    gradient_complexity: float
    histogram_uniformity: float
    smooth_area_ratio: float
    image_complexity_score: float
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
    foreground_ratio: float
    entropy: float
    edge_complexity: float
    connected_components: int
    compactness: float
    fill_ratio: float
    symmetry_score: float
    stroke_complexity: float
    structural_complexity: float
    watermark_complexity_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# IMAGE LOADING / BASIC HELPERS
# ============================================================

def load_image_as_rgb(file, target_size: Tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    image_np = np.array(image)

    if target_size is not None:
        image_np = cv2.resize(image_np, target_size)

    return image_np


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return float(a / b)


# ============================================================
# HOST IMAGE FEATURES
# ============================================================

def compute_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def compute_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray))


def compute_entropy(gray: np.ndarray) -> float:
    return float(shannon_entropy(gray))


def compute_edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    return float(np.sum(edges > 0) / edges.size)


def compute_texture_variance(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_frequency_energy(gray: np.ndarray) -> float:
    gray_float = np.float32(gray)

    fft = np.fft.fft2(gray_float)
    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)

    return float(np.mean(log_magnitude))


def compute_high_frequency_ratio(gray: np.ndarray) -> float:
    gray_float = np.float32(gray)

    fft = np.fft.fft2(gray_float)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    radius = min(h, w) // 8

    y, x = np.ogrid[:h, :w]
    low_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    high_mask = ~low_mask

    low_energy = np.sum(magnitude[low_mask])
    high_energy = np.sum(magnitude[high_mask])

    return safe_divide(high_energy, low_energy + high_energy)


def compute_dynamic_range(gray: np.ndarray) -> float:
    return float(np.max(gray) - np.min(gray))


def compute_noise_level(gray: np.ndarray) -> float:
    gray_float = np.float32(gray)

    blurred = cv2.GaussianBlur(gray_float, (3, 3), 0)
    noise = gray_float - blurred

    return float(np.std(noise))


def compute_sharpness(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def compute_gradient_complexity(gray: np.ndarray) -> float:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return float(np.mean(magnitude))


def compute_histogram_uniformity(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / np.sum(hist)

    uniformity = np.sum(hist ** 2)

    return float(uniformity)


def compute_smooth_area_ratio(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.abs(laplacian)

    threshold = np.percentile(abs_lap, 35)
    smooth_pixels = np.sum(abs_lap <= threshold)

    return float(smooth_pixels / abs_lap.size)


def compute_resolution(gray: np.ndarray) -> int:
    height, width = gray.shape
    return int(height * width)


def compute_aspect_ratio(gray: np.ndarray) -> float:
    height, width = gray.shape
    return float(width / height)


def normalize_feature(value: float, min_val: float, max_val: float) -> float:
    if max_val - min_val == 0:
        return 0.0

    normalized = (value - min_val) / (max_val - min_val)
    return float(max(0.0, min(1.0, normalized)))


def compute_image_complexity_score(gray: np.ndarray) -> float:
    entropy = compute_entropy(gray)
    edge_density = compute_edge_density(gray)
    texture = compute_texture_variance(gray)
    high_freq = compute_high_frequency_ratio(gray)
    gradient = compute_gradient_complexity(gray)
    noise = compute_noise_level(gray)

    entropy_n = normalize_feature(entropy, 0, 8)
    edge_n = normalize_feature(edge_density, 0, 0.25)
    texture_n = normalize_feature(texture, 0, 2000)
    high_freq_n = normalize_feature(high_freq, 0, 1)
    gradient_n = normalize_feature(gradient, 0, 120)
    noise_n = normalize_feature(noise, 0, 25)

    score = (
        0.25 * entropy_n +
        0.20 * edge_n +
        0.20 * texture_n +
        0.15 * high_freq_n +
        0.10 * gradient_n +
        0.10 * noise_n
    )

    return float(round(score * 100, 4))


def extract_host_features(image_rgb: np.ndarray) -> HostFeatures:
    gray = rgb_to_gray(image_rgb)
    height, width = gray.shape

    return HostFeatures(
        brightness=round(compute_brightness(gray), 4),
        contrast=round(compute_contrast(gray), 4),
        entropy=round(compute_entropy(gray), 4),
        edge_density=round(compute_edge_density(gray), 6),
        texture_variance=round(compute_texture_variance(gray), 4),
        frequency_energy=round(compute_frequency_energy(gray), 4),
        high_frequency_ratio=round(compute_high_frequency_ratio(gray), 6),
        dynamic_range=round(compute_dynamic_range(gray), 4),
        noise_level=round(compute_noise_level(gray), 4),
        sharpness=round(compute_sharpness(gray), 4),
        gradient_complexity=round(compute_gradient_complexity(gray), 4),
        histogram_uniformity=round(compute_histogram_uniformity(gray), 6),
        smooth_area_ratio=round(compute_smooth_area_ratio(gray), 6),
        image_complexity_score=round(compute_image_complexity_score(gray), 4),
        resolution=compute_resolution(gray),
        width=int(width),
        height=int(height),
        aspect_ratio=round(compute_aspect_ratio(gray), 4),
    )


# ============================================================
# WATERMARK CREATION / LOADING
# ============================================================

def load_watermark_as_binary(
    file,
    target_size: Tuple[int, int] = (64, 64),
    threshold: int = 127
) -> np.ndarray:
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
    width, height = target_size
    pattern = np.zeros((height, width), dtype=np.uint8)

    pattern[::2, ::2] = 1
    pattern[1::2, 1::2] = 1

    return pattern


def ensure_binary_watermark(watermark: np.ndarray) -> np.ndarray:
    watermark = np.asarray(watermark)

    if watermark.max() > 1:
        watermark = watermark / 255.0

    binary = np.where(watermark > 0.5, 1, 0)
    return binary.astype(np.uint8)


# ============================================================
# WATERMARK FEATURES
# ============================================================

def compute_payload_bits(binary_wm: np.ndarray) -> int:
    return int(binary_wm.size)


def compute_watermark_density(binary_wm: np.ndarray) -> float:
    return float(np.sum(binary_wm > 0) / binary_wm.size)


def compute_foreground_ratio(binary_wm: np.ndarray) -> float:
    return compute_watermark_density(binary_wm)


def compute_watermark_entropy(binary_wm: np.ndarray) -> float:
    return float(shannon_entropy(binary_wm))


def compute_edge_complexity(binary_wm: np.ndarray) -> float:
    wm_uint8 = (binary_wm * 255).astype(np.uint8)
    edges = cv2.Canny(wm_uint8, 50, 150)

    return float(np.sum(edges > 0) / edges.size)


def compute_connected_components(binary_wm: np.ndarray) -> int:
    binary_wm = binary_wm.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary_wm)

    return int(max(0, num_labels - 1))


def compute_compactness(binary_wm: np.ndarray) -> float:
    wm_uint8 = binary_wm.astype(np.uint8)

    contours, _ = cv2.findContours(
        wm_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    total_area = 0.0
    total_perimeter = 0.0

    for contour in contours:
        total_area += cv2.contourArea(contour)
        total_perimeter += cv2.arcLength(contour, True)

    if total_perimeter == 0:
        return 0.0

    compactness = (4 * np.pi * total_area) / (total_perimeter ** 2)

    return float(max(0.0, min(1.0, compactness)))


def compute_fill_ratio(binary_wm: np.ndarray) -> float:
    coords = np.argwhere(binary_wm > 0)

    if coords.size == 0:
        return 0.0

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
    foreground = np.sum(binary_wm > 0)

    return safe_divide(foreground, bbox_area)


def compute_symmetry_score(binary_wm: np.ndarray) -> float:
    horizontal_flip = np.fliplr(binary_wm)
    vertical_flip = np.flipud(binary_wm)

    h_similarity = 1.0 - np.mean(np.abs(binary_wm - horizontal_flip))
    v_similarity = 1.0 - np.mean(np.abs(binary_wm - vertical_flip))

    return float(max(0.0, min(1.0, (h_similarity + v_similarity) / 2)))


def compute_stroke_complexity(binary_wm: np.ndarray) -> float:
    wm_uint8 = (binary_wm * 255).astype(np.uint8)

    skeleton_like = cv2.Canny(wm_uint8, 50, 150)
    edge_pixels = np.sum(skeleton_like > 0)
    foreground_pixels = np.sum(binary_wm > 0)

    return safe_divide(edge_pixels, foreground_pixels)


def compute_structural_complexity(binary_wm: np.ndarray) -> float:
    entropy = compute_watermark_entropy(binary_wm)
    edge_complexity = compute_edge_complexity(binary_wm)
    components = compute_connected_components(binary_wm)
    compactness = compute_compactness(binary_wm)
    stroke_complexity = compute_stroke_complexity(binary_wm)

    entropy_n = normalize_feature(entropy, 0, 1)
    components_n = min(components / 50.0, 1.0)
    stroke_n = normalize_feature(stroke_complexity, 0, 2)

    complexity = (
        0.30 * entropy_n +
        0.25 * edge_complexity +
        0.20 * components_n +
        0.15 * stroke_n +
        0.10 * (1 - compactness)
    )

    return float(complexity)


def compute_watermark_complexity_score(binary_wm: np.ndarray) -> float:
    density = compute_watermark_density(binary_wm)
    entropy = compute_watermark_entropy(binary_wm)
    edge_complexity = compute_edge_complexity(binary_wm)
    components = compute_connected_components(binary_wm)
    stroke = compute_stroke_complexity(binary_wm)
    fill = compute_fill_ratio(binary_wm)

    density_n = normalize_feature(density, 0, 1)
    entropy_n = normalize_feature(entropy, 0, 1)
    components_n = min(components / 50.0, 1.0)
    stroke_n = normalize_feature(stroke, 0, 2)
    fill_n = normalize_feature(fill, 0, 1)

    score = (
        0.20 * density_n +
        0.25 * entropy_n +
        0.20 * edge_complexity +
        0.15 * components_n +
        0.10 * stroke_n +
        0.10 * fill_n
    )

    return float(round(score * 100, 4))


def extract_watermark_features(binary_wm: np.ndarray) -> WatermarkFeatures:
    binary_wm = ensure_binary_watermark(binary_wm)
    height, width = binary_wm.shape

    return WatermarkFeatures(
        width=int(width),
        height=int(height),
        payload_bits=compute_payload_bits(binary_wm),
        density=round(compute_watermark_density(binary_wm), 6),
        foreground_ratio=round(compute_foreground_ratio(binary_wm), 6),
        entropy=round(compute_watermark_entropy(binary_wm), 4),
        edge_complexity=round(compute_edge_complexity(binary_wm), 6),
        connected_components=compute_connected_components(binary_wm),
        compactness=round(compute_compactness(binary_wm), 6),
        fill_ratio=round(compute_fill_ratio(binary_wm), 6),
        symmetry_score=round(compute_symmetry_score(binary_wm), 6),
        stroke_complexity=round(compute_stroke_complexity(binary_wm), 6),
        structural_complexity=round(compute_structural_complexity(binary_wm), 6),
        watermark_complexity_score=round(
            compute_watermark_complexity_score(binary_wm), 4
        ),
    )


# ============================================================
# QUALITATIVE INTERPRETATION
# ============================================================

def classify_low_medium_high(
    value: float,
    low_threshold: float,
    high_threshold: float
) -> str:
    if value < low_threshold:
        return "Low"
    if value < high_threshold:
        return "Medium"
    return "High"


def classify_brightness(value: float) -> str:
    return classify_low_medium_high(value, 85, 170)


def classify_contrast(value: float) -> str:
    return classify_low_medium_high(value, 35, 70)


def classify_entropy(value: float) -> str:
    return classify_low_medium_high(value, 4, 6.5)


def classify_edge_density(value: float) -> str:
    return classify_low_medium_high(value, 0.03, 0.10)


def classify_texture_variance(value: float) -> str:
    return classify_low_medium_high(value, 100, 800)


def classify_noise_level(value: float) -> str:
    return classify_low_medium_high(value, 3, 12)


def classify_sharpness(value: float) -> str:
    return classify_low_medium_high(value, 100, 800)


def classify_complexity_score(value: float) -> str:
    return classify_low_medium_high(value, 35, 70)


def classify_smooth_area_ratio(value: float) -> str:
    return classify_low_medium_high(value, 0.35, 0.70)


def classify_watermark_density(value: float) -> str:
    if value < 0.20:
        return "Sparse"
    if value < 0.55:
        return "Balanced"
    return "Dense"


def classify_watermark_complexity(value: float) -> str:
    return classify_low_medium_high(value, 35, 70)


def infer_host_profile(features: HostFeatures) -> str:
    if features.smooth_area_ratio > 0.70 and features.edge_density < 0.05:
        return "Smooth / Medical-like"

    if features.edge_density > 0.12 and features.texture_variance > 800:
        return "Highly Textured / Natural"

    if features.entropy > 6.5 and features.high_frequency_ratio > 0.55:
        return "High-frequency Rich"

    if features.contrast < 35 and features.brightness > 150:
        return "Low Contrast / Bright"

    return "Balanced"


def interpret_host_features(features: HostFeatures) -> Dict[str, str]:
    return {
        "brightness_level": classify_brightness(features.brightness),
        "contrast_level": classify_contrast(features.contrast),
        "entropy_level": classify_entropy(features.entropy),
        "edge_density_level": classify_edge_density(features.edge_density),
        "texture_level": classify_texture_variance(features.texture_variance),
        "noise_level": classify_noise_level(features.noise_level),
        "sharpness_level": classify_sharpness(features.sharpness),
        "smooth_area_level": classify_smooth_area_ratio(features.smooth_area_ratio),
        "image_complexity_level": classify_complexity_score(
            features.image_complexity_score
        ),
        "host_profile": infer_host_profile(features),
    }


def interpret_watermark_features(features: WatermarkFeatures) -> Dict[str, str]:
    return {
        "density_level": classify_watermark_density(features.density),
        "complexity_level": classify_watermark_complexity(
            features.watermark_complexity_score
        ),
    }


# ============================================================
# FULL ANALYSIS PIPELINES
# ============================================================

def analyze_host_image(image_rgb: np.ndarray) -> Dict[str, Any]:
    features = extract_host_features(image_rgb)
    interpretation = interpret_host_features(features)

    return {
        "features": features.to_dict(),
        "interpretation": interpretation
    }


def analyze_watermark(binary_wm: np.ndarray) -> Dict[str, Any]:
    features = extract_watermark_features(binary_wm)
    interpretation = interpret_watermark_features(features)

    return {
        "features": features.to_dict(),
        "interpretation": interpretation
    }


def analyze_host_and_watermark(
    image_rgb: np.ndarray,
    binary_wm: np.ndarray
) -> Dict[str, Any]:
    return {
        "host": analyze_host_image(image_rgb),
        "watermark": analyze_watermark(binary_wm)
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
