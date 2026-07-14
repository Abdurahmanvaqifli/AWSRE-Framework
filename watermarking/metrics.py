"""AWSRE defensive image-quality and watermark-robustness metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


def _array(value, *, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is None.")
    result = np.asarray(value)
    if result.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return result


def validate_images(img1, img2):
    first = _array(img1, name="First image")
    second = _array(img2, name="Second image")
    if first.shape != second.shape:
        raise ValueError(
            f"Image shape mismatch {first.shape} != {second.shape}"
        )
    return first, second


def normalize_binary_watermark(watermark):
    array = _array(watermark, name="Watermark")
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] in (3, 4):
            array = np.mean(array[:, :, :3], axis=2)
        else:
            raise ValueError(
                f"Unsupported watermark channels: {array.shape[2]}."
            )
    if array.ndim != 2:
        raise ValueError(
            f"Watermark must be 2D; received {array.shape}."
        )
    values = array.astype(np.float64, copy=False)
    threshold = 0.5 if float(np.max(values)) <= 1.0 else 127.5
    return (values >= threshold).astype(np.uint8)


def calculate_mse(original, compared) -> float:
    first, second = validate_images(original, compared)
    difference = first.astype(np.float64) - second.astype(np.float64)
    result = float(np.mean(np.square(difference)))
    if not np.isfinite(result):
        raise ValueError("MSE is not finite.")
    return result


def calculate_psnr(original, compared, data_range: float = 255.0) -> float:
    mse = calculate_mse(original, compared)
    if mse == 0.0:
        return 100.0
    result = float(10.0 * np.log10((float(data_range) ** 2) / mse))
    if not np.isfinite(result):
        raise ValueError("PSNR is not finite.")
    return result


def calculate_ssim(original, compared, data_range: float = 255.0) -> float:
    first, second = validate_images(original, compared)
    channel_axis = None if first.ndim == 2 else -1
    result = float(
        skimage_ssim(
            first.astype(np.float64),
            second.astype(np.float64),
            data_range=float(data_range),
            channel_axis=channel_axis,
        )
    )
    if not np.isfinite(result):
        raise ValueError("SSIM is not finite.")
    return result


def calculate_ber(original_watermark, extracted_watermark) -> float:
    original_raw, extracted_raw = validate_images(
        original_watermark,
        extracted_watermark,
    )
    original = normalize_binary_watermark(original_raw)
    extracted = normalize_binary_watermark(extracted_raw)
    errors = int(np.count_nonzero(original != extracted))
    result = float(errors / int(original.size))
    if not np.isfinite(result):
        raise ValueError("BER is not finite.")
    return float(np.clip(result, 0.0, 1.0))


def calculate_correlation(original_watermark, extracted_watermark) -> float:
    original_raw, extracted_raw = validate_images(
        original_watermark,
        extracted_watermark,
    )
    first = normalize_binary_watermark(original_raw).astype(np.float64).ravel()
    second = normalize_binary_watermark(extracted_raw).astype(np.float64).ravel()
    denominator = float(
        np.sqrt(np.dot(first, first) * np.dot(second, second))
    )
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return 0.0
    result = float(np.dot(first, second) / denominator)
    if not np.isfinite(result):
        return 0.0
    return float(np.clip(result, -1.0, 1.0))


if __name__ == "__main__":
    original = np.zeros((32, 32), dtype=np.uint8)
    original[::2, ::2] = 1
    assert calculate_ber(original, original.copy()) == 0.0
    assert calculate_correlation(original, original.copy()) == 1.0
    print("✅ AWSRE metrics self test passed.")
