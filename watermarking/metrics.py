"""
AWSRE Metrics Module

Backward-compatible, defensive implementations for image-quality and
watermark-robustness metrics, including legacy AWSRE aliases.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


def _array(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is None.")
    result = np.asarray(value)
    if result.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return result


def validate_images(
    img1: Any,
    img2: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    first = _array(img1, name="First image")
    second = _array(img2, name="Second image")
    if first.shape != second.shape:
        raise ValueError(
            f"Image shape mismatch {first.shape} != {second.shape}"
        )
    return first, second


def normalize_binary_watermark(watermark: Any) -> np.ndarray:
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
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    threshold = 0.5 if minimum >= 0.0 and maximum <= 1.0 else 127.5
    return (values >= threshold).astype(np.uint8)


def calculate_mse(original: Any, compared: Any) -> float:
    first, second = validate_images(original, compared)
    difference = (
        first.astype(np.float64, copy=False)
        - second.astype(np.float64, copy=False)
    )
    result = float(np.mean(np.square(difference)))
    if not np.isfinite(result):
        raise ValueError("MSE is not finite.")
    return result


def calculate_psnr(
    original: Any,
    compared: Any,
    data_range: float = 255.0,
) -> float:
    mse = calculate_mse(original, compared)
    if mse == 0.0:
        return 100.0

    result = float(
        10.0 * np.log10((float(data_range) ** 2) / mse)
    )
    if not np.isfinite(result):
        raise ValueError("PSNR is not finite.")
    return result


def calculate_ssim(
    original: Any,
    compared: Any,
    data_range: float = 255.0,
) -> float:
    first, second = validate_images(original, compared)

    if first.ndim == 2:
        channel_axis = None
    elif first.ndim == 3:
        channel_axis = -1
    else:
        raise ValueError(
            f"SSIM supports 2D or 3D arrays; received {first.ndim}D."
        )

    result = float(
        skimage_ssim(
            first.astype(np.float64, copy=False),
            second.astype(np.float64, copy=False),
            data_range=float(data_range),
            channel_axis=channel_axis,
        )
    )
    if not np.isfinite(result):
        raise ValueError("SSIM is not finite.")
    return result


def calculate_ber(
    original_watermark: Any,
    extracted_watermark: Any,
) -> float:
    original_raw, extracted_raw = validate_images(
        original_watermark,
        extracted_watermark,
    )

    original = normalize_binary_watermark(original_raw)
    extracted = normalize_binary_watermark(extracted_raw)

    if original.size == 0:
        raise ValueError("Cannot calculate BER for an empty watermark.")

    errors = int(np.count_nonzero(original != extracted))
    result = float(errors / int(original.size))

    if not np.isfinite(result):
        raise ValueError("BER is not finite.")

    return float(np.clip(result, 0.0, 1.0))


def calculate_correlation(
    original_watermark: Any,
    extracted_watermark: Any,
) -> float:
    original_raw, extracted_raw = validate_images(
        original_watermark,
        extracted_watermark,
    )

    first = (
        normalize_binary_watermark(original_raw)
        .astype(np.float64)
        .reshape(-1)
    )
    second = (
        normalize_binary_watermark(extracted_raw)
        .astype(np.float64)
        .reshape(-1)
    )

    numerator = float(np.dot(first, second))
    denominator = float(
        np.sqrt(
            np.dot(first, first)
            * np.dot(second, second)
        )
    )

    if not np.isfinite(denominator) or denominator <= 1e-12:
        return 0.0

    result = float(numerator / denominator)
    if not np.isfinite(result):
        return 0.0

    return float(np.clip(result, -1.0, 1.0))


def evaluate_embedding(
    original_image,
    watermarked_image,
    original_watermark=None,
    extracted_watermark=None,
    *,
    data_range: float = 255.0,
):
    """
    Calculate a complete embedding evaluation.

    With two image arguments:
    - MSE
    - PSNR
    - SSIM

    With all four arguments:
    - MSE
    - PSNR
    - SSIM
    - BER
    - Correlation
    """
    metrics = {
        "MSE": calculate_mse(
            original_image,
            watermarked_image,
        ),
        "PSNR": calculate_psnr(
            original_image,
            watermarked_image,
            data_range=data_range,
        ),
        "SSIM": calculate_ssim(
            original_image,
            watermarked_image,
            data_range=data_range,
        ),
    }

    if (
        original_watermark is None
        and extracted_watermark is None
    ):
        return metrics

    if (
        original_watermark is None
        or extracted_watermark is None
    ):
        raise ValueError(
            "Both original_watermark and extracted_watermark "
            "must be supplied together."
        )

    metrics.update({
        "BER": calculate_ber(
            original_watermark,
            extracted_watermark,
        ),
        "Correlation": calculate_correlation(
            original_watermark,
            extracted_watermark,
        ),
    })

    return metrics

def evaluate_extraction(
    original_watermark: Any,
    extracted_watermark: Any,
) -> Dict[str, float]:
    return {
        "ber": calculate_ber(
            original_watermark,
            extracted_watermark,
        ),
        "correlation": calculate_correlation(
            original_watermark,
            extracted_watermark,
        ),
    }


def calculate_quality_metrics(
    original_image: Any,
    watermarked_image: Any,
    *,
    data_range: float = 255.0,
) -> Dict[str, float]:
    return evaluate_embedding(
        original_image,
        watermarked_image,
        data_range=data_range,
    )


def calculate_robustness_metrics(
    original_watermark: Any,
    extracted_watermark: Any,
) -> Dict[str, float]:
    return evaluate_extraction(
        original_watermark,
        extracted_watermark,
    )



def calculate_watermark_metrics(
    original_watermark: Any,
    extracted_watermark: Any,
) -> Dict[str, float]:
    """
    Backward-compatible alias used by the AWSRE smoke test and
    older framework modules.
    """
    return evaluate_extraction(
        original_watermark,
        extracted_watermark,
    )

def calculate_summary_metrics(
    original_image: Any,
    watermarked_image: Any,
    original_watermark: Optional[Any] = None,
    extracted_watermark: Optional[Any] = None,
    *,
    data_range: float = 255.0,
) -> Dict[str, float]:
    result = evaluate_embedding(
        original_image,
        watermarked_image,
        data_range=data_range,
    )

    if (
        original_watermark is None
        and extracted_watermark is None
    ):
        return result

    if (
        original_watermark is None
        or extracted_watermark is None
    ):
        raise ValueError(
            "Both original_watermark and extracted_watermark "
            "must be supplied together."
        )

    result.update(
        evaluate_extraction(
            original_watermark,
            extracted_watermark,
        )
    )
    return result


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    original_image = rng.integers(
        0,
        256,
        size=(64, 64),
        dtype=np.uint8,
    )
    watermarked_image = original_image.copy()
    watermarked_image[0, 0] = np.uint8(
        (int(watermarked_image[0, 0]) + 1) % 256
    )

    original_watermark = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )
    original_watermark[::2, ::2] = 1
    extracted_watermark = original_watermark.copy()

    summary = calculate_summary_metrics(
        original_image,
        watermarked_image,
        original_watermark,
        extracted_watermark,
    )

    assert summary["ber"] == 0.0
    assert summary["correlation"] == 1.0
    assert all(
        np.isfinite(float(value))
        for value in summary.values()
    )

    print("=" * 72)
    print("AWSRE METRICS SELF TEST")
    print("=" * 72)
    print(summary)
    print("\n✅ AWSRE metrics self test passed.")
