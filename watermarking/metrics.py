"""
AWSRE Metrics Module

Central metric computation module.

Every watermarking algorithm,
benchmark experiment,
and recommendation model
uses these metrics.

No embedding algorithm should compute
metrics internally.
"""

from __future__ import annotations

import cv2
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

# ==========================================================
# IMAGE VALIDATION
# ==========================================================

def validate_images(img1, img2):

    if img1 is None:
        raise ValueError("First image is None.")

    if img2 is None:
        raise ValueError("Second image is None.")

    if img1.shape != img2.shape:

        raise ValueError(
            f"Image shape mismatch "
            f"{img1.shape} != {img2.shape}"
        )


# ==========================================================
# MSE
# ==========================================================

def calculate_mse(
    original,
    processed,
):

    validate_images(
        original,
        processed,
    )

    mse = np.mean(

        (

            original.astype(np.float64)

            -

            processed.astype(np.float64)

        ) ** 2

    )

    return float(mse)


# ==========================================================
# PSNR
# ==========================================================

def calculate_psnr(

    original,

    processed,

):

    validate_images(

        original,

        processed,

    )

    return float(

        peak_signal_noise_ratio(

            original,

            processed,

            data_range=255,

        )

    )


# ==========================================================
# SSIM
# ==========================================================

def calculate_ssim(

    original,

    processed,

):

    validate_images(

        original,

        processed,

    )

    return float(

        structural_similarity(

            original,

            processed,

            data_range=255,

        )

    )


# ==========================================================
# BER
# ==========================================================

def normalize_binary_watermark(watermark):
    """
    Convert 0/1, 0/255 or floating-point watermark arrays
    into a binary uint8 array containing only 0 and 1.
    """
    watermark = np.asarray(watermark)

    if watermark.size == 0:
        raise ValueError("Watermark array is empty.")

    watermark_float = watermark.astype(np.float32)

    if np.max(watermark_float) <= 1.0:
        binary = watermark_float >= 0.5
    else:
        binary = watermark_float >= 127.5

    return binary.astype(np.uint8)


def calculate_ber(
    original_watermark,
    extracted_watermark,
):
    """
    Calculate Bit Error Rate for binary watermarks.

    Supports both:
    - 0/1 arrays
    - 0/255 arrays
    """
    validate_images(
        original_watermark,
        extracted_watermark,
    )

    original = normalize_binary_watermark(
        original_watermark
    )

    extracted = normalize_binary_watermark(
        extracted_watermark
    )

    errors = np.count_nonzero(
        original != extracted
    )

    return float(errors / original.size)

# ==========================================================
# NORMALIZED CORRELATION
# ==========================================================

def calculate_correlation(
    original,
    extracted,
):
    """
    Calculate normalized correlation safely.

    Returns 0.0 when correlation is undefined, such as when
    one of the watermark arrays contains no signal energy.
    """

    validate_images(
        original,
        extracted,
    )

    a = np.asarray(
        original,
        dtype=np.float64,
    ).reshape(-1)

    b = np.asarray(
        extracted,
        dtype=np.float64,
    ).reshape(-1)

    if (
        not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        return 0.0

    numerator = float(
        np.dot(a, b)
    )

    energy_a = float(
        np.dot(a, a)
    )

    energy_b = float(
        np.dot(b, b)
    )

    denominator = float(
        np.sqrt(
            energy_a * energy_b
        )
    )

    if (
        not np.isfinite(denominator)
        or denominator <= 1e-12
    ):
        return 0.0

    correlation = (
        numerator / denominator
    )

    if not np.isfinite(correlation):
        return 0.0

    return float(
        np.clip(
            correlation,
            -1.0,
            1.0,
        )
    )
  # ==========================================================
# RGB METRICS
# ==========================================================

def calculate_rgb_metrics(
    original,
    processed,
):
    """
    Calculate metrics for each RGB channel separately.
    """

    validate_images(original, processed)

    if original.ndim != 3:
        raise ValueError(
            "RGB image required."
        )

    channels = ["Blue", "Green", "Red"]

    results = {}

    for idx, channel in enumerate(channels):

        o = original[:, :, idx]
        p = processed[:, :, idx]

        results[channel] = {

            "MSE": calculate_mse(o, p),

            "PSNR": calculate_psnr(o, p),

            "SSIM": calculate_ssim(o, p),

        }

    return results


# ==========================================================
# SUMMARY METRICS
# ==========================================================

def calculate_summary_metrics(

    original,

    processed,

):

    """
    Compute all standard image-quality metrics.
    """

    return {

        "MSE":
            calculate_mse(
                original,
                processed,
            ),

        "PSNR":
            calculate_psnr(
                original,
                processed,
            ),

        "SSIM":
            calculate_ssim(
                original,
                processed,
            ),

    }


# ==========================================================
# WATERMARK SUMMARY
# ==========================================================

def calculate_watermark_metrics(

    original_watermark,

    extracted_watermark,

):

    """
    Compute watermark comparison metrics.
    """

    return {

        "BER":
            calculate_ber(

                original_watermark,

                extracted_watermark,

            ),

        "Correlation":
            calculate_correlation(

                original_watermark,

                extracted_watermark,

            ),

    }


# ==========================================================
# COMPLETE EVALUATION
# ==========================================================

def evaluate_embedding(

    original,

    watermarked,

    original_watermark,

    extracted_watermark,

):

    """
    One function used by
    benchmark,
    Streamlit,
    desktop,
    and API.
    """

    metrics = {}

    metrics.update(

        calculate_summary_metrics(

            original,

            watermarked,

        )

    )

    metrics.update(

        calculate_watermark_metrics(

            original_watermark,

            extracted_watermark,

        )

    )

    return metrics


# ==========================================================
# BATCH STATISTICS
# ==========================================================

def summarize_experiments(

    experiments,

):

    """
    experiments =
    list of metric dictionaries.
    """

    if len(experiments) == 0:

        return {}

    summary = {}

    keys = [

        "MSE",

        "PSNR",

        "SSIM",

        "BER",

        "Correlation",

    ]

    for key in keys:

        values = [

            x[key]

            for x in experiments

            if key in x

        ]

        if len(values) == 0:

            continue

        summary[key] = {

            "mean":
                float(np.mean(values)),

            "std":
                float(np.std(values)),

            "min":
                float(np.min(values)),

            "max":
                float(np.max(values)),

        }

    return summary


# ==========================================================
# PRETTY PRINT
# ==========================================================

def print_metrics(

    metrics,

):

    print()

    print("=" * 60)

    print("AWSRE Metrics")

    print("=" * 60)

    for key, value in metrics.items():

        if isinstance(value, dict):

            print()

            print(key)

            for k, v in value.items():

                print(

                    f"   {k}: {v:.6f}"

                )

        else:

            print(

                f"{key}: {value:.6f}"

            )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    img1 = np.random.randint(

        0,

        256,

        (512, 512),

        dtype=np.uint8,

    )

    img2 = img1.copy()

    img2[10:50, 10:50] = 255

    wm1 = np.random.randint(

        0,

        2,

        (32, 32),

        dtype=np.uint8,

    ) * 255

    wm2 = wm1.copy()

    wm2[0, 0] = 255 - wm2[0, 0]

    metrics = evaluate_embedding(

        img1,

        img2,

        wm1,

        wm2,

    )

    print_metrics(metrics)
