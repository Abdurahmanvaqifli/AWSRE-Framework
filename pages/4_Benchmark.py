"""
AWSRE Framework
Interactive Benchmark Page
"""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from watermarking.metrics import (
    calculate_ber,
    calculate_correlation,
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
)
from watermarking.registry import create_watermarker


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AWSRE Benchmark",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_grayscale_image(uploaded_file) -> np.ndarray:
    """
    Load an uploaded image as an 8-bit grayscale NumPy array.
    """
    image = Image.open(uploaded_file).convert("L")

    return np.asarray(
        image,
        dtype=np.uint8,
    )


def normalize_binary_watermark(
    watermark: np.ndarray,
) -> np.ndarray:
    """
    Convert a grayscale watermark to binary values 0 and 1.
    """
    array = np.asarray(
        watermark,
        dtype=np.float32,
    )

    threshold = (
        0.5
        if float(np.max(array)) <= 1.0
        else 127.5
    )

    return (
        array >= threshold
    ).astype(np.uint8)


def resize_binary_watermark(
    watermark: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a watermark to the requested (height, width) and
    return a binary uint8 array.
    """
    target_height = int(target_shape[0])
    target_width = int(target_shape[1])

    if (
        target_height <= 0
        or target_width <= 0
    ):
        raise ValueError(
            "Target watermark dimensions must be positive."
        )

    pil_watermark = Image.fromarray(
        np.uint8(
            np.clip(
                watermark,
                0,
                255,
            )
        )
    )

    resized = pil_watermark.resize(
        (
            target_width,
            target_height,
        ),
        Image.Resampling.NEAREST,
    )

    return normalize_binary_watermark(
        np.asarray(
            resized,
            dtype=np.uint8,
        )
    )


def determine_watermark_shape(
    algorithm,
    host: np.ndarray,
    requested_size: int,
) -> Tuple[int, int]:
    """
    Determine a square watermark size supported by the algorithm.
    """
    if hasattr(
        algorithm,
        "maximum_watermark_shape",
    ):
        maximum_shape = (
            algorithm.maximum_watermark_shape(
                host
            )
        )

        maximum_height = int(
            maximum_shape[0]
        )

        maximum_width = int(
            maximum_shape[1]
        )

        final_size = min(
            int(requested_size),
            maximum_height,
            maximum_width,
        )
    else:
        final_size = int(
            requested_size
        )

    if final_size <= 0:
        raise ValueError(
            "The host image is too small for watermark embedding."
        )

    return final_size, final_size


def benchmark_method(
    method: str,
    host: np.ndarray,
    watermark: np.ndarray,
    alpha: float,
    requested_watermark_size: int,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    """
    Benchmark one watermarking method.
    """
    algorithm = create_watermarker(
        method=method,
        alpha=alpha,
    )

    watermark_shape = determine_watermark_shape(
        algorithm,
        host,
        requested_watermark_size,
    )

    binary_watermark = resize_binary_watermark(
        watermark,
        watermark_shape,
    )

    embedding_result = algorithm.embed(
        host,
        binary_watermark,
    )

    extraction_result = algorithm.extract(
        host,
        embedding_result.watermarked_image,
        binary_watermark.shape,
    )

    extracted_watermark = normalize_binary_watermark(
        extraction_result.extracted_watermark
    )

    row: Dict[str, object] = {
        "Method": method,
        "Alpha": float(alpha),
        "Watermark Size": (
            f"{binary_watermark.shape[1]}"
            f"×"
            f"{binary_watermark.shape[0]}"
        ),
        "MSE": calculate_mse(
            host,
            embedding_result.watermarked_image,
        ),
        "PSNR (dB)": calculate_psnr(
            host,
            embedding_result.watermarked_image,
        ),
        "SSIM": calculate_ssim(
            host,
            embedding_result.watermarked_image,
        ),
        "BER": calculate_ber(
            binary_watermark,
            extracted_watermark,
        ),
        "Correlation": calculate_correlation(
            binary_watermark,
            extracted_watermark,
        ),
        "Embedding Time (s)": float(
            embedding_result.runtime
        ),
        "Extraction Time (s)": float(
            extraction_result.runtime
        ),
        "Total Time (s)": float(
            embedding_result.runtime
            + extraction_result.runtime
        ),
        "Status": "Success",
    }

    return (
        row,
        embedding_result.watermarked_image,
        extracted_watermark,
    )


def dataframe_to_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """
    Convert a DataFrame to downloadable UTF-8 CSV bytes.
    """
    return dataframe.to_csv(
        index=False,
    ).encode(
        "utf-8"
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 AWSRE Benchmark")

st.caption(
    "Compare invisible watermarking methods using the same "
    "host image, watermark and embedding conditions."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

upload_left, upload_right = st.columns(2)

with upload_left:
    host_file = st.file_uploader(
        "Upload host image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff",
        ],
        key="benchmark_host",
    )

with upload_right:
    watermark_file = st.file_uploader(
        "Upload watermark",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff",
        ],
        key="benchmark_watermark",
    )


# ============================================================
# METHOD CONFIGURATION
# ============================================================

st.subheader("Benchmark configuration")

config_left, config_middle, config_right = st.columns(3)

with config_left:
    selected_methods = st.multiselect(
        "Methods",
        options=[
            "DCT",
            "DWT",
            "DCT-SVD",
        ],
        default=[
            "DCT",
            "DWT",
            "DCT-SVD",
        ],
    )

with config_middle:
    alpha = st.slider(
        "Alpha",
        min_value=1,
        max_value=50,
        value=20,
        step=1,
    )

with config_right:
    watermark_size = st.select_slider(
        "Requested watermark size",
        options=[
            8,
            16,
            24,
            32,
            48,
            64,
        ],
        value=32,
        help=(
            "The page automatically reduces this value "
            "when the selected algorithm has lower capacity."
        ),
    )


# ============================================================
# IMAGE PREVIEW
# ============================================================

if (
    host_file is not None
    and watermark_file is not None
):
    try:
        preview_host = load_grayscale_image(
            host_file
        )

        preview_watermark = load_grayscale_image(
            watermark_file
        )

        preview_left, preview_right = st.columns(2)

        with preview_left:
            st.image(
                preview_host,
                caption=(
                    f"Host image — "
                    f"{preview_host.shape[1]}×"
                    f"{preview_host.shape[0]}"
                ),
                use_container_width=True,
                clamp=True,
            )

        with preview_right:
            st.image(
                preview_watermark,
                caption=(
                    f"Original watermark — "
                    f"{preview_watermark.shape[1]}×"
                    f"{preview_watermark.shape[0]}"
                ),
                width=260,
                clamp=True,
            )

    except Exception as preview_error:
        st.error(
            f"Image preview failed: {preview_error}"
        )


# ============================================================
# BENCHMARK EXECUTION
# ============================================================

st.divider()

run_benchmark = st.button(
    "▶ Run Benchmark",
    type="primary",
    use_container_width=True,
)

if run_benchmark:
    if host_file is None:
        st.error(
            "Upload a host image before running the benchmark."
        )
        st.stop()

    if watermark_file is None:
        st.error(
            "Upload a watermark before running the benchmark."
        )
        st.stop()

    if not selected_methods:
        st.error(
            "Select at least one watermarking method."
        )
        st.stop()

    try:
        host = load_grayscale_image(
            host_file
        )

        watermark = load_grayscale_image(
            watermark_file
        )

    except Exception as image_error:
        st.exception(
            image_error
        )
        st.stop()

    results: List[Dict[str, object]] = []
    generated_images: Dict[str, np.ndarray] = {}
    extracted_images: Dict[str, np.ndarray] = {}

    progress_bar = st.progress(
        0,
        text="Preparing benchmark...",
    )

    total_methods = len(
        selected_methods
    )

    for index, method in enumerate(
        selected_methods,
        start=1,
    ):
        progress_bar.progress(
            (
                index - 1
            )
            / total_methods,
            text=f"Running {method}...",
        )

        try:
            (
                result_row,
                watermarked_image,
                extracted_watermark,
            ) = benchmark_method(
                method=method,
                host=host,
                watermark=watermark,
                alpha=float(alpha),
                requested_watermark_size=int(
                    watermark_size
                ),
            )

            results.append(
                result_row
            )

            generated_images[
                method
            ] = watermarked_image

            extracted_images[
                method
            ] = extracted_watermark

        except Exception as method_error:
            results.append({
                "Method": method,
                "Alpha": float(alpha),
                "Watermark Size": "-",
                "MSE": np.nan,
                "PSNR (dB)": np.nan,
                "SSIM": np.nan,
                "BER": np.nan,
                "Correlation": np.nan,
                "Embedding Time (s)": np.nan,
                "Extraction Time (s)": np.nan,
                "Total Time (s)": np.nan,
                "Status": (
                    f"Failed: {method_error}"
                ),
            })

    progress_bar.progress(
        1.0,
        text="Benchmark completed.",
    )

    result_dataframe = pd.DataFrame(
        results
    )

    st.session_state[
        "benchmark_results"
    ] = result_dataframe

    st.session_state[
        "benchmark_watermarked_images"
    ] = generated_images

    st.session_state[
        "benchmark_extracted_images"
    ] = extracted_images


# ============================================================
# RESULTS
# ============================================================

if "benchmark_results" in st.session_state:
    result_dataframe = st.session_state[
        "benchmark_results"
    ]

    st.success(
        "Benchmark execution completed."
    )

    st.subheader("Results table")

    display_dataframe = (
        result_dataframe.copy()
    )

    numeric_rounding = {
        "MSE": 6,
        "PSNR (dB)": 4,
        "SSIM": 6,
        "BER": 6,
        "Correlation": 6,
        "Embedding Time (s)": 6,
        "Extraction Time (s)": 6,
        "Total Time (s)": 6,
    }

    display_dataframe = display_dataframe.round(
        numeric_rounding
    )

    st.dataframe(
        display_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    successful_rows = result_dataframe[
        result_dataframe[
            "Status"
        ] == "Success"
    ]

    if not successful_rows.empty:
        st.subheader("Quick comparison")

        metric_1, metric_2, metric_3, metric_4 = st.columns(
            4
        )

        best_psnr_index = successful_rows[
            "PSNR (dB)"
        ].idxmax()

        best_ssim_index = successful_rows[
            "SSIM"
        ].idxmax()

        best_ber_index = successful_rows[
            "BER"
        ].idxmin()

        fastest_index = successful_rows[
            "Total Time (s)"
        ].idxmin()

        with metric_1:
            st.metric(
                "Best PSNR",
                successful_rows.loc[
                    best_psnr_index,
                    "Method",
                ],
                (
                    f"{successful_rows.loc[best_psnr_index, 'PSNR (dB)']:.2f} dB"
                ),
            )

        with metric_2:
            st.metric(
                "Best SSIM",
                successful_rows.loc[
                    best_ssim_index,
                    "Method",
                ],
                (
                    f"{successful_rows.loc[best_ssim_index, 'SSIM']:.4f}"
                ),
            )

        with metric_3:
            st.metric(
                "Lowest BER",
                successful_rows.loc[
                    best_ber_index,
                    "Method",
                ],
                (
                    f"{successful_rows.loc[best_ber_index, 'BER']:.4f}"
                ),
            )

        with metric_4:
            st.metric(
                "Fastest Method",
                successful_rows.loc[
                    fastest_index,
                    "Method",
                ],
                (
                    f"{successful_rows.loc[fastest_index, 'Total Time (s)']:.4f} s"
                ),
            )

        st.subheader("Metric charts")

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.caption(
                "Imperceptibility"
            )

            st.bar_chart(
                successful_rows.set_index(
                    "Method"
                )[
                    [
                        "PSNR (dB)",
                    ]
                ]
            )

            st.bar_chart(
                successful_rows.set_index(
                    "Method"
                )[
                    [
                        "SSIM",
                    ]
                ]
            )

        with chart_right:
            st.caption(
                "Extraction quality and speed"
            )

            st.bar_chart(
                successful_rows.set_index(
                    "Method"
                )[
                    [
                        "BER",
                    ]
                ]
            )

            st.bar_chart(
                successful_rows.set_index(
                    "Method"
                )[
                    [
                        "Total Time (s)",
                    ]
                ]
            )

    st.subheader("Generated outputs")

    generated_images = st.session_state.get(
        "benchmark_watermarked_images",
        {},
    )

    extracted_images = st.session_state.get(
        "benchmark_extracted_images",
        {},
    )

    for method in generated_images:
        with st.expander(
            f"{method} outputs"
        ):
            image_left, image_right = st.columns(
                2
            )

            with image_left:
                st.image(
                    generated_images[
                        method
                    ],
                    caption=(
                        f"{method} watermarked image"
                    ),
                    use_container_width=True,
                    clamp=True,
                )

            with image_right:
                st.image(
                    extracted_images[
                        method
                    ] * 255,
                    caption=(
                        f"{method} extracted watermark"
                    ),
                    width=260,
                    clamp=True,
                )

    st.download_button(
        "⬇ Download Benchmark CSV",
        data=dataframe_to_csv_bytes(
            result_dataframe
        ),
        file_name=(
            "awsre_benchmark_results.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info(
        "Upload the required images and press "
        "'Run Benchmark' to generate comparison results."
    )
