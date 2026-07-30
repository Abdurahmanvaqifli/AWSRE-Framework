"""
AWSRE Recommendation Page

Runs the selected watermarking algorithms under the same
conditions and recommends the most suitable strategy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.decision_engine import (
    PriorityWeights,
    recommend_strategy,
)
from watermarking.metrics import (
    calculate_ber,
    calculate_correlation,
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
)


# Import algorithm modules so their @watermarker decorators run.
# These imports register the algorithms in the registry.
import watermarking.dct  # noqa: F401
import watermarking.dwt  # noqa: F401
import watermarking.dct_svd  # noqa: F401

try:
    import watermarking.block_svd  # noqa: F401
except Exception:
    pass

try:
    import watermarking.dwt_svd  # noqa: F401
except Exception:
    pass

from watermarking.registry import create_watermarker


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AWSRE Recommendation",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_METHODS = [
    "DCT",
    "DWT",
    "DCT-SVD",
]


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_grayscale_image(
    uploaded_file,
) -> np.ndarray:
    """
    Convert an uploaded image to a grayscale uint8 array.
    """
    image = Image.open(
        uploaded_file
    ).convert(
        "L"
    )

    return np.asarray(
        image,
        dtype=np.uint8,
    )


def normalize_binary_watermark(
    watermark: np.ndarray,
) -> np.ndarray:
    """
    Convert a watermark to binary values 0 and 1.
    """
    array = np.asarray(
        watermark,
        dtype=np.float32,
    )

    if array.size == 0:
        raise ValueError(
            "The watermark image is empty."
        )

    threshold = (
        0.5
        if float(np.max(array)) <= 1.0
        else 127.5
    )

    return (
        array >= threshold
    ).astype(
        np.uint8
    )


def resize_binary_watermark(
    watermark: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Resize the watermark to (height, width).
    """
    height = int(
        shape[0]
    )

    width = int(
        shape[1]
    )

    if height <= 0 or width <= 0:
        raise ValueError(
            "Watermark dimensions must be positive."
        )

    source = np.uint8(
        np.clip(
            watermark,
            0,
            255,
        )
    )

    resized = Image.fromarray(
        source
    ).resize(
        (
            width,
            height,
        ),
        Image.Resampling.NEAREST,
    )

    return normalize_binary_watermark(
        np.asarray(
            resized,
            dtype=np.uint8,
        )
    )


# ============================================================
# REGISTRY COMPATIBILITY
# ============================================================

def build_algorithm(
    method: str,
    alpha: float,
):
    """
    Create an algorithm while supporting common registry APIs.

    This makes the page more tolerant if create_watermarker()
    expects either a positional method or a named method.
    """
    errors: List[str] = []

    attempts = [
        lambda: create_watermarker(
            method,
            alpha=alpha,
        ),
        lambda: create_watermarker(
            method=method,
            alpha=alpha,
        ),
        lambda: create_watermarker(
            name=method,
            alpha=alpha,
        ),
    ]

    for attempt in attempts:
        try:
            return attempt()
        except Exception as error:
            errors.append(
                str(error)
            )

    raise RuntimeError(
        f"Could not create '{method}'. "
        f"Registry errors: {' | '.join(errors)}"
    )


# ============================================================
# CAPACITY HELPERS
# ============================================================

def determine_watermark_shape(
    algorithm,
    host: np.ndarray,
    requested_size: int,
) -> Tuple[int, int]:
    """
    Select a watermark shape supported by the algorithm.
    """
    requested_size = int(
        requested_size
    )

    if requested_size <= 0:
        raise ValueError(
            "Requested watermark size must be positive."
        )

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
            requested_size,
            maximum_height,
            maximum_width,
        )

    else:
        # Safe fallback for algorithms without an explicit
        # maximum_watermark_shape() method.
        final_size = min(
            requested_size,
            max(
                1,
                host.shape[0] // 8,
            ),
            max(
                1,
                host.shape[1] // 8,
            ),
        )

    if final_size <= 0:
        raise ValueError(
            "The host image is too small for embedding."
        )

    return (
        final_size,
        final_size,
    )


# ============================================================
# BENCHMARK ONE METHOD
# ============================================================

def evaluate_method(
    method: str,
    host: np.ndarray,
    watermark: np.ndarray,
    alpha: float,
    requested_size: int,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    """
    Embed, extract and evaluate one algorithm.
    """
    algorithm = build_algorithm(
        method=method,
        alpha=alpha,
    )

    watermark_shape = determine_watermark_shape(
        algorithm=algorithm,
        host=host,
        requested_size=requested_size,
    )

    prepared_watermark = resize_binary_watermark(
        watermark,
        watermark_shape,
    )

    embedding_result = algorithm.embed(
        host,
        prepared_watermark,
    )

    watermarked_image = np.asarray(
        embedding_result.watermarked_image,
        dtype=np.uint8,
    )

    extraction_result = algorithm.extract(
        host,
        watermarked_image,
        prepared_watermark.shape,
    )

    extracted_watermark = (
        normalize_binary_watermark(
            extraction_result.extracted_watermark
        )
    )

    embedding_runtime = float(
        embedding_result.runtime
    )

    extraction_runtime = float(
        extraction_result.runtime
    )

    capacity_bits = int(
        prepared_watermark.size
    )

    result = {
        "Method": method,
        "Alpha": float(alpha),
        "Watermark Size": (
            f"{prepared_watermark.shape[1]}"
            f"×"
            f"{prepared_watermark.shape[0]}"
        ),
        "Capacity (bits)": capacity_bits,
        "MSE": calculate_mse(
            host,
            watermarked_image,
        ),
        "PSNR (dB)": calculate_psnr(
            host,
            watermarked_image,
        ),
        "SSIM": calculate_ssim(
            host,
            watermarked_image,
        ),
        "BER": calculate_ber(
            prepared_watermark,
            extracted_watermark,
        ),
        "Correlation": calculate_correlation(
            prepared_watermark,
            extracted_watermark,
        ),
        "Embedding Time (s)": embedding_runtime,
        "Extraction Time (s)": extraction_runtime,
        "Total Time (s)": (
            embedding_runtime
            + extraction_runtime
        ),
        "Status": "Success",
    }

    return (
        result,
        watermarked_image,
        extracted_watermark,
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🧠 AWSRE Recommendation Engine"
)

st.write(
    "Upload a host image and watermark, define your priorities, "
    "and let AWSRE rank the available watermarking strategies."
)

st.divider()


# ============================================================
# STEP 1 — UPLOAD FILES
# ============================================================

st.subheader(
    "1. Upload images"
)

upload_left, upload_right = st.columns(
    2
)

with upload_left:
    host_file = st.file_uploader(
        "Host image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff",
        ],
        key="recommendation_host",
    )

with upload_right:
    watermark_file = st.file_uploader(
        "Watermark image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff",
        ],
        key="recommendation_watermark",
    )


# ============================================================
# IMAGE PREVIEW
# ============================================================

host_preview = None
watermark_preview = None

if host_file is not None:
    try:
        host_preview = load_grayscale_image(
            host_file
        )
    except Exception as error:
        st.error(
            f"Host image could not be opened: {error}"
        )

if watermark_file is not None:
    try:
        watermark_preview = load_grayscale_image(
            watermark_file
        )
    except Exception as error:
        st.error(
            f"Watermark could not be opened: {error}"
        )

if (
    host_preview is not None
    or watermark_preview is not None
):
    preview_left, preview_right = st.columns(
        2
    )

    with preview_left:
        if host_preview is not None:
            st.image(
                host_preview,
                caption=(
                    f"Host image: "
                    f"{host_preview.shape[1]}×"
                    f"{host_preview.shape[0]}"
                ),
                use_container_width=True,
                clamp=True,
            )

    with preview_right:
        if watermark_preview is not None:
            st.image(
                watermark_preview,
                caption=(
                    f"Watermark: "
                    f"{watermark_preview.shape[1]}×"
                    f"{watermark_preview.shape[0]}"
                ),
                width=260,
                clamp=True,
            )


# ============================================================
# STEP 2 — CONFIGURATION
# ============================================================

st.divider()

st.subheader(
    "2. Select methods and embedding settings"
)

settings_left, settings_middle, settings_right = (
    st.columns(
        3
    )
)

with settings_left:
    selected_methods = st.multiselect(
        "Algorithms",
        options=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
        help=(
            "Each selected method will be executed under "
            "the same conditions."
        ),
    )

with settings_middle:
    alpha = st.slider(
        "Embedding strength (alpha)",
        min_value=1,
        max_value=50,
        value=20,
        step=1,
    )

with settings_right:
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
            "AWSRE automatically reduces the size when "
            "the host or algorithm has lower capacity."
        ),
    )


# ============================================================
# STEP 3 — USER PRIORITIES
# ============================================================

st.divider()

st.subheader(
    "3. Define your priorities"
)

st.caption(
    "A larger value means that criterion is more important."
)

priority_left, priority_right = st.columns(
    2
)

with priority_left:
    imperceptibility_priority = st.slider(
        "Imperceptibility priority",
        min_value=0,
        max_value=100,
        value=80,
        step=5,
        help=(
            "Prefers high PSNR and SSIM values."
        ),
    )

    robustness_priority = st.slider(
        "Robustness priority",
        min_value=0,
        max_value=100,
        value=80,
        step=5,
        help=(
            "Prefers low BER and high correlation."
        ),
    )

with priority_right:
    speed_priority = st.slider(
        "Speed priority",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help=(
            "Prefers lower embedding and extraction time."
        ),
    )

    capacity_priority = st.slider(
        "Capacity priority",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help=(
            "Prefers algorithms that support more watermark bits."
        ),
    )


# ============================================================
# RUN BUTTON
# ============================================================

st.divider()

run_recommendation = st.button(
    "🚀 Recommend the Best Strategy",
    type="primary",
    use_container_width=True,
)


# ============================================================
# EXECUTION
# ============================================================

if run_recommendation:
    if host_file is None:
        st.error(
            "First upload the host image."
        )
        st.stop()

    if watermark_file is None:
        st.error(
            "First upload the watermark image."
        )
        st.stop()

    if not selected_methods:
        st.error(
            "Select at least one algorithm."
        )
        st.stop()

    total_priority = (
        imperceptibility_priority
        + robustness_priority
        + speed_priority
        + capacity_priority
    )

    if total_priority <= 0:
        st.error(
            "At least one priority must be greater than zero."
        )
        st.stop()

    try:
        host = load_grayscale_image(
            host_file
        )

        watermark = load_grayscale_image(
            watermark_file
        )

    except Exception as error:
        st.exception(
            error
        )
        st.stop()

    rows: List[Dict[str, object]] = []
    watermarked_outputs: Dict[
        str,
        np.ndarray,
    ] = {}
    extracted_outputs: Dict[
        str,
        np.ndarray,
    ] = {}

    progress = st.progress(
        0,
        text="Preparing recommendation...",
    )

    total_methods = len(
        selected_methods
    )

    for index, method in enumerate(
        selected_methods,
        start=1,
    ):
        progress.progress(
            (
                index - 1
            ) / total_methods,
            text=f"Evaluating {method}...",
        )

        try:
            (
                result_row,
                watermarked_image,
                extracted_watermark,
            ) = evaluate_method(
                method=method,
                host=host,
                watermark=watermark,
                alpha=float(alpha),
                requested_size=int(
                    watermark_size
                ),
            )

            rows.append(
                result_row
            )

            watermarked_outputs[
                method
            ] = watermarked_image

            extracted_outputs[
                method
            ] = extracted_watermark

        except Exception as error:
            rows.append({
                "Method": method,
                "Alpha": float(alpha),
                "Watermark Size": "-",
                "Capacity (bits)": np.nan,
                "MSE": np.nan,
                "PSNR (dB)": np.nan,
                "SSIM": np.nan,
                "BER": np.nan,
                "Correlation": np.nan,
                "Embedding Time (s)": np.nan,
                "Extraction Time (s)": np.nan,
                "Total Time (s)": np.nan,
                "Status": f"Failed: {error}",
            })

    progress.progress(
        1.0,
        text="Evaluation completed.",
    )

    raw_results = pd.DataFrame(
        rows
    )

    successful_results = raw_results[
        raw_results["Status"] == "Success"
    ].copy()

    if successful_results.empty:
        st.error(
            "No algorithm completed successfully."
        )

        st.dataframe(
            raw_results,
            use_container_width=True,
            hide_index=True,
        )

        st.stop()

    priorities = PriorityWeights(
        imperceptibility=float(
            imperceptibility_priority
        ),
        robustness=float(
            robustness_priority
        ),
        speed=float(
            speed_priority
        ),
        capacity=float(
            capacity_priority
        ),
    )

    recommendation = recommend_strategy(
        benchmark_results=successful_results,
        priorities=priorities,
    )

    st.session_state[
        "awsre_recommendation"
    ] = recommendation

    st.session_state[
        "awsre_raw_recommendation_results"
    ] = raw_results

    st.session_state[
        "awsre_recommendation_outputs"
    ] = watermarked_outputs

    st.session_state[
        "awsre_extracted_outputs"
    ] = extracted_outputs


# ============================================================
# DISPLAY SAVED RESULT
# ============================================================

if "awsre_recommendation" in st.session_state:
    recommendation = st.session_state[
        "awsre_recommendation"
    ]

    raw_results = st.session_state[
        "awsre_raw_recommendation_results"
    ]

    watermarked_outputs = st.session_state[
        "awsre_recommendation_outputs"
    ]

    extracted_outputs = st.session_state[
        "awsre_extracted_outputs"
    ]

    st.divider()

    st.subheader(
        "🏆 Recommended strategy"
    )

    winner_left, winner_middle, winner_right = (
        st.columns(
            3
        )
    )

    with winner_left:
        st.metric(
            "Recommended Method",
            recommendation.recommended_method,
        )

    with winner_middle:
        st.metric(
            "Confidence",
            f"{recommendation.confidence:.2f}%",
        )

    with winner_right:
        best_score = float(
            recommendation.ranking.iloc[
                0
            ]["Final Score"]
        )

        st.metric(
            "AWSRE Score",
            f"{best_score:.2f}/100",
        )

    st.success(
        f"AWSRE recommends "
        f"{recommendation.recommended_method} "
        f"for the selected priorities."
    )

    st.markdown(
        "### Why was this method selected?"
    )

    for reason in recommendation.explanation:
        st.write(
            f"✅ {reason}"
        )

    st.markdown(
        "### Normalized priority weights"
    )

    weight_columns = st.columns(
        4
    )

    weight_names = [
        (
            "Imperceptibility",
            "imperceptibility",
        ),
        (
            "Robustness",
            "robustness",
        ),
        (
            "Speed",
            "speed",
        ),
        (
            "Capacity",
            "capacity",
        ),
    ]

    for column, (
        label,
        key,
    ) in zip(
        weight_columns,
        weight_names,
    ):
        with column:
            st.metric(
                label,
                (
                    f"{recommendation.weights[key] * 100:.1f}%"
                ),
            )

    st.divider()

    st.subheader(
        "Algorithm ranking"
    )

    ranking_columns = [
        "Rank",
        "Method",
        "Final Score",
        "PSNR (dB)",
        "SSIM",
        "BER",
        "Correlation",
        "Total Time (s)",
        "Capacity (bits)",
        "Imperceptibility Score",
        "Robustness Score",
        "Speed Score",
        "Capacity Score",
    ]

    available_ranking_columns = [
        column
        for column in ranking_columns
        if column
        in recommendation.ranking.columns
    ]

    display_ranking = recommendation.ranking[
        available_ranking_columns
    ].copy()

    display_ranking = display_ranking.round({
        "Final Score": 2,
        "PSNR (dB)": 4,
        "SSIM": 6,
        "BER": 6,
        "Correlation": 6,
        "Total Time (s)": 6,
        "Imperceptibility Score": 4,
        "Robustness Score": 4,
        "Speed Score": 4,
        "Capacity Score": 4,
    })

    st.dataframe(
        display_ranking,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Final score comparison"
    )

    score_chart = (
        recommendation.ranking[
            [
                "Method",
                "Final Score",
            ]
        ]
        .set_index(
            "Method"
        )
    )

    st.bar_chart(
        score_chart,
        use_container_width=True,
    )

    st.subheader(
        "Criterion score comparison"
    )

    criterion_chart = (
        recommendation.ranking[
            [
                "Method",
                "Imperceptibility Score",
                "Robustness Score",
                "Speed Score",
                "Capacity Score",
            ]
        ]
        .set_index(
            "Method"
        )
    )

    st.bar_chart(
        criterion_chart,
        use_container_width=True,
    )

    st.divider()

    st.subheader(
        "Generated images"
    )

    for method in recommendation.ranking[
        "Method"
    ].tolist():
        if method not in watermarked_outputs:
            continue

        with st.expander(
            f"{method} result"
        ):
            output_left, output_right = st.columns(
                2
            )

            with output_left:
                st.image(
                    watermarked_outputs[
                        method
                    ],
                    caption=(
                        f"{method} watermarked image"
                    ),
                    use_container_width=True,
                    clamp=True,
                )

            with output_right:
                st.image(
                    extracted_outputs[
                        method
                    ] * 255,
                    caption=(
                        f"{method} extracted watermark"
                    ),
                    width=260,
                    clamp=True,
                )

    failed_results = raw_results[
        raw_results["Status"] != "Success"
    ]

    if not failed_results.empty:
        with st.expander(
            "Algorithms that failed"
        ):
            st.dataframe(
                failed_results[
                    [
                        "Method",
                        "Status",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    csv_data = recommendation.ranking.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "⬇ Download Recommendation Report",
        data=csv_data,
        file_name=(
            "awsre_recommendation_report.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.info(
        "Upload the two images, choose your priorities, "
        "and press the recommendation button."
    )
