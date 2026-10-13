"""
AWSRE recommendation core (no Streamlit code).

host + watermark features -> ML prediction for every trained
(method, alpha) strategy -> weight-free Pareto filtering -> the
Pareto-optimal candidate closest to the normalized ideal (utopia) point.

This is the same selection rule as
experiments/train_performance_models.py:evaluate_recommendations, which
is how the framework was validated offline. Both pages/6_Intelligent_Mode.py
and the simple home page (pages/0_Home.py) call into this module, so they
always recommend the same strategy for the same input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# ============================================================
# MODEL LOCATION
# These MUST match the model_metadata.json of the loaded models exactly,
# since the models were fit on data with these columns, in this order.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PRIMARY_RESULTS_DIR = PROJECT_ROOT / "experiments" / "scientific_results"
_BACKUP_RESULTS_DIR = PROJECT_ROOT / "experiments" / "scientific_results_v1_backup_3methods"


def _resolve_results_dir() -> Path:
    """Prefer the primary results directory once it has a real, trained
    model_metadata.json (i.e. a finished train_performance_models.py run
    on the current, possibly-larger dataset). Until then, fall back to the
    v1 backup -- fully trained on the original MS-COCO + Oxford-Pet run for
    DCT/DWT/DCT-SVD. No file is copied or moved; this only decides which
    directory to read models from.
    """
    if (_PRIMARY_RESULTS_DIR / "models" / "model_metadata.json").exists():
        return _PRIMARY_RESULTS_DIR
    if (_BACKUP_RESULTS_DIR / "models" / "model_metadata.json").exists():
        return _BACKUP_RESULTS_DIR
    return _PRIMARY_RESULTS_DIR


def _download_models_if_missing() -> None:
    """On the hosted website the trained models are not in the code
    repository (they are ~1.4 GB); fetch them once from the Hugging Face
    model repository into the backup results folder."""
    for folder in (_PRIMARY_RESULTS_DIR, _BACKUP_RESULTS_DIR):
        if (folder / "models" / "model_metadata.json").exists():
            return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="avaqifli/awsre-models",
        local_dir=str(_BACKUP_RESULTS_DIR / "models"),
    )


_download_models_if_missing()
RESULTS_DIR = _resolve_results_dir()
USING_BACKUP_RESULTS = RESULTS_DIR == _BACKUP_RESULTS_DIR
MODEL_DIR = RESULTS_DIR / "models"

CLEAN_QUALITY_MODEL_PATH = MODEL_DIR / "clean_quality_model.joblib"
STRATEGY_PERFORMANCE_MODEL_PATH = MODEL_DIR / "strategy_performance_model.joblib"

# The attack model may have been replaced with the smaller retrained
# version (see experiments/retrain_attack_model_compact.py). Try the
# original name first, then the compact name.
ATTACK_MODEL_CANDIDATES = [
    MODEL_DIR / "attack_recovery_model.joblib",
    MODEL_DIR / "attack_recovery_model_compact.joblib",
]

MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ============================================================
# FEATURE COLUMNS
# ============================================================

HOST_FEATURES = [
    "host_width", "host_height", "host_resolution", "host_aspect_ratio",
    "host_brightness", "host_contrast", "host_entropy", "host_edge_density",
    "host_texture_variance", "host_frequency_energy",
    "host_high_frequency_ratio", "host_dynamic_range", "host_noise_level",
    "host_gradient_complexity", "host_histogram_uniformity",
    "host_smooth_area_ratio", "host_image_complexity_score",
]

WATERMARK_FEATURES = [
    "watermark_density", "watermark_entropy", "watermark_edge_complexity",
    "watermark_connected_components", "watermark_compactness",
    "watermark_fill_ratio", "watermark_symmetry_score",
    "watermark_stroke_complexity", "watermark_structural_complexity",
    "watermark_complexity_score",
]

STRATEGY_FEATURES = ["alpha", "method"]
ATTACK_EXTRA_FEATURES = ["attack_parameter_numeric", "attack_type"]

CLEAN_MODEL_FEATURES = HOST_FEATURES + WATERMARK_FEATURES + STRATEGY_FEATURES
STRATEGY_MODEL_FEATURES = HOST_FEATURES + WATERMARK_FEATURES + STRATEGY_FEATURES
ATTACK_MODEL_FEATURES = HOST_FEATURES + WATERMARK_FEATURES + STRATEGY_FEATURES + ATTACK_EXTRA_FEATURES

# core/feature_extraction.py's HostFeatures/WatermarkFeatures dataclasses do
# NOT use the "host_"/"watermark_" prefixed names the trained models expect
# (those prefixes were added when the scientific dataset CSV was built).
# This maps each required model column to the dataclass field it comes from.
HOST_FEATURE_KEY_MAP = {
    "host_width": "width",
    "host_height": "height",
    "host_resolution": "resolution",
    "host_aspect_ratio": "aspect_ratio",
    "host_brightness": "brightness",
    "host_contrast": "contrast",
    "host_entropy": "entropy",
    "host_edge_density": "edge_density",
    "host_texture_variance": "texture_variance",
    "host_frequency_energy": "frequency_energy",
    "host_high_frequency_ratio": "high_frequency_ratio",
    "host_dynamic_range": "dynamic_range",
    "host_noise_level": "noise_level",
    "host_gradient_complexity": "gradient_complexity",
    "host_histogram_uniformity": "histogram_uniformity",
    "host_smooth_area_ratio": "smooth_area_ratio",
    "host_image_complexity_score": "image_complexity_score",
}

WATERMARK_FEATURE_KEY_MAP = {
    "watermark_density": "density",
    "watermark_entropy": "entropy",
    "watermark_edge_complexity": "edge_complexity",
    "watermark_connected_components": "connected_components",
    "watermark_compactness": "compactness",
    "watermark_fill_ratio": "fill_ratio",
    "watermark_symmetry_score": "symmetry_score",
    "watermark_stroke_complexity": "stroke_complexity",
    "watermark_structural_complexity": "structural_complexity",
    # this one is already stored under its full prefixed name in the dataclass
    "watermark_complexity_score": "watermark_complexity_score",
}

# The scientific data collection always resized watermarks to this size
# (experiments/scientific_config.py: REQUESTED_WATERMARK_SIZE = 32) before
# extracting features, so the same size must be used here.
WATERMARK_FEATURE_SIZE = (32, 32)


# ============================================================
# TRAINED (method, alpha) DOMAIN
# Read live from model_metadata.json so new methods/alphas are picked up
# automatically after retraining.
# ============================================================

_DEFAULT_TRAINED_METHODS = ["DCT", "DWT", "DCT-SVD"]
_DEFAULT_TRAINED_ALPHAS = [10.0, 15.0, 20.0, 25.0, 30.0]


def load_trained_domain(metadata_path: Path) -> Tuple[List[str], List[float]]:
    """Read the trained (method, alpha) domain from model_metadata.json."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        methods = [str(m) for m in metadata.get("methods", [])]
        alphas = [float(a) for a in metadata.get("alpha_domain", [])]

        if methods and alphas:
            return methods, alphas
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    return list(_DEFAULT_TRAINED_METHODS), list(_DEFAULT_TRAINED_ALPHAS)


TRAINED_METHODS, TRAINED_ALPHAS = load_trained_domain(MODEL_METADATA_PATH)


# ============================================================
# MODEL LOADING (callers cache the result, e.g. st.cache_resource)
# ============================================================

def load_models(include_attack_model: bool = True):
    """Return (clean_model, strategy_model, attack_model,
    attack_model_path_used, errors).

    The forests are large; pages that never simulate a single attack (the
    home page) pass include_attack_model=False to skip the attack model and
    save several hundred MB of RAM.
    """
    errors = []

    clean_model = None
    strategy_model = None
    attack_model = None
    attack_model_path_used = None

    if CLEAN_QUALITY_MODEL_PATH.exists():
        clean_model = joblib.load(CLEAN_QUALITY_MODEL_PATH)
    else:
        errors.append(f"Missing: {CLEAN_QUALITY_MODEL_PATH}")

    if STRATEGY_PERFORMANCE_MODEL_PATH.exists():
        strategy_model = joblib.load(STRATEGY_PERFORMANCE_MODEL_PATH)
    else:
        errors.append(f"Missing: {STRATEGY_PERFORMANCE_MODEL_PATH}")

    for candidate in ATTACK_MODEL_CANDIDATES if include_attack_model else []:
        if candidate.exists():
            attack_model = joblib.load(candidate)
            attack_model_path_used = candidate
            break

    return clean_model, strategy_model, attack_model, attack_model_path_used, errors


# ============================================================
# CANDIDATE FRAME
# ============================================================

def build_candidate_frame(
    host_row: Dict[str, float],
    watermark_row: Dict[str, float],
    methods: List[str],
    alphas: List[float],
) -> pd.DataFrame:
    rows = []
    for method in methods:
        for alpha in alphas:
            row = {}
            for feature in HOST_FEATURES:
                row[feature] = host_row[HOST_FEATURE_KEY_MAP[feature]]
            for feature in WATERMARK_FEATURES:
                row[feature] = watermark_row[WATERMARK_FEATURE_KEY_MAP[feature]]
            row["alpha"] = float(alpha)
            row["method"] = method
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# PARETO SELECTION
# Mirrors experiments/train_performance_models.py:pareto_mask /
# evaluate_recommendations.
# ============================================================

def pareto_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    count = len(values)
    keep = np.ones(count, dtype=bool)

    for i in range(count):
        dominated = (
            np.all(values <= values[i], axis=1)
            & np.any(values < values[i], axis=1)
        )
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False

    return keep


def estimate_correlation_from_ber(
    predicted_ber: float, watermark_density: float, n_bits: int
) -> float:
    """Analytic estimate of watermarking.metrics.calculate_correlation()'s
    cosine-similarity score, derived from a predicted BER -- NOT a
    separately trained regression model (the strategy-performance model
    only predicts clean_mse / mean_attack_ber / embedding_time).

    Assumes the predicted BER acts like an i.i.d. bit-flip channel at rate
    e on the binary watermark (n_bits bits, n1 = density * n_bits ones):
        dot(w, w')  = n1 * (1 - e)
        ones(w')    = n1 * (1 - e) + (n_bits - n1) * e
        correlation = dot(w, w') / sqrt(n1 * ones(w'))
    Real attacks are not i.i.d. bit flips, so treat this as indicative.
    """
    e = float(np.clip(predicted_ber, 0.0, 1.0))
    n1 = float(watermark_density) * float(n_bits)
    if n1 <= 0.0 or n1 >= n_bits:
        return float("nan")

    dot = n1 * (1.0 - e)
    ones_after = n1 * (1.0 - e) + (n_bits - n1) * e
    denominator = np.sqrt(n1 * ones_after)
    if denominator <= 0:
        return float("nan")
    return float(np.clip(dot / denominator, 0.0, 1.0))


def select_utopia_candidate(objectives: np.ndarray):
    """
    objectives: (n_candidates, n_objectives), all MINIMIZED.
    Returns (winner row index, Pareto mask, winner's normalized distance to
    the utopia point).
    """
    mask = pareto_mask(objectives)
    pareto_indexes = np.where(mask)[0]

    candidate = objectives[pareto_indexes]
    minimum = np.min(candidate, axis=0)
    maximum = np.max(candidate, axis=0)
    denominator = maximum - minimum
    denominator[denominator == 0] = 1.0

    normalized = (candidate - minimum) / denominator
    distance = np.linalg.norm(normalized, axis=1)

    best_local = int(np.argmin(distance))
    return int(pareto_indexes[best_local]), mask, float(distance[best_local])


# ============================================================
# ONE-CALL RECOMMENDATION
# ============================================================

OBJECTIVE_COLUMNS = ["pred_clean_mse", "pred_mean_attack_ber", "pred_embedding_time_seconds"]


@dataclass
class Recommendation:
    candidates: pd.DataFrame
    winner_index: int
    pareto_flags: np.ndarray
    utopia_distance: float

    @property
    def winner(self) -> pd.Series:
        return self.candidates.iloc[self.winner_index]


def predict_candidates(
    host_row: Dict[str, float],
    watermark_row: Dict[str, float],
    clean_model,
    strategy_model,
    methods: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Predicted clean quality, robustness and time for every strategy."""
    candidates = build_candidate_frame(
        host_row,
        watermark_row,
        methods or TRAINED_METHODS,
        alphas or TRAINED_ALPHAS,
    )

    clean_predictions = clean_model.predict(candidates[CLEAN_MODEL_FEATURES])
    strategy_predictions = strategy_model.predict(candidates[STRATEGY_MODEL_FEATURES])

    candidates["pred_clean_psnr"] = clean_predictions[:, 0]
    candidates["pred_clean_ssim"] = clean_predictions[:, 1]
    candidates["pred_clean_mse"] = strategy_predictions[:, 0]
    candidates["pred_mean_attack_ber"] = strategy_predictions[:, 1]
    candidates["pred_embedding_time_seconds"] = strategy_predictions[:, 2]

    n_bits = int(WATERMARK_FEATURE_SIZE[0] * WATERMARK_FEATURE_SIZE[1])
    density = float(watermark_row["density"])
    candidates["pred_mean_attack_corr"] = candidates["pred_mean_attack_ber"].apply(
        lambda ber: estimate_correlation_from_ber(ber, density, n_bits)
    )
    return candidates


def predict_attack_outcomes(
    candidate: pd.Series,
    attack_model,
    attacks: List[Tuple[str, float]],
) -> pd.DataFrame:
    """attack_recovery_model's predicted BER and correlation for one
    candidate strategy under each (attack, parameter). Like every other
    prediction here, nothing is embedded or attacked."""
    rows = []
    for attack, parameter in attacks:
        row = {feature: candidate[feature] for feature in HOST_FEATURES + WATERMARK_FEATURES}
        row["alpha"] = float(candidate["alpha"])
        row["method"] = str(candidate["method"])
        row["attack_type"] = attack
        row["attack_parameter_numeric"] = float(parameter)
        rows.append(row)

    frame = pd.DataFrame(rows)[ATTACK_MODEL_FEATURES]
    predictions = np.asarray(attack_model.predict(frame))

    return pd.DataFrame({
        "attack": [attack for attack, _ in attacks],
        "parameter": [float(parameter) for _, parameter in attacks],
        "pred_ber": np.clip(predictions[:, 0], 0.0, 1.0),
        "pred_corr": np.clip(predictions[:, 1], 0.0, 1.0),
    })


def recommend(
    host_row: Dict[str, float],
    watermark_row: Dict[str, float],
    clean_model,
    strategy_model,
) -> Recommendation:
    """Predict every candidate and pick the Pareto + ideal-point winner."""
    candidates = predict_candidates(host_row, watermark_row, clean_model, strategy_model)
    objectives = candidates[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
    winner_index, pareto_flags, distance = select_utopia_candidate(objectives)
    candidates["pareto_optimal"] = pareto_flags
    return Recommendation(candidates, winner_index, pareto_flags, distance)
