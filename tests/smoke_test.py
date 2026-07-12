"""
AWSRE Framework Smoke Test

This test verifies that the currently implemented modules:

- import correctly;
- extract host image features;
- extract watermark features;
- create recommendation data models;
- calculate image and watermark metrics;
- serialize experiment records;
- create an in-memory experiment database.

Run from the repository root:

    python -m tests.smoke_test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# IMPORT TESTS
# ============================================================

print("=" * 72)
print("AWSRE FRAMEWORK SMOKE TEST")
print("=" * 72)

print("\n[1/8] Testing module imports...")

from core.feature_extraction import (
    analyze_host_image,
    analyze_watermark,
    create_binary_pattern_watermark,
)

from core.strategy_library import (
    STRATEGY_LIBRARY,
    get_strategy,
    list_methods,
    strategy_to_dataframe,
)

from core.recommendation_models import (
    HostImageRecord,
    WatermarkRecord,
    EmbeddingConfiguration,
    AttackConfiguration,
    ImperceptibilityMetrics,
    RobustnessMetrics,
    RuntimeMetrics,
    ExperimentRecord,
    ExperimentDatabase,
)

from watermarking.base import (
    estimate_capacity,
    image_hash,
    prepare_inputs,
    save_image,
)

from watermarking.metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    calculate_ber,
    calculate_correlation,
    calculate_summary_metrics,
    calculate_watermark_metrics,
    evaluate_embedding,
)

print("✅ All current AWSRE modules imported successfully.")


# ============================================================
# SYNTHETIC TEST DATA
# ============================================================

print("\n[2/8] Creating deterministic test data...")

rng = np.random.default_rng(seed=42)

host_rgb = rng.integers(
    low=0,
    high=256,
    size=(512, 512, 3),
    dtype=np.uint8,
)

host_gray = cv2.cvtColor(
    host_rgb,
    cv2.COLOR_RGB2GRAY,
)

watermark_01 = create_binary_pattern_watermark(
    target_size=(32, 32)
)

watermark_255 = watermark_01 * 255

processed_gray = host_gray.copy()
processed_gray[100:140, 100:140] = np.clip(
    processed_gray[100:140, 100:140].astype(np.int16) + 5,
    0,
    255,
).astype(np.uint8)

extracted_01 = watermark_01.copy()
extracted_01[0, 0] = 1 - extracted_01[0, 0]

extracted_255 = extracted_01 * 255

assert host_rgb.shape == (512, 512, 3)
assert host_gray.shape == (512, 512)
assert watermark_01.shape == (32, 32)
assert set(np.unique(watermark_01)).issubset({0, 1})

print("✅ Synthetic host image and watermark created.")


# ============================================================
# FEATURE EXTRACTION TEST
# ============================================================

print("\n[3/8] Testing feature extraction...")

host_analysis = analyze_host_image(host_rgb)
watermark_analysis = analyze_watermark(watermark_01)

required_host_features = {
    "brightness",
    "contrast",
    "entropy",
    "edge_density",
    "texture_variance",
    "frequency_energy",
    "high_frequency_ratio",
    "dynamic_range",
    "noise_level",
    "sharpness",
    "gradient_complexity",
    "histogram_uniformity",
    "smooth_area_ratio",
    "image_complexity_score",
    "resolution",
    "width",
    "height",
    "aspect_ratio",
}

required_watermark_features = {
    "width",
    "height",
    "payload_bits",
    "density",
    "foreground_ratio",
    "entropy",
    "edge_complexity",
    "connected_components",
    "compactness",
    "fill_ratio",
    "symmetry_score",
    "stroke_complexity",
    "structural_complexity",
    "watermark_complexity_score",
}

host_feature_keys = set(
    host_analysis["features"].keys()
)

watermark_feature_keys = set(
    watermark_analysis["features"].keys()
)

missing_host = required_host_features - host_feature_keys
missing_watermark = (
    required_watermark_features
    - watermark_feature_keys
)

assert not missing_host, (
    f"Missing host features: {missing_host}"
)

assert not missing_watermark, (
    f"Missing watermark features: {missing_watermark}"
)

assert host_analysis["features"]["resolution"] == 512 * 512
assert watermark_analysis["features"]["payload_bits"] == 32 * 32

print("✅ Host and watermark feature extraction passed.")
print(
    "   Host profile:",
    host_analysis["interpretation"]["host_profile"],
)
print(
    "   Image complexity:",
    host_analysis["features"]["image_complexity_score"],
)
print(
    "   Watermark complexity:",
    watermark_analysis["features"][
        "watermark_complexity_score"
    ],
)


# ============================================================
# STRATEGY LIBRARY TEST
# ============================================================

print("\n[4/8] Testing strategy library...")

methods = list_methods()

expected_methods = {
    "DCT",
    "DWT",
    "DCT-SVD",
    "DWT-SVD",
    "BLOCK-SVD",
}

assert set(methods) == expected_methods
assert len(STRATEGY_LIBRARY) == 5

dwt_svd = get_strategy("DWT-SVD")

assert dwt_svd is not None
assert dwt_svd.method == "DWT-SVD"
assert len(dwt_svd.alpha_values()) > 0
assert dwt_svd.supports_size((32, 32))

strategy_df = strategy_to_dataframe()

assert isinstance(strategy_df, pd.DataFrame)
assert len(strategy_df) == 5
assert "Method" in strategy_df.columns

print("✅ Strategy library passed.")
print("   Methods:", ", ".join(methods))


# ============================================================
# METRICS TEST
# ============================================================

print("\n[5/8] Testing metric calculations...")

mse_value = calculate_mse(
    host_gray,
    processed_gray,
)

psnr_value = calculate_psnr(
    host_gray,
    processed_gray,
)

ssim_value = calculate_ssim(
    host_gray,
    processed_gray,
)

ber_01 = calculate_ber(
    watermark_01,
    extracted_01,
)

ber_255 = calculate_ber(
    watermark_255,
    extracted_255,
)

correlation_value = calculate_correlation(
    watermark_255,
    extracted_255,
)

expected_ber = 1 / watermark_01.size

assert mse_value >= 0
assert psnr_value > 0
assert 0 <= ssim_value <= 1

assert abs(ber_01 - expected_ber) < 1e-12
assert abs(ber_255 - expected_ber) < 1e-12

assert -1 <= correlation_value <= 1

summary_metrics = calculate_summary_metrics(
    host_gray,
    processed_gray,
)

watermark_metrics = calculate_watermark_metrics(
    watermark_255,
    extracted_255,
)

complete_metrics = evaluate_embedding(
    host_gray,
    processed_gray,
    watermark_255,
    extracted_255,
)

for metric_name in [
    "MSE",
    "PSNR",
    "SSIM",
    "BER",
    "Correlation",
]:
    assert metric_name in complete_metrics

print("✅ Metrics passed for both 0/1 and 0/255 watermarks.")
print(f"   MSE: {mse_value:.6f}")
print(f"   PSNR: {psnr_value:.4f}")
print(f"   SSIM: {ssim_value:.6f}")
print(f"   BER: {ber_01:.8f}")
print(f"   Correlation: {correlation_value:.6f}")


# ============================================================
# DATA MODEL TEST
# ============================================================

print("\n[6/8] Testing recommendation and experiment models...")

host_record = HostImageRecord.from_feature_analysis(
    file_name="synthetic_host.png",
    analysis=host_analysis,
    benchmark_tag="SMOKE_TEST",
)

watermark_record = WatermarkRecord.from_feature_analysis(
    file_name="synthetic_watermark.png",
    analysis=watermark_analysis,
    watermark_type="Binary Pattern",
    owner_label="AWSRE",
)

embedding_config = EmbeddingConfiguration(
    method="DCT",
    alpha=10,
    watermark_width=32,
    watermark_height=32,
)

attack_config = AttackConfiguration(
    attack_type="none",
    parameter=None,
)

imperceptibility = ImperceptibilityMetrics(
    mse=mse_value,
    psnr=psnr_value,
    ssim=ssim_value,
)

robustness = RobustnessMetrics(
    ber=ber_01,
    correlation=correlation_value,
    extracted_successfully=True,
)

runtime = RuntimeMetrics(
    embedding_time=0.01,
    attack_time=0.0,
    extraction_time=0.01,
)

experiment = ExperimentRecord(
    benchmark_version="0.1.0-smoke-test",
    benchmark_tag="SMOKE_TEST",
    host=host_record,
    watermark=watermark_record,
    embedding=embedding_config,
    attack=attack_config,
    imperceptibility=imperceptibility,
    robustness=robustness,
    runtime=runtime,
    status="SUCCESS",
)

assert experiment.psnr == psnr_value
assert experiment.ber == ber_01
assert experiment.runtime_seconds == 0.02
assert experiment.experiment_key
assert experiment.embedding.candidate_key()
assert experiment.attack.attack_key()

print("✅ Experiment data models passed.")
print("   Experiment ID:", experiment.experiment_id)
print("   Experiment key:", experiment.experiment_key)


# ============================================================
# DATABASE AND SERIALIZATION TEST
# ============================================================

print("\n[7/8] Testing database, JSON and CSV export...")

database = ExperimentDatabase()
database.add(experiment)

assert len(database) == 1
assert database.already_exists(
    experiment.experiment_key
)

database_summary = database.summary()

assert database_summary.total_experiments == 1
assert database_summary.successful_experiments == 1
assert database_summary.failed_experiments == 0

database_df = database.to_dataframe()

assert isinstance(database_df, pd.DataFrame)
assert len(database_df) == 1
assert database_df.iloc[0]["method"] == "DCT"

with tempfile.TemporaryDirectory() as temp_directory:
    temp_path = Path(temp_directory)

    json_path = temp_path / "experiment.json"
    csv_path = temp_path / "experiments.csv"
    image_path = temp_path / "test_image.png"

    experiment.save_json(json_path)
    database.save_csv(csv_path)
    save_image(host_gray, image_path)

    assert json_path.exists()
    assert json_path.stat().st_size > 0

    assert csv_path.exists()
    assert csv_path.stat().st_size > 0

    assert image_path.exists()
    assert image_path.stat().st_size > 0

    reloaded_df = pd.read_csv(csv_path)

    assert len(reloaded_df) == 1
    assert reloaded_df.iloc[0]["method"] == "DCT"

print("✅ JSON, CSV and image writing passed.")


# ============================================================
# BASE HELPERS TEST
# ============================================================

print("\n[8/8] Testing base watermarking helpers...")

prepared_host, prepared_watermark = prepare_inputs(
    host_gray,
    watermark_255,
)

capacity = estimate_capacity(
    prepared_host,
    prepared_watermark,
)

host_digest = image_hash(prepared_host)
watermark_digest = image_hash(prepared_watermark)

assert prepared_host.ndim == 2
assert prepared_watermark.ndim == 2

assert capacity["host_pixels"] == 512 * 512
assert capacity["watermark_pixels"] == 32 * 32
assert 0 < capacity["capacity_ratio"] < 1

assert len(host_digest) == 64
assert len(watermark_digest) == 64

print("✅ Base helpers passed.")
print(
    "   Capacity ratio:",
    f"{capacity['capacity_ratio']:.8f}",
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 72)
print("✅ ALL AWSRE SMOKE TESTS PASSED")
print("=" * 72)

print(
    "\nThe current core, strategy, model, base and metric "
    "modules are ready for the first real embedding algorithm."
)
