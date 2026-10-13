"""
AWSRE simple protect / verify workflow (no Streamlit code).

This is the engine behind the simple home page (pages/0_Home.py):

    protect(): colour photo + mark -> AWSRE picks (method, alpha) with the
               trained models (core/recommender.py) -> mark embedded into
               the luminance channel -> colour protected photo.

    verify():  suspect photo + original photo -> every trained method's
               extractor is tried -> "watermark found" when one of them
               recovers the mark far better than chance.

Two choices worth knowing when explaining the results:

1. Colour. The AWSRE methods work on one grey channel. The photo is
   converted to YCrCb, the mark goes into Y (luminance, what the classic
   methods were measured on), and Cr/Cb are left untouched, so the
   protected photo stays in colour.

2. Keyed scrambling. The mark's bits are XOR-ed with a key-dependent
   pseudo-random pattern before embedding. This makes every embedded
   pattern about 50 % ones whatever the logo looks like, so an image with
   no watermark decodes to ~50 % bit errors (chance) instead of happening
   to match a mostly-black logo. The model features and the recommendation
   are computed for this scrambled pattern, because that is what is
   actually embedded.

Extraction is non-blind for all trained methods: verify() needs the
original photo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from core.feature_extraction import extract_host_features, extract_watermark_features
from core.recommender import (
    TRAINED_METHODS,
    WATERMARK_FEATURE_SIZE,
    Recommendation,
    predict_attack_outcomes,
    recommend,
)
from watermarking.metrics import (
    calculate_ber,
    calculate_correlation,
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
)
from watermarking.registry import create_watermarker


DEFAULT_KEY = 2026

# Smallest photo side the 32x32 mark fits into (32 blocks of 8x8 pixels).
MIN_SIDE = WATERMARK_FEATURE_SIZE[0] * 8

# A bit error rate this far below chance (0.5) cannot happen by accident
# for a 1024-bit mark: under "no watermark" the BER is ~N(0.5, 0.016), so
# 0.30 is more than 12 standard deviations away.
FOUND_BER_THRESHOLD = 0.30

# Below this mean absolute luminance difference the suspect photo is the
# original itself (nothing to extract).
IDENTICAL_DIFFERENCE = 0.25


# ============================================================
# MARK
# ============================================================

def default_mark() -> np.ndarray:
    """The built-in 32x32 'AW' mark (binary 0/1), drawn thick enough to
    survive at this size."""
    height, width = WATERMARK_FEATURE_SIZE
    mark = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(mark, "AW", (1, 23), cv2.FONT_HERSHEY_DUPLEX, 0.62, 1, 2, cv2.LINE_AA)
    return (mark > 0).astype(np.uint8)


def mark_from_image(image: np.ndarray) -> np.ndarray:
    """Convert an uploaded logo (grey or colour) to a 32x32 binary mark,
    using the same nearest-neighbour rule as the scientific dataset."""
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grey, (WATERMARK_FEATURE_SIZE[1], WATERMARK_FEATURE_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    return (resized >= 128).astype(np.uint8)


def _key_pattern(key: int) -> np.ndarray:
    generator = np.random.default_rng(int(key))
    return generator.integers(0, 2, size=WATERMARK_FEATURE_SIZE, dtype=np.uint8)


def scramble(mark: np.ndarray, key: int = DEFAULT_KEY) -> np.ndarray:
    """XOR with the key pattern. Applying it twice gives the mark back."""
    return np.bitwise_xor(mark.astype(np.uint8), _key_pattern(key))


# ============================================================
# COLOUR <-> LUMINANCE
# ============================================================

def _split_luma(rgb: np.ndarray):
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return ycrcb[:, :, 0].copy(), ycrcb


def _merge_luma(luma: np.ndarray, ycrcb: np.ndarray) -> np.ndarray:
    merged = ycrcb.copy()
    merged[:, :, 0] = luma
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


# ============================================================
# PROTECT
# ============================================================

@dataclass
class ProtectResult:
    protected: np.ndarray
    method: str
    alpha: float
    psnr: float
    ssim: float
    recommendation: Recommendation


def protect(
    photo: np.ndarray,
    mark: np.ndarray,
    clean_model,
    strategy_model,
    key: int = DEFAULT_KEY,
) -> ProtectResult:
    """Let AWSRE choose the strategy, embed, and return the colour photo."""
    rgb = _as_rgb(photo)
    if min(rgb.shape[:2]) < MIN_SIDE:
        raise ValueError(
            f"Photo is too small: {rgb.shape[1]}x{rgb.shape[0]} px. "
            f"Both sides must be at least {MIN_SIDE} px."
        )

    embedded_bits = scramble(mark, key)

    host_row = extract_host_features(rgb).to_dict()
    watermark_row = extract_watermark_features(embedded_bits).to_dict()
    recommendation = recommend(host_row, watermark_row, clean_model, strategy_model)
    method = str(recommendation.winner["method"])
    alpha = float(recommendation.winner["alpha"])

    luma, ycrcb = _split_luma(rgb)
    algorithm = create_watermarker(method, alpha=alpha)
    marked_luma = np.asarray(algorithm.embed(luma, embedded_bits).watermarked_image, dtype=np.uint8)
    protected = _merge_luma(marked_luma, ycrcb)

    return ProtectResult(
        protected=protected,
        method=method,
        alpha=alpha,
        # Measured on luminance, where the mark lives (only Y changes); this
        # also keeps SSIM's memory use down on large photos.
        psnr=float(calculate_psnr(luma, marked_luma)),
        ssim=float(calculate_ssim(luma, marked_luma)),
        recommendation=recommendation,
    )


# ============================================================
# PREDICTION vs REALITY, AND FIXED STRATEGIES (measured, on this photo)
# ============================================================

# Exactly the attacked conditions the models were trained on
# (experiments/scientific_config.py:ATTACKS without "none"), so the measured
# mean attack BER is the same quantity the strategy model predicts.
TRAINING_ATTACKS = [
    ("jpeg", 90), ("jpeg", 70), ("jpeg", 50),
    ("gaussian_noise", 0.005), ("gaussian_noise", 0.010), ("gaussian_noise", 0.020),
    ("gaussian_blur", 3), ("gaussian_blur", 5),
    ("crop", 0.95), ("crop", 0.90),
    ("rotation", 2.0), ("rotation", 5.0),
    ("brightness", -20.0), ("brightness", 20.0),
    ("contrast", -20.0), ("contrast", 20.0),
]


# Attacks shown as "attacked image -> extracted mark" pairs in the UI.
GALLERY_ATTACKS = [("jpeg", 50), ("gaussian_noise", 0.020), ("gaussian_blur", 5), ("rotation", 5.0)]

THUMBNAIL_SIDE = 320


@dataclass
class AttackOutcome:
    attack: str
    parameter: float
    ber: float
    correlation: float
    predicted_ber: float = float("nan")
    predicted_correlation: float = float("nan")


@dataclass
class GalleryItem:
    attack: str
    parameter: float
    attacked_thumbnail: np.ndarray
    extracted_mark: np.ndarray
    ber: float


@dataclass
class StrategyMeasurement:
    method: str
    alpha: float
    is_awsre: bool
    # Predictions, made before anything is embedded:
    #   PSNR, SSIM          clean_quality_model
    #   MSE, BER            strategy_performance_model (BER: mean over the 16 attacks)
    #   correlation (NC)    attack_recovery_model, averaged over the 16 attacks
    predicted_psnr: float
    predicted_ssim: float
    predicted_mse: float
    predicted_ber: float
    predicted_correlation: float
    psnr: float
    ssim: float
    mse: float
    mean_attack_ber: float
    mean_attack_correlation: float
    attacks: List[AttackOutcome]
    clean_extracted_mark: np.ndarray
    clean_ber: float
    gallery: List[GalleryItem]


def _thumbnail(grey: np.ndarray) -> np.ndarray:
    scale = THUMBNAIL_SIDE / max(grey.shape[:2])
    if scale >= 1:
        return grey.copy()
    size = (max(1, int(grey.shape[1] * scale)), max(1, int(grey.shape[0] * scale)))
    return cv2.resize(grey, size, interpolation=cv2.INTER_AREA)


def _binary(extracted) -> np.ndarray:
    extracted = np.asarray(extracted)
    return (extracted >= (0.5 if extracted.max() <= 1 else 127.5)).astype(np.uint8)


def measure_strategy(
    photo: np.ndarray,
    mark: np.ndarray,
    method: str,
    alpha: float,
    key: int = DEFAULT_KEY,
) -> dict:
    """Really embed one strategy, extract it clean, attack it with the
    training attacks and extract again: no prediction involved. Everything
    is measured on the luminance channel the mark lives in -- the same grey
    channel the models were trained on. Recovered marks are unscrambled so
    they can be shown next to the original logo."""
    from attacks.pipeline import apply_attack_array

    rgb = _as_rgb(photo)
    embedded_bits = scramble(mark, key)
    luma, _ = _split_luma(rgb)
    algorithm = create_watermarker(method, alpha=alpha)
    marked_luma = np.asarray(algorithm.embed(luma, embedded_bits).watermarked_image, dtype=np.uint8)

    clean = _binary(algorithm.extract(luma, marked_luma, embedded_bits.shape).extracted_watermark)

    outcomes, gallery = [], []
    for attack, parameter in TRAINING_ATTACKS:
        attacked = np.asarray(apply_attack_array(marked_luma, attack, parameter, seed=2026), dtype=np.uint8)
        extracted = _binary(algorithm.extract(luma, attacked, embedded_bits.shape).extracted_watermark)
        ber = float(calculate_ber(embedded_bits, extracted))
        outcomes.append(AttackOutcome(
            attack=attack,
            parameter=float(parameter),
            ber=ber,
            correlation=float(calculate_correlation(embedded_bits, extracted)),
        ))
        if (attack, parameter) in GALLERY_ATTACKS:
            gallery.append(GalleryItem(
                attack=attack,
                parameter=float(parameter),
                attacked_thumbnail=_thumbnail(attacked),
                extracted_mark=scramble(extracted, key),
                ber=ber,
            ))

    return {
        "psnr": float(calculate_psnr(luma, marked_luma)),
        "ssim": float(calculate_ssim(luma, marked_luma)),
        "mse": float(calculate_mse(luma, marked_luma)),
        "mean_attack_ber": float(np.mean([o.ber for o in outcomes])),
        "mean_attack_correlation": float(np.mean([o.correlation for o in outcomes])),
        "attacks": outcomes,
        "clean_extracted_mark": scramble(clean, key),
        "clean_ber": float(calculate_ber(embedded_bits, clean)),
        "gallery": gallery,
    }


def measure_candidate(
    photo: np.ndarray,
    mark: np.ndarray,
    recommendation: Recommendation,
    method: str,
    alpha: float,
    attack_model=None,
    key: int = DEFAULT_KEY,
) -> StrategyMeasurement:
    """One strategy's predictions (from the models, no embedding needed)
    next to its real measurement on this photo.

    Only AWSRE's own choice is measured automatically; any other strategy
    is measured only when the user asks to compare with it. Measuring every
    alternative up front would defeat the point of predicting them.
    """
    candidates = recommendation.candidates
    predicted = candidates[(candidates["method"] == method) & (candidates["alpha"] == float(alpha))].iloc[0]
    winner = recommendation.winner
    measured = measure_strategy(photo, mark, method, float(alpha), key)

    predicted_correlation = float("nan")
    if attack_model is not None:
        per_attack = predict_attack_outcomes(predicted, attack_model, TRAINING_ATTACKS)
        for outcome, (_, prediction) in zip(measured["attacks"], per_attack.iterrows()):
            outcome.predicted_ber = float(prediction["pred_ber"])
            outcome.predicted_correlation = float(prediction["pred_corr"])
        predicted_correlation = float(per_attack["pred_corr"].mean())

    return StrategyMeasurement(
        method=method,
        alpha=float(alpha),
        is_awsre=(method, float(alpha)) == (str(winner["method"]), float(winner["alpha"])),
        predicted_psnr=float(predicted["pred_clean_psnr"]),
        predicted_ssim=float(predicted["pred_clean_ssim"]),
        predicted_mse=float(predicted["pred_clean_mse"]),
        predicted_ber=float(predicted["pred_mean_attack_ber"]),
        predicted_correlation=predicted_correlation,
        **measured,
    )


# ============================================================
# VERIFY
# ============================================================

@dataclass
class VerifyResult:
    found: bool
    identical: bool
    method: Optional[str]
    ber: float
    extracted_mark: Optional[np.ndarray]
    per_method_ber: dict

    @property
    def match_percent(self) -> float:
        return 100.0 * (1.0 - self.ber)


def verify(
    suspect: np.ndarray,
    original: np.ndarray,
    mark: np.ndarray,
    key: int = DEFAULT_KEY,
    methods: Optional[List[str]] = None,
) -> VerifyResult:
    """Try every trained method's extractor and keep the best match."""
    original_rgb = _as_rgb(original)
    suspect_rgb = _as_rgb(suspect)

    # A resized copy is brought back to the original's size first.
    if suspect_rgb.shape[:2] != original_rgb.shape[:2]:
        suspect_rgb = cv2.resize(
            suspect_rgb,
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    original_luma, _ = _split_luma(original_rgb)
    suspect_luma, _ = _split_luma(suspect_rgb)

    difference = float(np.mean(np.abs(original_luma.astype(np.float32) - suspect_luma.astype(np.float32))))
    if difference < IDENTICAL_DIFFERENCE:
        return VerifyResult(False, True, None, 0.5, None, {})

    embedded_bits = scramble(mark, key)
    per_method_ber = {}
    best = (None, 1.0, None)

    for method in methods or TRAINED_METHODS:
        algorithm = create_watermarker(method)
        extracted = np.asarray(
            algorithm.extract(original_luma, suspect_luma, embedded_bits.shape).extracted_watermark
        )
        extracted = (extracted >= (0.5 if extracted.max() <= 1 else 127.5)).astype(np.uint8)
        ber = float(calculate_ber(embedded_bits, extracted))
        per_method_ber[method] = ber
        if ber < best[1]:
            best = (method, ber, scramble(extracted, key))

    method, ber, recovered_mark = best
    return VerifyResult(
        found=ber <= FOUND_BER_THRESHOLD,
        identical=False,
        method=method,
        ber=ber,
        extracted_mark=recovered_mark,
        per_method_ber=per_method_ber,
    )
