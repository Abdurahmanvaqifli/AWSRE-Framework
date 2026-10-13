"""
AWSRE Framework
Non-blind attack recovery ("geometric + photometric synchronization")

Every AWSRE watermarking method (DCT, DWT, DCT-SVD, DWT-SVD, BLOCK-SVD) is
non-blind: extraction always has the original host image available, not
just the attacked/watermarked one. This module exploits that property to
UNDO certain attacks before calling algorithm.extract() -- without
touching any of the per-method embedding/extraction math in dct.py,
dwt.py, dct_svd.py, dwt_svd.py or block_svd.py at all. It is a pure,
optional pre-processing wrapper that any page can call.

Two independent, composable corrections:

1. geometric_align() -- fixes rotation and crop (and any residual
   scale/translation) by registering the attacked image back onto the
   original host with OpenCV's ECC (Enhanced Correlation Coefficient)
   algorithm. A coarse-to-fine, multi-hypothesis search (try a handful of
   plausible rotation/crop-scale starting guesses on a small downsampled
   copy first, refine only the best ones at full resolution) makes this
   converge reliably even for fairly aggressive attacks, in well under a
   second on a typical image.

2. photometric_normalize() -- fixes brightness/contrast changes (and
   approximates gamma correction) by fitting a linear intensity model
   between the original and the attacked image and inverting it.

Empirically (see experiments/ or the Comparison page's "Live Head-to-Head"
tab for a live demo on your own images), applying both before extraction:
  - turns near-random BER (~0.5) under a 5-10 degree rotation into ~0.02-0.08
  - turns near-random BER under a 10-20% center crop into ~0.10-0.30
    (still a large win, though crop destroys real information so it is not
    a full recovery to BER=0)
  - turns near-random BER under a brightness/contrast/gamma shift into
    ~0.00
  - is a safe no-op on attacks that are neither geometric nor photometric
    (JPEG, additive noise, blur, sharpen): BER is unchanged either way,
    confirmed by direct testing across all 5 AWSRE methods.

JPEG compression, additive Gaussian noise, blurring and sharpening are
NOT corrected by this module -- they are genuine pixel-level information
loss, not something that can be "undone" just by knowing the original
host's geometry/exposure. AWSRE already tolerates mild versions of these
reasonably well (see the Comparison page); a bit-level redundancy /
majority-vote scheme was prototyped for these but, in testing, helped
blur/sharpen/heavy-JPEG cases while making heavy Gaussian-noise cases
WORSE (repeating a bit that is systematically -- not randomly -- flipped
under strong noise just lets majority voting confidently reproduce the
wrong value more often). Because that trade-off is attack-severity
dependent and not a clean universal win, it was deliberately left out of
this module rather than shipped half-validated; contact the dissertation
notes / experiments/ folder if you want to pick this back up.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _geometric_candidates(w: float, h: float):
    """A handful of plausible starting guesses for the attack's transform:
    identity (no attack), a range of center-crop scale factors, and a range
    of rotation angles. ECC only needs to be started near the true answer,
    not AT it, so this short list is enough to make convergence reliable."""
    cx, cy = w / 2.0, h / 2.0
    candidates = [np.eye(2, 3, dtype=np.float32)]
    for ratio in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
        s = 1.0 / ratio
        candidates.append(np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], dtype=np.float32))
    for angle in (-15, -10, -5, 5, 10, 15):
        theta = np.radians(angle)
        c, s2 = np.cos(theta), np.sin(theta)
        candidates.append(np.array(
            [[c, -s2, cx - (c * cx - s2 * cy)], [s2, c, cy - (s2 * cx + c * cy)]],
            dtype=np.float32,
        ))
    return candidates


def _try_ecc(orig_f, att_f, warp_init, iters, eps=1e-6, gauss=3):
    warp = warp_init.copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, warp = cv2.findTransformECC(orig_f, att_f, warp, cv2.MOTION_AFFINE, criteria, None, gauss)
        return cc, warp
    except cv2.error:
        return -1.0, warp_init


def geometric_align(
    original_gray: np.ndarray,
    attacked_gray: np.ndarray,
    coarse_size: int = 80,
    coarse_iters: int = 40,
    refine_iters: int = 200,
    top_k: int = 2,
) -> Tuple[np.ndarray, float]:
    """Register `attacked_gray` back onto `original_gray`'s coordinate frame.

    Returns (aligned_image, confidence). `confidence` is ECC's correlation
    coefficient in [-1, 1]; values above ~0.9 mean a good geometric match
    was found (i.e. there probably WAS a rotation/crop to undo). Low or
    negative values usually mean there was nothing geometric to correct --
    the returned image is then very close to the untouched input.

    Requires OpenCV; if unavailable, returns the input unchanged with
    confidence 0.0 so callers can treat that as "no correction applied".
    """
    if not CV2_AVAILABLE:
        return attacked_gray, 0.0

    h, w = original_gray.shape[:2]
    scale = coarse_size / max(h, w)
    small_orig = cv2.resize(
        original_gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )
    small_att = cv2.resize(
        attacked_gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )
    so_f = small_orig.astype(np.float32) / 255.0
    sa_f = small_att.astype(np.float32) / 255.0

    scored = []
    for warp_init in _geometric_candidates(small_orig.shape[1], small_orig.shape[0]):
        cc, warp = _try_ecc(so_f, sa_f, warp_init, coarse_iters)
        scored.append((cc, warp))
    scored.sort(key=lambda t: t[0], reverse=True)

    orig_f = original_gray.astype(np.float32) / 255.0
    att_f = attacked_gray.astype(np.float32) / 255.0

    best_cc, best_warp = -2.0, np.eye(2, 3, dtype=np.float32)
    for _cc_small, warp_small in scored[:top_k]:
        warp_full = warp_small.copy()
        warp_full[0, 2] /= scale
        warp_full[1, 2] /= scale
        cc, warp = _try_ecc(orig_f, att_f, warp_full, refine_iters)
        if cc > best_cc:
            best_cc, best_warp = cc, warp

    aligned = cv2.warpAffine(
        attacked_gray, best_warp, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, float(best_cc)


def photometric_normalize(
    original_gray: np.ndarray,
    attacked_gray: np.ndarray,
    border_margin: float = 0.2,
    trim_fraction: float = 0.15,
) -> np.ndarray:
    """Undo a global brightness/contrast (and approximately gamma) shift by
    fitting attacked ~= slope*original + intercept via least squares and
    inverting it. A safe no-op when intensity statistics already match.

    IMPORTANT (found and fixed while validating crop-attack recovery):
    after geometric_align() corrects a crop attack, the outer ring of the
    image it returns is filled in by warpAffine's border handling, not
    genuinely recovered content -- that ring was permanently discarded by
    the crop before the attack resized what remained back up to full
    size. Fitting slope/intercept on ALL pixels let that reconstructed
    ring bias the fit and made crop recovery measurably WORSE than
    geometric alignment alone (BER roughly doubled at a 10-20% crop when
    this first shipped). Excluding a fixed border margin from the fit,
    plus lightly trimming the remaining outliers, removes that
    contamination. It costs nothing on attacks with no such border:
    brightness/contrast/gamma still correct to ~BER 0.0, and rotation
    (whose border artifact is much thinner) is unaffected.
    """
    h, w = original_gray.shape[:2]
    my, mx = int(h * border_margin), int(w * border_margin)
    if h - 2 * my < 8 or w - 2 * mx < 8:
        my = mx = 0  # image too small to spare a margin -- use it whole
    core_o = original_gray[my:h - my, mx:w - mx].astype(np.float64).flatten()
    core_a = attacked_gray[my:h - my, mx:w - mx].astype(np.float64).flatten()
    design = np.vstack([core_o, np.ones_like(core_o)]).T
    try:
        slope, intercept = np.linalg.lstsq(design, core_a, rcond=None)[0]
        if len(core_a) > 100:
            residual = np.abs(core_a - (slope * core_o + intercept))
            threshold = np.quantile(residual, 1.0 - trim_fraction)
            inliers = residual <= threshold
            if inliers.sum() > 100:
                slope, intercept = np.linalg.lstsq(design[inliers], core_a[inliers], rcond=None)[0]
    except np.linalg.LinAlgError:
        return attacked_gray
    if abs(slope) < 1e-6:
        return attacked_gray
    corrected = (attacked_gray.astype(np.float64) - intercept) / slope
    return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)


def recover_before_extraction(original_gray: np.ndarray, attacked_gray: np.ndarray) -> dict:
    """Full recovery pipeline: geometric realignment, then photometric
    normalization. Call this on an attacked/watermarked image and pass the
    returned "corrected" array to algorithm.extract() instead of the raw
    attacked image.

    Returns {"corrected": np.ndarray, "geometric_confidence": float}.
    """
    if not CV2_AVAILABLE:
        return {"corrected": attacked_gray, "geometric_confidence": 0.0}

    aligned, geo_confidence = geometric_align(original_gray, attacked_gray)
    corrected = photometric_normalize(original_gray, aligned)
    return {"corrected": corrected, "geometric_confidence": geo_confidence}
