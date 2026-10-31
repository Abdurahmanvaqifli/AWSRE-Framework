"""
AWSRE robust blind watermark ("Dayanıqlı" mode on the site).

Same algorithm as research/robust_tiled/robust_wm.py (the measured
prototype), plus the wrappers the Protect & Verify page uses.

It is NOT one of the 15 strategies the ML recommender chooses from and has
no predictions: everything shown for it on the site is measured.

Idea (classic self-referencing / periodic watermark, Kutter 1998 style):

1. A secret key generates one N x N noise tile. It is the sum of a sync
   pattern P0 and one pattern P_i per payload bit (sign = bit value).
   All patterns are band-limited to low/mid frequencies, which survive
   JPEG, downscaling and blur.
2. The tile is repeated over the whole photo (periodic), scaled by a
   texture mask (stronger in busy areas, weaker in flat ones) and added
   to the luminance.
3. Detection is BLIND (no original needed, only the key):
   - high-pass the suspect photo to suppress the host image;
   - its autocorrelation shows peaks on the tile lattice; two lattice
     vectors give the affine transform (scale, rotation, shear) that
     the photo went through;
   - undo that transform and fold the photo into one N x N tile
     (this makes cropping irrelevant: any piece of the photo holds the
     whole tile, just cyclically shifted);
   - find the shift (and one of 8 lattice symmetries) by FFT
     cross-correlation with P0;
   - read each bit as the sign of the correlation with P_i;
   - accept the payload only if its CRC-16 matches.

The payload is 32 bits (e.g. a user / recipient ID) + 16-bit CRC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

TILE = 128
INFO_BITS = 32
CRC_BITS = 16
CODED_BITS = INFO_BITS + CRC_BITS
BAND = (3.0, 22.0)          # cycles per tile (wavelength ~6..43 px)
SYNC_SHARE = 0.5            # fraction of the tile energy spent on sync


# ------------------------------------------------------------
# payload coding
# ------------------------------------------------------------

def crc16(bits: np.ndarray) -> np.ndarray:
    """CRC-16/CCITT over a bit array, returned as 16 bits."""
    reg = 0xFFFF
    for bit in bits.astype(int):
        top = (reg >> 15) & 1
        reg = (reg << 1) & 0xFFFF
        if top ^ bit:
            reg ^= 0x1021
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)


def encode_payload(user_id: int) -> np.ndarray:
    info = np.array([(user_id >> (INFO_BITS - 1 - i)) & 1 for i in range(INFO_BITS)], dtype=np.uint8)
    return np.concatenate([info, crc16(info)])


def decode_payload(bits: np.ndarray) -> Optional[int]:
    info, crc = bits[:INFO_BITS], bits[INFO_BITS:]
    if not np.array_equal(crc16(info), crc):
        return None
    return int("".join(str(int(b)) for b in info), 2)


# ------------------------------------------------------------
# patterns
# ------------------------------------------------------------

def _band_noise(rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.fft2(rng.standard_normal((TILE, TILE)))
    fy = np.fft.fftfreq(TILE) * TILE
    radius = np.hypot(fy[:, None], fy[None, :])
    spectrum[(radius < BAND[0]) | (radius > BAND[1])] = 0
    pattern = np.real(np.fft.ifft2(spectrum))
    return (pattern - pattern.mean()) / pattern.std()


def patterns(key: int) -> np.ndarray:
    """Array (1 + CODED_BITS, TILE, TILE): sync pattern then bit patterns."""
    rng = np.random.default_rng(int(key))
    return np.stack([_band_noise(rng) for _ in range(1 + CODED_BITS)])


def build_tile(key: int, coded: np.ndarray) -> np.ndarray:
    p = patterns(key)
    signs = 2.0 * coded.astype(float) - 1.0
    data = np.tensordot(signs, p[1:], axes=1) / np.sqrt(CODED_BITS)
    tile = np.sqrt(SYNC_SHARE) * p[0] + np.sqrt(1 - SYNC_SHARE) * data
    return tile / tile.std()


# ------------------------------------------------------------
# embedding
# ------------------------------------------------------------

def texture_mask(luma: np.ndarray) -> np.ndarray:
    """Local activity, normalised to mean 1 and clipped to [0.35, 2.5]."""
    y = luma.astype(np.float32)
    mean = cv2.GaussianBlur(y, (0, 0), 4)
    std = np.sqrt(np.maximum(cv2.GaussianBlur(y * y, (0, 0), 4) - mean * mean, 0)) + 2.0
    mask = std / std.mean()
    return np.clip(mask, 0.35, 2.5)


def embed(rgb: np.ndarray, user_id: int, key: int, strength: float = 2.6) -> np.ndarray:
    """Add the tiled pattern to luminance. `strength` = RMS change in grey levels."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    h, w = y.shape
    tile = build_tile(key, encode_payload(user_id)).astype(np.float32)
    tiled = np.tile(tile, (h // TILE + 1, w // TILE + 1))[:h, :w]
    mask = texture_mask(y)
    signal = tiled * mask
    signal *= strength / np.sqrt(np.mean(signal ** 2))
    ycrcb[:, :, 0] = np.clip(np.round(y + signal), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# ------------------------------------------------------------
# detection
# ------------------------------------------------------------

def _prefilter(luma: np.ndarray) -> np.ndarray:
    """Remove most of the host image, keep the band where the mark lives."""
    y = luma.astype(np.float32)
    residual = y - cv2.GaussianBlur(y, (0, 0), 6.0)
    # local normalisation: divides out the texture mask and host energy
    local = np.sqrt(cv2.GaussianBlur(residual * residual, (0, 0), 8.0)) + 1.0
    return residual / local


def _autocorrelation(r: np.ndarray) -> np.ndarray:
    h, w = r.shape
    size = (cv2.getOptimalDFTSize(2 * h), cv2.getOptimalDFTSize(2 * w))
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    spectrum = np.fft.rfft2(r * window, s=size)
    power = np.abs(spectrum) ** 2
    power /= np.maximum(cv2.GaussianBlur(power.astype(np.float32), (0, 0), 3.0), 1e-9)  # whiten
    ac = np.fft.irfft2(power, s=size)
    return np.fft.fftshift(ac)


def _lattice_candidates(ac: np.ndarray, min_r: float, max_r: float, count: int = 12):
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    local_max = ac == cv2.dilate(ac.astype(np.float32), np.ones((7, 7), np.uint8))
    yy, xx = np.nonzero(local_max)
    dy, dx = yy - cy, xx - cx
    r = np.hypot(dy, dx)
    keep = (r >= min_r) & (r <= max_r) & ((dy > 0) | ((dy == 0) & (dx > 0)))  # half plane
    yy, xx, dy, dx = yy[keep], xx[keep], dy[keep], dx[keep]
    order = np.argsort(-ac[yy, xx])[:count]
    out = []
    for i in order:
        y0, x0 = yy[i], xx[i]
        # sub-pixel refinement (parabola)
        def refine(a, b, c):
            d = a - 2 * b + c
            return 0.0 if d == 0 else 0.5 * (a - c) / d
        oy = refine(ac[y0 - 1, x0], ac[y0, x0], ac[y0 + 1, x0])
        ox = refine(ac[y0, x0 - 1], ac[y0, x0], ac[y0, x0 + 1])
        out.append((np.array([dx[i] + ox, dy[i] + oy]), float(ac[y0, x0])))
    return out


def _reduce(v1: np.ndarray, v2: np.ndarray):
    """Lagrange-Gauss reduction: shortest basis of the same 2-D lattice."""
    if np.dot(v1, v1) > np.dot(v2, v2):
        v1, v2 = v2, v1
    for _ in range(20):
        mu = np.round(np.dot(v1, v2) / np.dot(v1, v1))
        v2 = v2 - mu * v1
        if np.dot(v2, v2) >= np.dot(v1, v1):
            break
        v1, v2 = v2, v1
    return v1, v2


def _basis_options(cands, min_r: float, max_r: float, tol: float = 2.5):
    """
    Candidate lattice bases from pairs of autocorrelation peaks.

    A pair is scored by how many of the strong peaks are integer
    combinations of it (the true lattice explains all of them; a
    sub-lattice explains fewer), then by peak strength. Each pair is
    reduced to its shortest basis so only the 8 square-lattice
    symmetries remain ambiguous (handled by the caller).
    """
    vecs = [v for v, _ in cands]
    strength = [s for _, s in cands]
    options = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            b1, b2 = _reduce(vecs[i], vecs[j])
            basis = np.column_stack([b1, b2])
            det = abs(np.linalg.det(basis))
            n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
            if n1 < min_r or n2 > max_r:
                continue
            if det < 0.5 * n1 * n2 or n2 / n1 > 1.6:   # square tile: near-orthogonal, similar lengths
                continue
            coeffs = np.linalg.solve(basis, np.array(vecs).T)
            residual = np.linalg.norm(basis @ np.round(coeffs) - np.array(vecs).T, axis=0)
            ok = residual < tol
            explained = int(np.sum(ok))
            if explained >= 3:
                # least-squares fit on every explained peak: long vectors
                # (2v, 3v, v1+v2 ...) pin the basis down to sub-pixel accuracy
                n = np.round(coeffs[:, ok])
                basis = (np.array(vecs).T[:, ok] @ np.linalg.pinv(n))
            options.append((explained, strength[i] + strength[j], basis))
    options.sort(key=lambda o: (-o[0], -o[1]))
    unique = []
    for _, _, basis in options:
        if all(np.abs(np.abs(np.linalg.det(basis)) - np.abs(np.linalg.det(u))) > 1.0
               or np.linalg.norm(np.abs(basis) - np.abs(u)) > 3.0 for u in unique):
            unique.append(basis)
        if len(unique) == 4:
            break
    return unique


def _fold(r: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Warp so that basis -> (TILE,0),(0,TILE), then fold into one tile."""
    h, w = r.shape
    to_canon = TILE * np.linalg.inv(basis)          # maps image offsets -> canonical
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float).T
    canon = to_canon @ corners
    lo = np.floor(canon.min(axis=1))
    hi = np.ceil(canon.max(axis=1))
    size = (hi - lo).astype(int)
    size = ((size // TILE) + 1) * TILE
    M = np.hstack([to_canon, -lo[:, None]]).astype(np.float32)
    warped = cv2.warpAffine(r, M, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones_like(r), M, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR)
    # keep the folded phase consistent with absolute canonical coords
    shift = (lo.astype(int) % TILE)
    warped = np.roll(warped, (shift[1], shift[0]), axis=(0, 1))
    valid = np.roll(valid, (shift[1], shift[0]), axis=(0, 1))
    ty, tx = warped.shape[0] // TILE, warped.shape[1] // TILE
    acc = warped.reshape(ty, TILE, tx, TILE).sum(axis=(0, 2))
    cnt = valid.reshape(ty, TILE, tx, TILE).sum(axis=(0, 2))
    tile = acc / np.maximum(cnt, 1e-3)
    return tile - tile.mean()


def _dihedral(tile: np.ndarray):
    for k in range(4):
        rot = np.rot90(tile, k)
        yield rot
        yield rot.T


CHASE_BITS = 6   # try flipping any subset of the 6 least reliable bits


def chase_decode(soft: np.ndarray):
    """
    Soft-decision decoding with the CRC as the only check (Chase-2):
    hard decision first, then flip subsets of the least reliable bits
    until the CRC matches. 2^6 = 64 trials, so the CRC's false-accept
    chance rises from 1/65536 to about 1/1000 (still gated by sync).
    """
    hard = (soft > 0).astype(np.uint8)
    weak = np.argsort(np.abs(soft))[:CHASE_BITS]
    trials = sorted(range(1 << CHASE_BITS), key=lambda m: sum(abs(soft[weak[k]]) for k in range(CHASE_BITS) if m >> k & 1))
    for m in trials:
        bits = hard.copy()
        for k in range(CHASE_BITS):
            if m >> k & 1:
                bits[weak[k]] ^= 1
        user_id = decode_payload(bits)
        if user_id is not None:
            return bits, user_id
    return hard, None


@dataclass
class Detection:
    found: bool
    user_id: Optional[int]
    sync_score: float        # z-score of the sync correlation peak
    bit_errors: Optional[int]  # vs. the expected payload, if given
    soft: Optional[np.ndarray] = None


def _sync(r, basis, sync_f, only=None):
    """Fold with `basis`, try the 8 symmetries, return (z, aligned tile, symmetry)."""
    folded = _fold(r, basis)
    best = (-np.inf, None, None)
    for k, variant in enumerate(_dihedral(folded)):
        if only is not None and k != only:
            continue
        corr = np.real(np.fft.ifft2(np.fft.fft2(variant) * sync_f))
        z = (corr.max() - corr.mean()) / (corr.std() + 1e-9)
        if z > best[0]:
            dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
            best = (z, np.roll(variant, (-dy, -dx), axis=(0, 1)), k)
    return best


def _perturbations(step):
    a = step
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return [np.eye(2) * (1 + a), np.eye(2) * (1 - a), rot(a), rot(-a),
            np.diag([1 + a, 1 - a]), np.diag([1 - a, 1 + a]),
            np.array([[1, a], [0, 1]]), np.array([[1, -a], [0, 1]]),
            np.array([[1, 0], [a, 1]]), np.array([[1, 0], [-a, 1]])]


def _identity_option():
    return np.array([[TILE, 0.0], [0.0, TILE]])


def detect(rgb: np.ndarray, key: int, expected_id: Optional[int] = None,
           min_scale: float = 0.25, max_scale: float = 2.0,
           sync_threshold: float = 7.0, _windows: bool = True) -> Detection:
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    r = _prefilter(luma)
    p = patterns(key)
    sync_f = np.conj(np.fft.fft2(p[0]))

    ac = _autocorrelation(r)
    min_r, max_r = TILE * min_scale * 0.9, TILE * max_scale * 1.1
    cands = _lattice_candidates(ac, min_r, max_r)
    options = [_identity_option()] + _basis_options(cands, min_r, max_r)

    best = (-np.inf, None, None, None)
    for basis in options:
        z, aligned, k = _sync(r, basis, sync_f)
        if z > best[0]:
            best = (z, aligned, basis, k)

    # local refinement of the best guess: small scale / rotation / shear steps
    z, aligned, basis, k = best
    for step in (0.01, 0.004, 0.0015):
        improved = True
        while improved:
            improved = False
            for d in _perturbations(step):
                trial = d @ basis
                tz, ta, _ = _sync(r, trial, sync_f, only=k)
                if tz > z + 1e-6:
                    z, aligned, basis, improved = tz, ta, trial, True

    # fallback: if the lattice was not found, sweep isotropic scale directly
    if z < sync_threshold:
        for s in np.geomspace(min_scale, max_scale, 140):
            tz, ta, tk = _sync(r, np.eye(2) * TILE * s, sync_f)
            if tz > z:
                z, aligned, basis, k = tz, ta, np.eye(2) * TILE * s, tk
        for step in (0.004, 0.0015):
            improved = True
            while improved:
                improved = False
                for d in _perturbations(step):
                    trial = d @ basis
                    tz, ta, _ = _sync(r, trial, sync_f, only=k)
                    if tz > z + 1e-6:
                        z, aligned, basis, improved = tz, ta, trial, True

    best = (z, aligned)
    z, aligned = best
    soft = np.tensordot(p[1:], aligned, axes=([1, 2], [0, 1]))
    bits, user_id = chase_decode(soft)
    errors = None
    if expected_id is not None:
        errors = int(np.sum(bits != encode_payload(expected_id)))
    found = (z >= sync_threshold) and (user_id is not None)
    result = Detection(found, user_id if found else None, float(z), errors, soft)

    # Perspective (photo of a screen) is not one affine map for the whole
    # photo, but it is close to one inside a smaller window. If the whole
    # photo fails, try 5 windows (centre + 4 quadrants, each ~1/3 of a side
    # larger than a quarter) independently.
    if not found and _windows:
        h, w = luma.shape
        wh, ww = int(h * 0.6), int(w * 0.6)
        starts = [((h - wh) // 2, (w - ww) // 2), (0, 0), (0, w - ww), (h - wh, 0), (h - wh, w - ww)]
        for y0, x0 in starts:
            sub = detect(rgb[y0:y0 + wh, x0:x0 + ww], key, expected_id, min_scale, max_scale,
                         sync_threshold, _windows=False)
            if sub.found:
                return sub
            if sub.sync_score > result.sync_score and not result.found:
                result = Detection(False, None, sub.sync_score, sub.bit_errors, sub.soft)
    return result


# ============================================================
# SITE WRAPPERS
# ============================================================

SITE_KEY = 2026
DEFAULT_STRENGTH = 2.0      # RMS change ~2 grey levels (PSNR ~42 dB on test photos)
MAX_ID = 2 ** INFO_BITS - 1
MIN_SIDE = TILE * 2         # at least 2x2 tiles
# Detection memory/time grows with the photo; the method is scale-invariant,
# so very large suspects are shrunk first.
MAX_DETECT_SIDE = 1400


def _as_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def _luma(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]


@dataclass
class RobustProtectResult:
    protected: np.ndarray
    owner_id: int
    psnr: float
    ssim: float


def protect_robust(photo: np.ndarray, owner_id: int, key: int = SITE_KEY,
                   strength: float = DEFAULT_STRENGTH) -> RobustProtectResult:
    from watermarking.metrics import calculate_psnr, calculate_ssim

    rgb = _as_rgb(photo)
    if min(rgb.shape[:2]) < MIN_SIDE:
        raise ValueError(f"Photo is too small: both sides must be at least {MIN_SIDE} px.")
    if not 0 <= int(owner_id) <= MAX_ID:
        raise ValueError(f"ID must be between 0 and {MAX_ID}.")
    protected = embed(rgb, int(owner_id), key, strength)
    a, b = _luma(rgb), _luma(protected)
    return RobustProtectResult(protected, int(owner_id), float(calculate_psnr(a, b)), float(calculate_ssim(a, b)))


def verify_blind(suspect: np.ndarray, key: int = SITE_KEY, expected_id: Optional[int] = None) -> Detection:
    """Look for the robust watermark using only the key (no original)."""
    rgb = _as_rgb(suspect)
    h, w = rgb.shape[:2]
    scale = MAX_DETECT_SIDE / max(h, w)
    if scale < 1:
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return detect(rgb, key, expected_id)


# ------------------------------------------------------------
# real-world attacks, shared by both methods in the comparison
# ------------------------------------------------------------

def _jpeg(x, q):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _scale(x, s, interp=cv2.INTER_AREA):
    h, w = x.shape[:2]
    return cv2.resize(x, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def _crop(x, keep_area, seed=2026):
    rng = np.random.default_rng(seed)
    h, w = x.shape[:2]
    k = np.sqrt(keep_area)
    ch, cw = int(h * k), int(w * k)
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    return x[y0:y0 + ch, x0:x0 + cw]


def _rotate_crop(x, deg):
    h, w = x.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    r = cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR)
    a = np.radians(abs(deg))
    # largest axis-aligned rectangle inside the rotated frame (same aspect)
    f = 1.0 / (np.cos(a) + np.sin(a) * max(h, w) / min(h, w))
    ch, cw = int(h * f), int(w * f)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return r[y0:y0 + ch, x0:x0 + cw]


def _instagram(x):
    h, w = x.shape[:2]
    if w / h > 0.8:
        nw = int(h * 0.8)
        x = x[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else:
        nh = int(w / 0.8)
        x = x[(h - nh) // 2:(h - nh) // 2 + nh]
    return _jpeg(_scale(x, 0.6), 80)


def _noise(x, sigma, seed=2026):
    rng = np.random.default_rng(seed)
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 255).astype(np.uint8)


# (id, English label, Azerbaijani label, function)
REAL_WORLD_ATTACKS = [
    ("jpeg70", "JPEG q70", "JPEG q70", lambda x: _jpeg(x, 70)),
    ("jpeg50", "JPEG q50", "JPEG q50", lambda x: _jpeg(x, 50)),
    ("jpeg30", "JPEG q30", "JPEG q30", lambda x: _jpeg(x, 30)),
    ("down50", "Downscale to 50%", "50%-ə kiçiltmə", lambda x: _scale(x, 0.5)),
    ("crop50", "Crop (50% of area kept)", "Kəsmə (sahənin 50%-i qalır)", lambda x: _crop(x, 0.5)),
    ("crop25", "Crop (25% kept) + JPEG 70", "Kəsmə (25% qalır) + JPEG 70", lambda x: _jpeg(_crop(x, 0.25), 70)),
    ("rot5", "Rotate 5° + crop", "Fırlatma 5° + kəsmə", lambda x: _rotate_crop(x, 5)),
    ("insta", "Instagram-like (4:5 crop, 60%, JPEG 80)", "Instagram-a bənzər (4:5 kəsmə, 60%, JPEG 80)", _instagram),
    ("shot", "Screenshot at 80% zoom", "80% miqyasda ekran şəkli", lambda x: _scale(x, 0.8, cv2.INTER_LINEAR)),
    ("noise", "Noise σ8", "Küy σ8", lambda x: _noise(x, 8)),
]


@dataclass
class RealWorldRow:
    attack_id: str
    label_en: str
    label_az: str
    robust_found: bool
    robust_id: Optional[int]
    awsre_found: Optional[bool] = None
    awsre_ber: Optional[float] = None


def real_world_test(robust_protected: np.ndarray, owner_id: int,
                    original: Optional[np.ndarray] = None,
                    awsre_protected: Optional[np.ndarray] = None,
                    mark: Optional[np.ndarray] = None,
                    key: int = SITE_KEY):
    """
    Apply each real-world attack and verify the way the site does.

    Robust photo: verify_blind() with the key only; found means the exact
    ID came back. If an AWSRE-protected photo is also given, it gets the
    same attacks and the site's non-blind verify() (the original is given),
    so both columns are measured, not predicted.
    """
    from core.protect import verify

    rows = []
    for attack_id, en, az, fn in REAL_WORLD_ATTACKS:
        robust = verify_blind(fn(_as_rgb(robust_protected)), key)
        row = RealWorldRow(attack_id, en, az,
                           robust_found=bool(robust.found and robust.user_id == owner_id),
                           robust_id=robust.user_id)
        if awsre_protected is not None:
            awsre = verify(fn(_as_rgb(awsre_protected)), original, mark)
            row.awsre_found, row.awsre_ber = bool(awsre.found), float(awsre.ber)
        rows.append(row)
    return rows
