"""
Benchmark: tiled self-synchronising prototype vs the app's current methods.

Run from the repository root:
    python research/robust_tiled/benchmark_robust.py

Hosts: 8 standard test photos from scikit-image, resized so the long side
is 1024 px. Each attack is applied to each host (seeded, reproducible).

Baselines use exactly what the site uses: 32x32 key-scrambled mark on
luminance, DCT alpha 15 (non-blind, the original is given to the verifier,
and the suspect is resized back to the original's size as core/protect.py
does) and DCT-QIM alpha 15 (blind). Found = BER <= 0.30, the site's rule.

Prototype: 32-bit ID + CRC-16, blind (key only). Found = the exact ID is
decoded and the sync peak passes the threshold. This is a stricter rule
than the baselines' BER threshold.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import skimage.data as data
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import robust_wm as R  # noqa: E402
import watermarking.dct  # noqa: E402,F401
import watermarking.dct_qim  # noqa: E402,F401
from watermarking.registry import create_watermarker  # noqa: E402

KEY = 7
STRENGTH = 2.0          # RMS change ~2 grey levels -> PSNR ~42 dB, same as DCT alpha 15
FOUND_BER = 0.30

HOSTS = ["astronaut", "coffee", "chelsea", "camera", "rocket",
         "immunohistochemistry", "hubble_deep_field", "cat"]


def load_hosts():
    out = {}
    for name in HOSTS:
        img = getattr(data, name)()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img[:, :, :3]
        h, w = img.shape[:2]
        s = 1024 / max(h, w)
        out[name] = cv2.resize(img, (round(w * s), round(h * s)),
                               interpolation=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
    return out


# ------------------------------------------------------------ attacks

def jpeg(x, q):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def scale(x, s, interp=cv2.INTER_AREA):
    h, w = x.shape[:2]
    return cv2.resize(x, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def crop(x, keep_area, rng):
    h, w = x.shape[:2]
    k = np.sqrt(keep_area)
    ch, cw = int(h * k), int(w * k)
    y0 = rng.integers(0, h - ch + 1)
    x0 = rng.integers(0, w - cw + 1)
    return x[y0:y0 + ch, x0:x0 + cw]


def rotate(x, deg):
    """Rotate, then keep the largest centred crop without black corners."""
    h, w = x.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    r = cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR)
    a = np.radians(abs(deg))
    side = min(h, w) / (np.cos(a) + np.sin(a))
    c = (min(h, w) - side) / 2
    return r[int(c + (h - min(h, w)) / 2):int(c + (h - min(h, w)) / 2 + side),
             int(c + (w - min(h, w)) / 2):int(c + (w - min(h, w)) / 2 + side)]


def instagram(x, rng):
    """Centre crop to 4:5, resize to 60%, JPEG 80."""
    h, w = x.shape[:2]
    if w / h > 0.8:
        nw = int(h * 0.8)
        x0 = (w - nw) // 2
        x = x[:, x0:x0 + nw]
    else:
        nh = int(w / 0.8)
        y0 = (h - nh) // 2
        x = x[y0:y0 + nh]
    return jpeg(scale(x, 0.6), 80)


def screenshot(x, rng):
    """Shown on screen at 80% zoom (bilinear), screenshot saved as PNG."""
    return scale(x, 0.8, cv2.INTER_LINEAR)


def gamma(x, g):
    return np.clip(255 * (x / 255.0) ** g, 0, 255).astype(np.uint8)


def camera(x, rng, tilt=0.04):
    """
    Simulated phone photo of a screen: perspective (corners moved up to
    `tilt` of the size), 60% resolution, lens blur, gamma/contrast change,
    sensor noise, JPEG 80. A simulation, not a real camera.
    """
    h, w = x.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-tilt, tilt, size=(4, 2)) * [w, h]
    dst = np.float32(src + jitter)
    P = cv2.getPerspectiveTransform(src, dst)
    y = cv2.warpPerspective(x, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = int(tilt * max(w, h)) + 2                      # drop border garbage
    y = y[m:h - m, m:w - m]
    y = scale(y, 0.6, cv2.INTER_LINEAR)
    y = cv2.GaussianBlur(y, (0, 0), 1.0)
    y = gamma(y, 1.15).astype(np.float32)
    y = 0.9 * (y - 128) + 128 + rng.normal(0, 3.0, y.shape)
    return jpeg(np.clip(y, 0, 255).astype(np.uint8), 80)


def noise(x, sigma, rng):
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 255).astype(np.uint8)


ATTACKS = {
    "none":               lambda x, r: x,
    "JPEG q70":           lambda x, r: jpeg(x, 70),
    "JPEG q50":           lambda x, r: jpeg(x, 50),
    "JPEG q30":           lambda x, r: jpeg(x, 30),
    "Downscale 75%":      lambda x, r: scale(x, 0.75),
    "Downscale 50%":      lambda x, r: scale(x, 0.5),
    "Downscale 35%":      lambda x, r: scale(x, 0.35),
    "Crop 50% area":      lambda x, r: crop(x, 0.5, r),
    "Crop 25% area":      lambda x, r: crop(x, 0.25, r),
    "Crop 25% + JPEG 70": lambda x, r: jpeg(crop(x, 0.25, r), 70),
    "Rotate 5° + crop":   lambda x, r: rotate(x, 5),
    "Rotate 20° + crop":  lambda x, r: rotate(x, 20),
    "Noise σ8":           lambda x, r: noise(x, 8, r),
    "Brightness/contrast": lambda x, r: np.clip(1.3 * (x.astype(float) - 128) + 158, 0, 255).astype(np.uint8),
    "Instagram-like":     instagram,
    "Screenshot 80%":     screenshot,
    "Screen→camera (sim.)": camera,
    "Screen→camera strong tilt (sim.)": lambda x, r: camera(x, r, tilt=0.08),
}


# ------------------------------------------------------------ methods

def luma(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)[:, :, 0]


def baseline_embed(rgb, method, bits):
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    algo = create_watermarker(method, alpha=15)
    ycc[:, :, 0] = np.asarray(algo.embed(ycc[:, :, 0], bits).watermarked_image, dtype=np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def baseline_ber(original, suspect, method, bits):
    if suspect.shape[:2] != original.shape[:2]:
        suspect = cv2.resize(suspect, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    algo = create_watermarker(method, alpha=15)
    if method == "DCT-QIM":
        ext = algo.extract_blind(luma(suspect), bits.shape).extracted_watermark
    else:
        ext = algo.extract(luma(original), luma(suspect), bits.shape).extracted_watermark
    ext = np.asarray(ext)
    ext = (ext >= (0.5 if ext.max() <= 1 else 127.5)).astype(np.uint8)
    return float(np.mean(ext != bits))


def quality(a, b):
    ya, yb = luma(a).astype(float), luma(b).astype(float)
    psnr = 10 * np.log10(255 ** 2 / np.mean((ya - yb) ** 2))
    ssim = structural_similarity(ya, yb, data_range=255)
    return psnr, ssim


def main():
    hosts = load_hosts()
    rng_bits = np.random.default_rng(2026)
    rows, quality_rows, false_pos = [], [], []

    from multiprocessing import Pool
    with Pool(4) as pool:
        parts = pool.map(run_host, list(enumerate(hosts.items())))
    for r1, q1, f1 in parts:
        rows += r1
        quality_rows += q1
        false_pos += f1
    _report(hosts, rows, quality_rows, false_pos)


def run_host(item):
    hi, (name, host) = item
    rng_bits = np.random.default_rng(2026 + hi)
    rows, quality_rows, false_pos = [], [], []
    if True:
        user_id = int(rng_bits.integers(0, 2 ** 32))
        bits = rng_bits.integers(0, 2, (32, 32), dtype=np.uint8)
        marked = {
            "Tiled prototype (blind)": R.embed(host, user_id, KEY, STRENGTH),
            "DCT α15 (site, non-blind)": baseline_embed(host, "DCT", bits),
            "DCT-QIM α15 (blind)": baseline_embed(host, "DCT-QIM", bits),
        }
        for method, img in marked.items():
            p, s = quality(host, img)
            quality_rows.append({"host": name, "method": method, "psnr": p, "ssim": s})

        # false positives: unmarked host, and marked host read with a wrong key
        for k in range(5):
            d = R.detect(host, KEY + 100 + k)
            false_pos.append({"case": "unmarked", "found": bool(d.found), "z": d.sync_score})
            d = R.detect(marked["Tiled prototype (blind)"], KEY + 200 + k)
            false_pos.append({"case": "wrong key", "found": bool(d.found), "z": d.sync_score})

        for attack, fn in ATTACKS.items():
            for method, img in marked.items():
                rng = np.random.default_rng(hi * 1000 + len(attack))
                attacked = fn(img, rng)
                t0 = time.time()
                if method.startswith("Tiled"):
                    d = R.detect(attacked, KEY, expected_id=user_id)
                    found = bool(d.found and d.user_id == user_id)
                    wrong_id = bool(d.found and d.user_id != user_id)
                    ber = d.bit_errors / R.CODED_BITS
                    z = d.sync_score
                else:
                    ber = baseline_ber(host, attacked, "DCT" if method.startswith("DCT α") else "DCT-QIM", bits)
                    found, wrong_id, z = ber <= FOUND_BER, False, None
                rows.append({"host": name, "attack": attack, "method": method, "found": found,
                             "wrong_id": wrong_id, "raw_ber": ber, "z": z,
                             "seconds": time.time() - t0, "size": list(attacked.shape[:2])})
        print("done", name, flush=True)
    return rows, quality_rows, false_pos


def _report(hosts, rows, quality_rows, false_pos):
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps({"rows": rows, "quality": quality_rows, "false_positive": false_pos}, indent=1, default=float))

    # summary
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print("\nQuality (mean over hosts)")
    for m in methods:
        q = [r for r in quality_rows if r["method"] == m]
        print(f"  {m:28s} PSNR {np.mean([r['psnr'] for r in q]):.1f} dB  SSIM {np.mean([r['ssim'] for r in q]):.4f}")
    print("\nFound (of %d hosts) / mean raw BER" % len(hosts))
    print(f"  {'attack':34s}" + "".join(f"{m[:24]:>28s}" for m in methods))
    for a in ATTACKS:
        line = f"  {a:34s}"
        for m in methods:
            rr = [r for r in rows if r["attack"] == a and r["method"] == m]
            line += f"{sum(r['found'] for r in rr):>14d}/{len(rr)}  BER {np.mean([r['raw_ber'] for r in rr]):.2f}"
        print(line)
    print("\nWrong ID accepted:", sum(r["wrong_id"] for r in rows))
    fp = false_pos
    print("False positives:", sum(f["found"] for f in fp), "of", len(fp),
          " max z:", round(max(f["z"] for f in fp), 2))
    tiled = [r for r in rows if r["method"].startswith("Tiled")]
    print("Prototype detect time: mean %.2fs" % np.mean([r["seconds"] for r in tiled]))


if __name__ == "__main__":
    main()
