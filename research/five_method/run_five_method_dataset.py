"""AWSRE 5(+1)-method scientific dataset runner (parallel, resumable).

A parallel version of experiments/scientific_dataset_runner.py. Every row
is computed exactly the same way (same features, same metrics, same
experiment key, same per-row attack seed), so rows already in the SQLite
database from the earlier, half-finished 5-method run are recognised and
skipped. Only the work is spread over several processes: one host image
(with its 2 watermarks, every method, alpha and attack) per task.

Run from the repository root:
    python research/five_method/run_five_method_dataset.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from five_method_config import (  # noqa: E402
    ALPHAS, ATTACKS, CSV_PATH, DATABASE_PATH, ERROR_LOG_PATH, IMAGE_EXTENSIONS,
    LOCAL_BLOCK_SIZE, MAX_IMAGES_PER_DATASET, METHODS, MS_COCO_DIR, OUTPUT_DIR,
    OXFORD_PET_DIR, RANDOM_SEED, REQUESTED_WATERMARK_SIZE, WATERMARK_DIR,
    WATERMARKS_PER_HOST, WORKERS,
)


# ============================================================
# HELPERS (identical to the original runner)
# ============================================================

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash_text(text: str) -> str:
    return stable_hash_bytes(text.encode("utf-8"))


def deterministic_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") ^ int(RANDOM_SEED)


def normalize_binary_watermark(watermark: np.ndarray) -> np.ndarray:
    array = np.asarray(watermark, dtype=np.float32)
    threshold = 0.5 if float(np.max(array)) <= 1.0 else 127.5
    return (array >= threshold).astype(np.uint8)


def resize_binary_watermark(watermark: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    source = np.uint8(np.clip(watermark, 0, 255))
    resized = Image.fromarray(source).resize((width, height), Image.Resampling.NEAREST)
    return normalize_binary_watermark(np.asarray(resized, dtype=np.uint8))


def determine_watermark_shape(algorithm, host: np.ndarray, requested_size: int) -> Tuple[int, int]:
    if hasattr(algorithm, "maximum_watermark_shape"):
        maximum = algorithm.maximum_watermark_shape(host)
        size = min(int(requested_size), int(maximum[0]), int(maximum[1]))
    else:
        size = min(int(requested_size), max(1, host.shape[0] // 8), max(1, host.shape[1] // 8))
    if size <= 0:
        raise ValueError("Host image is too small for watermark embedding.")
    return size, size


def list_images(directory: Path, limit: Optional[int] = None) -> List[Path]:
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return files if limit is None else files[: int(limit)]


def choose_watermarks_for_host(paths: List[Path], dataset: str, image_hash: str, count: int) -> List[Path]:
    count = min(max(1, int(count)), len(paths))
    rng = np.random.default_rng(deterministic_seed(f"wm-pair|{dataset}|{image_hash}"))
    return [paths[int(i)] for i in rng.choice(len(paths), size=count, replace=False)]


def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def local_distortion_metrics(original: np.ndarray, modified: np.ndarray, block_size: int) -> Dict[str, Optional[float]]:
    a = original.astype(np.float64)
    b = modified.astype(np.float64)
    error = np.abs(a - b)
    squared = (a - b) ** 2
    h, w = original.shape[:2]
    bs = max(4, int(block_size))
    block_mses = [
        float(np.mean(squared[y:min(y + bs, h), x:min(x + bs, w)]))
        for y in range(0, h, bs) for x in range(0, w, bs)
    ]
    worst = max(block_mses) if block_mses else 0.0
    worst_psnr = None if worst <= 0 else 10.0 * np.log10((255.0 ** 2) / worst)
    return {
        "clean_mae": safe_float(np.mean(error)),
        "clean_p95_abs_error": safe_float(np.percentile(error, 95)),
        "clean_max_abs_error": safe_float(np.max(error)),
        "clean_worst_block_mse": safe_float(worst),
        "clean_worst_block_psnr": safe_float(worst_psnr),
    }


RESULT_COLUMNS = [
    "experiment_key", "timestamp_utc", "status", "error_message",
    "dataset", "image_name", "image_path", "image_hash",
    "watermark_name", "watermark_path", "watermark_hash", "watermark_pair_index",
    "host_width", "host_height", "host_resolution", "host_aspect_ratio",
    "host_brightness", "host_contrast", "host_entropy", "host_edge_density",
    "host_texture_variance", "host_frequency_energy", "host_high_frequency_ratio",
    "host_dynamic_range", "host_noise_level", "host_sharpness",
    "host_gradient_complexity", "host_histogram_uniformity", "host_smooth_area_ratio",
    "host_image_complexity_score",
    "watermark_width", "watermark_height", "watermark_payload_bits",
    "watermark_density", "watermark_foreground_ratio", "watermark_entropy",
    "watermark_edge_complexity", "watermark_connected_components",
    "watermark_compactness", "watermark_fill_ratio", "watermark_symmetry_score",
    "watermark_stroke_complexity", "watermark_structural_complexity",
    "watermark_complexity_score",
    "method", "alpha", "requested_watermark_size", "actual_watermark_size",
    "attack_type", "attack_parameter", "attack_seed",
    "clean_mse", "clean_psnr", "clean_ssim", "clean_ber", "clean_correlation",
    "clean_mae", "clean_p95_abs_error", "clean_max_abs_error",
    "clean_worst_block_mse", "clean_worst_block_psnr",
    "attack_ber", "attack_correlation",
    "attacked_psnr_vs_host", "attacked_ssim_vs_host",
    "attacked_psnr_vs_watermarked", "attacked_ssim_vs_watermarked",
    "embedding_time_seconds", "clean_extraction_time_seconds",
    "attack_time_seconds", "attack_extraction_time_seconds", "total_time_seconds",
]


def feature_columns(host_features, watermark_features) -> Dict[str, object]:
    h = asdict(host_features)
    w = asdict(watermark_features)
    host_keys = [
        "width", "height", "resolution", "aspect_ratio", "brightness", "contrast",
        "entropy", "edge_density", "texture_variance", "frequency_energy",
        "high_frequency_ratio", "dynamic_range", "noise_level", "sharpness",
        "gradient_complexity", "histogram_uniformity", "smooth_area_ratio",
        "image_complexity_score",
    ]
    wm_keys = [
        "width", "height", "payload_bits", "density", "foreground_ratio", "entropy",
        "edge_complexity", "connected_components", "compactness", "fill_ratio",
        "symmetry_score", "stroke_complexity", "structural_complexity",
    ]
    row = {f"host_{k}": h[k] for k in host_keys}
    row.update({f"watermark_{k}": w[k] for k in wm_keys})
    row["watermark_complexity_score"] = w["watermark_complexity_score"]
    return row


def build_key(dataset, image_hash, watermark_hash, method, alpha, actual_shape, attack_type, attack_parameter) -> str:
    raw = "|".join([
        dataset, image_hash, watermark_hash, method, f"{float(alpha):.8f}",
        f"{actual_shape[0]}x{actual_shape[1]}", attack_type, str(attack_parameter),
    ])
    return stable_hash_text(raw)


# ============================================================
# WORKER: one host image -> all its rows
# ============================================================

_DONE_KEYS: Set[str] = set()


def _init_worker(done_keys: Set[str]) -> None:
    global _DONE_KEYS
    _DONE_KEYS = done_keys
    cv2.setNumThreads(1)


def process_host(dataset: str, image_path: str, watermark_paths: List[str]) -> Tuple[List[Dict], int, List[str]]:
    from attacks.pipeline import apply_attack_array
    from core.feature_extraction import extract_host_features, extract_watermark_features
    from watermarking.metrics import (
        calculate_ber, calculate_correlation, calculate_mse, calculate_psnr, calculate_ssim,
    )
    from watermarking.registry import create_watermarker

    rows: List[Dict] = []
    errors: List[str] = []
    skipped = 0
    path = Path(image_path)
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        host_features = extract_host_features(rgb)
        image_hash = stable_hash_bytes(path.read_bytes())
    except Exception:
        return rows, skipped, [f"[{utc_timestamp()}] HOST ERROR: {path}\n{traceback.format_exc()}"]

    wm_paths = [Path(p) for p in watermark_paths]
    selected = choose_watermarks_for_host(wm_paths, dataset, image_hash, WATERMARKS_PER_HOST)

    for pair_index, wm_path in enumerate(selected, start=1):
        try:
            wm_gray = np.asarray(Image.open(wm_path).convert("L"), dtype=np.uint8)
            wm_hash = stable_hash_bytes(wm_path.read_bytes())
        except Exception:
            errors.append(f"[{utc_timestamp()}] WATERMARK ERROR: {wm_path}\n{traceback.format_exc()}")
            continue

        for method in METHODS:
            for alpha in ALPHAS:
                try:
                    algorithm = create_watermarker(method, alpha=float(alpha))
                    shape = determine_watermark_shape(algorithm, gray, REQUESTED_WATERMARK_SIZE)
                    keys = [
                        build_key(dataset, image_hash, wm_hash, method, alpha, shape, a, p)
                        for a, p in ATTACKS
                    ]
                    if all(k in _DONE_KEYS for k in keys):
                        skipped += len(keys)
                        continue

                    prepared = resize_binary_watermark(wm_gray, shape)
                    common = feature_columns(host_features, extract_watermark_features(prepared))

                    t0 = time.perf_counter()
                    embedded = algorithm.embed(gray, prepared)
                    embedding_time = float(getattr(embedded, "runtime", time.perf_counter() - t0))
                    marked = np.asarray(embedded.watermarked_image, dtype=np.uint8)

                    clean_extraction = algorithm.extract(gray, marked, prepared.shape)
                    clean_extracted = normalize_binary_watermark(clean_extraction.extracted_watermark)
                    clean_metrics = {
                        "clean_mse": safe_float(calculate_mse(gray, marked)),
                        "clean_psnr": safe_float(calculate_psnr(gray, marked)),
                        "clean_ssim": safe_float(calculate_ssim(gray, marked)),
                        "clean_ber": safe_float(calculate_ber(prepared, clean_extracted)),
                        "clean_correlation": safe_float(calculate_correlation(prepared, clean_extracted)),
                        **local_distortion_metrics(gray, marked, LOCAL_BLOCK_SIZE),
                    }
                    clean_time = float(clean_extraction.runtime)
                except Exception as exc:
                    errors.append(
                        f"[{utc_timestamp()}] EMBED ERROR {dataset}/{path.name}/{wm_path.name}/"
                        f"{method}/a={alpha}: {exc}"
                    )
                    continue

                for key, (attack_type, attack_parameter) in zip(keys, ATTACKS):
                    if key in _DONE_KEYS:
                        skipped += 1
                        continue
                    seed = deterministic_seed(key)
                    started = time.perf_counter()
                    base = {
                        "experiment_key": key, "timestamp_utc": utc_timestamp(),
                        "dataset": dataset, "image_name": path.name, "image_path": str(path),
                        "image_hash": image_hash, "watermark_name": wm_path.name,
                        "watermark_path": str(wm_path), "watermark_hash": wm_hash,
                        "watermark_pair_index": pair_index, "method": method, "alpha": float(alpha),
                        "requested_watermark_size": int(REQUESTED_WATERMARK_SIZE),
                        "actual_watermark_size": f"{prepared.shape[1]}x{prepared.shape[0]}",
                        "attack_type": attack_type, "attack_parameter": attack_parameter,
                        "attack_seed": seed, "embedding_time_seconds": embedding_time,
                        "clean_extraction_time_seconds": clean_time,
                        **common, **clean_metrics,
                    }
                    try:
                        if attack_type == "none":
                            attacked = marked.copy()
                            attack_time = 0.0
                            attacked_extracted = clean_extracted
                            attack_extraction_time = clean_time
                        else:
                            attacked = apply_attack_array(marked, attack_type, attack_parameter, seed=seed)
                            attack_time = time.perf_counter() - started
                            extraction = algorithm.extract(gray, attacked, prepared.shape)
                            attacked_extracted = normalize_binary_watermark(extraction.extracted_watermark)
                            attack_extraction_time = float(extraction.runtime)
                        base.update({
                            "status": "SUCCESS", "error_message": "",
                            "attack_ber": safe_float(calculate_ber(prepared, attacked_extracted)),
                            "attack_correlation": safe_float(calculate_correlation(prepared, attacked_extracted)),
                            "attacked_psnr_vs_host": safe_float(calculate_psnr(gray, attacked)),
                            "attacked_ssim_vs_host": safe_float(calculate_ssim(gray, attacked)),
                            "attacked_psnr_vs_watermarked": safe_float(calculate_psnr(marked, attacked)),
                            "attacked_ssim_vs_watermarked": safe_float(calculate_ssim(marked, attacked)),
                            "attack_time_seconds": float(attack_time),
                            "attack_extraction_time_seconds": float(attack_extraction_time),
                            "total_time_seconds": float(embedding_time + clean_time + attack_time + attack_extraction_time),
                        })
                    except Exception as exc:
                        base.update({"status": "FAILED", "error_message": str(exc),
                                     "attack_time_seconds": float(time.perf_counter() - started)})
                        errors.append(
                            f"[{utc_timestamp()}] ATTACK ERROR {dataset}/{path.name}/{method}/a={alpha}/"
                            f"{attack_type}={attack_parameter}: {exc}"
                        )
                    rows.append(base)
    return rows, skipped, errors


# ============================================================
# DATABASE
# ============================================================

def connect_database() -> sqlite3.Connection:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA journal_mode=WAL;")
    defs = ["experiment_key TEXT PRIMARY KEY" if c == "experiment_key" else f'"{c}"' for c in RESULT_COLUMNS]
    connection.execute(f"CREATE TABLE IF NOT EXISTS results ({', '.join(defs)})")
    connection.commit()
    return connection


def save_rows(connection: sqlite3.Connection, rows: List[Dict]) -> None:
    placeholders = ",".join("?" for _ in RESULT_COLUMNS)
    columns = ",".join(f'"{c}"' for c in RESULT_COLUMNS)
    values = []
    for row in rows:
        record = []
        for column in RESULT_COLUMNS:
            value = row.get(column)
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = safe_float(value)
            record.append(value)
        values.append(record)
    with connection:
        connection.executemany(f"INSERT OR REPLACE INTO results ({columns}) VALUES ({placeholders})", values)


def export_csv(connection: sqlite3.Connection) -> None:
    columns_sql = ",".join(f'"{c}"' for c in RESULT_COLUMNS)
    cursor = connection.execute(
        f"SELECT {columns_sql} FROM results "
        "ORDER BY dataset, image_name, watermark_name, method, alpha, attack_type, attack_parameter"
    )
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_COLUMNS)
        while True:
            batch = cursor.fetchmany(20000)
            if not batch:
                break
            writer.writerows(batch)


def log_errors(messages: List[str]) -> None:
    if not messages:
        return
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
        for message in messages:
            handle.write(message.rstrip() + "\n\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    for label, folder in (("MS_COCO_DIR", MS_COCO_DIR), ("OXFORD_PET_DIR", OXFORD_PET_DIR), ("WATERMARK_DIR", WATERMARK_DIR)):
        if not folder.is_dir():
            raise FileNotFoundError(f"{label} is not a folder: {folder}")

    watermark_paths = [str(p) for p in list_images(WATERMARK_DIR)]
    connection = connect_database()
    done = {row[0] for row in connection.execute("SELECT experiment_key FROM results WHERE status='SUCCESS'")}

    tasks = []
    for dataset, folder in (("MS_COCO", MS_COCO_DIR), ("OXFORD_PET", OXFORD_PET_DIR)):
        for image in list_images(folder, MAX_IMAGES_PER_DATASET):
            tasks.append((dataset, str(image), watermark_paths))

    # Optional slice of the host list, so a notebook can run the work in
    # chunks and save a checkpoint after each one.
    total_hosts = len(tasks)
    start = int(os.environ.get("AWSRE_HOST_START", "0"))
    end = int(os.environ.get("AWSRE_HOST_END", str(total_hosts)))
    tasks = tasks[start:end]
    export = os.environ.get("AWSRE_EXPORT_CSV", "1") == "1"

    print("=" * 78)
    print("AWSRE 5(+1)-METHOD DATASET RUN (resumable)")
    print(f"Host slice: {start}..{min(end, total_hosts)} of {total_hosts}")
    print(f"Methods: {METHODS}")
    print(f"Alphas: {ALPHAS} | attack conditions: {len(ATTACKS)}")
    print(f"Hosts: {len(tasks)} | watermarks per host: {WATERMARKS_PER_HOST} | workers: {WORKERS}")
    print(f"Rows already done: {len(done)}")
    print(f"Database: {DATABASE_PATH}")
    print("=" * 78, flush=True)

    counter = {"saved": 0, "skipped": 0, "errors": 0}
    started = time.time()
    try:
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(done,)) as pool:
            futures = [pool.submit(process_host, *task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                rows, skipped, errors = future.result()
                save_rows(connection, rows)
                log_errors(errors)
                counter["saved"] += len(rows)
                counter["skipped"] += skipped
                counter["errors"] += len(errors)
                elapsed = time.time() - started
                eta = elapsed / index * (len(tasks) - index)
                print(f"host {index}/{len(tasks)} | +{len(rows)} rows | skipped {skipped} | "
                      f"errors {len(errors)} | elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)
    finally:
        if export:
            export_csv(connection)
        connection.close()

    print("\nRUN FINISHED")
    print(f"New rows: {counter['saved']} | skipped (already done): {counter['skipped']} | errors: {counter['errors']}")
    if export:
        print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
