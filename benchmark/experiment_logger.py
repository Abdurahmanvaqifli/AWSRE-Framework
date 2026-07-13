"""
AWSRE Safe Experiment Logger

Safely stores benchmark experiments as per-experiment JSON files,
a continuously updated CSV index, an error log, checkpoints and backups.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd


LOGGER_VERSION = "1.0.1"

DEFAULT_EXPERIMENT_COLUMNS = [
    "experiment_id", "experiment_key", "timestamp_utc",
    "benchmark_version", "benchmark_tag",
    "host_image_id", "host_image_name", "host_image_hash",
    "host_dataset", "host_width", "host_height",
    "watermark_id", "watermark_type", "watermark_hash",
    "watermark_width", "watermark_height",
    "watermark_payload_bits", "watermark_density",
    "method", "alpha", "watermark_size", "channel",
    "block_size", "decomposition_level", "subband",
    "attack_type", "attack_parameter", "attack_steps",
    "mse", "psnr", "ssim", "ber", "correlation",
    "extracted_successfully", "detection_result",
    "embedding_time_seconds", "attack_time_seconds",
    "extraction_time_seconds", "total_time_seconds",
    "predicted_psnr", "predicted_ssim", "predicted_ber",
    "predicted_correlation", "prediction_confidence",
    "prediction_source", "experiment_json_path",
    "experiment_directory", "status", "error_message", "notes",
]

DEFAULT_ERROR_COLUMNS = [
    "timestamp_utc", "experiment_id", "experiment_key",
    "host_image_name", "watermark_type", "method", "alpha",
    "attack_type", "attack_parameter", "error_type",
    "error_message", "traceback",
]

VALID_STATUSES = {"PENDING", "RUNNING", "SUCCESS", "FAILED", "SKIPPED"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_not_none(*values: Any, default: Any = None) -> Any:
    """Return the first non-None value, preserving 0, 0.0 and False."""
    for value in values:
        if value is not None:
            return value
    return default


def make_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return make_json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def nested_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def atomic_write_text(
    output_path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            delete=False,
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
        ) as temporary_file:
            temporary_name = temporary_file.name
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_name, destination)
        return destination
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_json(
    output_path: str | Path,
    data: Dict[str, Any],
) -> Path:
    return atomic_write_text(
        output_path,
        json.dumps(
            make_json_safe(data),
            indent=4,
            ensure_ascii=False,
        ),
    )


def ensure_csv_header(
    csv_path: str | Path,
    columns: Iterable[str],
) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(columns),
                extrasaction="ignore",
            )
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

    return path


def append_csv_row(
    csv_path: str | Path,
    row: Dict[str, Any],
    columns: Iterable[str],
) -> None:
    columns_list = list(columns)
    path = ensure_csv_header(csv_path, columns_list)

    normalized = {
        column: make_json_safe(row.get(column))
        for column in columns_list
    }

    for key, value in list(normalized.items()):
        if isinstance(value, (list, dict)):
            normalized[key] = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns_list,
            extrasaction="ignore",
        )
        writer.writerow(normalized)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class ExperimentLoggerConfig:
    output_root: str | Path
    experiments_directory_name: str = "experiments"
    logs_directory_name: str = "logs"
    backups_directory_name: str = "backups"
    checkpoints_directory_name: str = "checkpoints"
    csv_file_name: str = "awsre_experiments.csv"
    errors_file_name: str = "awsre_errors.csv"
    checkpoint_file_name: str = "benchmark_checkpoint.json"
    backup_interval: int = 10
    checkpoint_interval: int = 10
    save_json_per_experiment: bool = True
    resume_enabled: bool = True
    experiment_columns: List[str] = field(
        default_factory=lambda: list(DEFAULT_EXPERIMENT_COLUMNS)
    )
    error_columns: List[str] = field(
        default_factory=lambda: list(DEFAULT_ERROR_COLUMNS)
    )

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.backup_interval = max(1, int(self.backup_interval))
        self.checkpoint_interval = max(1, int(self.checkpoint_interval))

    @property
    def experiments_path(self) -> Path:
        return self.output_root / self.experiments_directory_name

    @property
    def logs_path(self) -> Path:
        return self.output_root / self.logs_directory_name

    @property
    def backups_path(self) -> Path:
        return self.output_root / self.backups_directory_name

    @property
    def checkpoints_path(self) -> Path:
        return self.output_root / self.checkpoints_directory_name

    @property
    def csv_path(self) -> Path:
        return self.output_root / self.csv_file_name

    @property
    def errors_path(self) -> Path:
        return self.logs_path / self.errors_file_name

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoints_path / self.checkpoint_file_name

    @property
    def latest_backup_path(self) -> Path:
        return self.backups_path / "awsre_experiments_latest.csv"

    def create_directories(self) -> None:
        for path in (
            self.output_root,
            self.experiments_path,
            self.logs_path,
            self.backups_path,
            self.checkpoints_path,
        ):
            path.mkdir(parents=True, exist_ok=True)


def flatten_experiment_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested benchmark record into one CSV-compatible row."""
    safe = make_json_safe(record)
    if not isinstance(safe, dict):
        raise TypeError("Experiment record must serialize to a dictionary.")

    host = safe.get("host", {}) or {}
    watermark = safe.get("watermark", {}) or {}
    embedding = safe.get("embedding", {}) or {}
    attack = safe.get("attack", {}) or {}
    quality = safe.get("imperceptibility", {}) or {}
    robustness = safe.get("robustness", {}) or {}
    runtime = safe.get("runtime", {}) or {}
    prediction = safe.get("prediction", {}) or {}

    wm_width = first_not_none(
        watermark.get("width"),
        embedding.get("watermark_width"),
    )
    wm_height = first_not_none(
        watermark.get("height"),
        embedding.get("watermark_height"),
    )
    wm_size = (
        f"{wm_width}x{wm_height}"
        if wm_width is not None and wm_height is not None
        else safe.get("watermark_size")
    )

    embedding_time = first_not_none(
        runtime.get("embedding_time"),
        runtime.get("embedding_time_seconds"),
        default=0.0,
    )
    attack_time = first_not_none(
        runtime.get("attack_time"),
        runtime.get("attack_time_seconds"),
        default=0.0,
    )
    extraction_time = first_not_none(
        runtime.get("extraction_time"),
        runtime.get("extraction_time_seconds"),
        default=0.0,
    )
    total_time = first_not_none(
        runtime.get("total_runtime"),
        runtime.get("total_time_seconds"),
        default=(
            float(embedding_time)
            + float(attack_time)
            + float(extraction_time)
        ),
    )

    row = {
        "experiment_id": safe.get("experiment_id"),
        "experiment_key": safe.get("experiment_key"),
        "timestamp_utc": safe.get("timestamp_utc", utc_now_iso()),
        "benchmark_version": safe.get("benchmark_version"),
        "benchmark_tag": safe.get("benchmark_tag"),

        "host_image_id": first_not_none(
            host.get("image_id"), safe.get("host_image_id")
        ),
        "host_image_name": first_not_none(
            host.get("file_name"), safe.get("host_image_name")
        ),
        "host_image_hash": first_not_none(
            host.get("file_hash"), safe.get("host_image_hash")
        ),
        "host_dataset": first_not_none(
            host.get("benchmark_tag"),
            host.get("dataset_name"),
            safe.get("host_dataset"),
        ),
        "host_width": first_not_none(
            host.get("width"),
            host.get("processed_width"),
            safe.get("host_width"),
        ),
        "host_height": first_not_none(
            host.get("height"),
            host.get("processed_height"),
            safe.get("host_height"),
        ),

        "watermark_id": first_not_none(
            watermark.get("watermark_id"), safe.get("watermark_id")
        ),
        "watermark_type": first_not_none(
            watermark.get("watermark_type"), safe.get("watermark_type")
        ),
        "watermark_hash": first_not_none(
            watermark.get("file_hash"), safe.get("watermark_hash")
        ),
        "watermark_width": wm_width,
        "watermark_height": wm_height,
        "watermark_payload_bits": first_not_none(
            watermark.get("payload_bits"),
            safe.get("watermark_payload_bits"),
        ),
        "watermark_density": first_not_none(
            watermark.get("density"),
            safe.get("watermark_density"),
        ),

        "method": first_not_none(
            embedding.get("method"), safe.get("method")
        ),
        "alpha": first_not_none(
            embedding.get("alpha"), safe.get("alpha")
        ),
        "watermark_size": wm_size,
        "channel": first_not_none(
            embedding.get("channel"), safe.get("channel")
        ),
        "block_size": first_not_none(
            embedding.get("block_size"), safe.get("block_size")
        ),
        "decomposition_level": first_not_none(
            embedding.get("decomposition_level"),
            safe.get("decomposition_level"),
        ),
        "subband": first_not_none(
            embedding.get("subband"), safe.get("subband")
        ),

        "attack_type": first_not_none(
            attack.get("attack_type"),
            safe.get("attack_type"),
            default="none",
        ),
        "attack_parameter": (
            attack.get("parameter")
            if "parameter" in attack
            else safe.get("attack_parameter")
        ),
        "attack_steps": first_not_none(
            attack.get("steps"),
            safe.get("attack_steps"),
            default=[],
        ),

        # first_not_none is essential here: valid zeros must not become NaN.
        "mse": first_not_none(quality.get("mse"), safe.get("mse")),
        "psnr": first_not_none(quality.get("psnr"), safe.get("psnr")),
        "ssim": first_not_none(quality.get("ssim"), safe.get("ssim")),
        "ber": first_not_none(
            robustness.get("ber"), safe.get("ber")
        ),
        "correlation": first_not_none(
            robustness.get("correlation"),
            safe.get("correlation"),
        ),
        "extracted_successfully": (
            robustness.get("extracted_successfully")
            if "extracted_successfully" in robustness
            else safe.get("extracted_successfully")
        ),
        "detection_result": safe.get("detection_result"),

        "embedding_time_seconds": embedding_time,
        "attack_time_seconds": attack_time,
        "extraction_time_seconds": extraction_time,
        "total_time_seconds": total_time,

        "predicted_psnr": prediction.get("expected_psnr"),
        "predicted_ssim": prediction.get("expected_ssim"),
        "predicted_ber": prediction.get("expected_ber"),
        "predicted_correlation": prediction.get("expected_correlation"),
        "prediction_confidence": prediction.get("confidence"),
        "prediction_source": prediction.get("prediction_source"),

        "experiment_json_path": safe.get("experiment_json_path"),
        "experiment_directory": safe.get("experiment_directory"),
        "status": safe.get("status", "SUCCESS"),
        "error_message": safe.get("error_message", ""),
        "notes": safe.get("notes", ""),
    }

    # Explicit flat fields override nested fields, including zero values.
    for column in DEFAULT_EXPERIMENT_COLUMNS:
        if column in safe and safe[column] is not None:
            row[column] = safe[column]

    return make_json_safe(row)


class ExperimentLogger:
    """Safe persistent logger with resume, checkpoint and backup support."""

    def __init__(self, config: ExperimentLoggerConfig) -> None:
        if not isinstance(config, ExperimentLoggerConfig):
            raise TypeError("config must be ExperimentLoggerConfig.")

        self.config = config
        self.config.create_directories()
        self._lock = threading.RLock()

        ensure_csv_header(
            self.config.csv_path,
            self.config.experiment_columns,
        )
        ensure_csv_header(
            self.config.errors_path,
            self.config.error_columns,
        )

        self._completed_keys: Set[str] = (
            self._load_completed_keys()
            if self.config.resume_enabled
            else set()
        )
        self.successful_writes = 0
        self.failed_writes = 0

    def _load_completed_keys(self) -> Set[str]:
        path = self.config.csv_path
        if not path.exists() or path.stat().st_size == 0:
            return set()

        try:
            frame = pd.read_csv(
                path,
                usecols=["experiment_key", "status"],
            )
        except (ValueError, pd.errors.EmptyDataError):
            return set()

        if frame.empty:
            return set()

        keys = frame["experiment_key"].fillna("").astype(str)
        statuses = frame["status"].fillna("").astype(str).str.upper()

        return {
            key
            for key, status in zip(keys, statuses)
            if key and status == "SUCCESS"
        }

    def already_completed(self, experiment_key: str) -> bool:
        return str(experiment_key) in self._completed_keys

    @property
    def completed_count(self) -> int:
        return len(self._completed_keys)

    def experiment_directory(self, experiment_id: str) -> Path:
        safe_id = (
            str(experiment_id)
            .strip()
            .replace("/", "_")
            .replace("\\", "_")
        )
        if not safe_id:
            raise ValueError("experiment_id cannot be empty.")
        return self.config.experiments_path / safe_id

    def experiment_json_path(self, experiment_id: str) -> Path:
        return self.experiment_directory(experiment_id) / "metadata.json"

    def log_experiment(
        self,
        record: Any,
        *,
        allow_duplicate: bool = False,
    ) -> Dict[str, Any]:
        with self._lock:
            raw = make_json_safe(record)
            if not isinstance(raw, dict):
                raise TypeError(
                    "Experiment record must serialize to a dictionary."
                )

            experiment_id = raw.get("experiment_id")
            experiment_key = raw.get("experiment_key")

            if not experiment_id:
                raise ValueError(
                    "Experiment record requires experiment_id."
                )
            if not experiment_key:
                raise ValueError(
                    "Experiment record requires experiment_key."
                )
            if (
                not allow_duplicate
                and self.already_completed(experiment_key)
            ):
                raise ValueError(
                    f"Experiment has already been completed: {experiment_key}"
                )

            status = str(raw.get("status", "SUCCESS")).upper()
            if status not in VALID_STATUSES:
                raise ValueError(
                    f"Unsupported experiment status: {status}."
                )

            raw["status"] = status
            raw.setdefault("timestamp_utc", utc_now_iso())

            directory = self.experiment_directory(experiment_id)
            directory.mkdir(parents=True, exist_ok=True)
            json_path = self.experiment_json_path(experiment_id)

            raw["experiment_directory"] = str(directory)
            raw["experiment_json_path"] = str(json_path)

            try:
                if self.config.save_json_per_experiment:
                    atomic_write_json(json_path, raw)
                    self._verify_json(
                        json_path,
                        expected_experiment_id=experiment_id,
                    )

                flattened = flatten_experiment_record(raw)
                append_csv_row(
                    self.config.csv_path,
                    flattened,
                    self.config.experiment_columns,
                )
                self._verify_csv_record(experiment_id)

                if status == "SUCCESS":
                    self._completed_keys.add(str(experiment_key))

                self.successful_writes += 1

                if (
                    self.successful_writes
                    % self.config.checkpoint_interval
                    == 0
                ):
                    self.save_checkpoint()

                if (
                    self.successful_writes
                    % self.config.backup_interval
                    == 0
                ):
                    self.create_backup()

                return flattened
            except Exception:
                self.failed_writes += 1
                raise

    def _verify_json(
        self,
        path: Path,
        *,
        expected_experiment_id: str,
    ) -> None:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(
                "Experiment JSON was not written correctly."
            )

        with path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)

        if str(saved.get("experiment_id")) != str(
            expected_experiment_id
        ):
            raise RuntimeError(
                "Saved experiment JSON failed ID verification."
            )

    def _verify_csv_record(self, experiment_id: str) -> None:
        frame = pd.read_csv(self.config.csv_path)
        if frame.empty:
            raise RuntimeError(
                "CSV is empty after experiment write."
            )

        ids = frame["experiment_id"].fillna("").astype(str)
        if str(experiment_id) not in set(ids):
            raise RuntimeError(
                "Experiment was not found after CSV write."
            )

    def log_error(self, error_record: Dict[str, Any]) -> None:
        row = {
            column: error_record.get(column)
            for column in self.config.error_columns
        }
        row["timestamp_utc"] = first_not_none(
            row.get("timestamp_utc"),
            utc_now_iso(),
        )

        with self._lock:
            append_csv_row(
                self.config.errors_path,
                row,
                self.config.error_columns,
            )

    def save_checkpoint(
        self,
        *,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        checkpoint = {
            "logger_version": LOGGER_VERSION,
            "timestamp_utc": utc_now_iso(),
            "completed_count": self.completed_count,
            "successful_writes": self.successful_writes,
            "failed_writes": self.failed_writes,
            "csv_path": str(self.config.csv_path),
            "errors_path": str(self.config.errors_path),
            "completed_experiment_keys": sorted(
                self._completed_keys
            ),
            "extra_data": make_json_safe(extra_data or {}),
        }
        return atomic_write_json(
            self.config.checkpoint_path,
            checkpoint,
        )

    def load_checkpoint(self) -> Dict[str, Any]:
        path = self.config.checkpoint_path
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)

        self._completed_keys.update(
            str(key)
            for key in checkpoint.get(
                "completed_experiment_keys",
                [],
            )
            if key
        )
        return checkpoint

    def create_backup(self) -> Path:
        source = self.config.csv_path
        if not source.exists() or source.stat().st_size == 0:
            raise RuntimeError(
                "Cannot back up a missing or empty CSV."
            )

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S"
        )
        timestamped = (
            self.config.backups_path
            / f"awsre_experiments_{timestamp}.csv"
        )

        shutil.copy2(source, timestamped)
        shutil.copy2(source, self.config.latest_backup_path)

        if timestamped.stat().st_size != source.stat().st_size:
            raise RuntimeError(
                "Timestamped CSV backup verification failed."
            )
        if (
            self.config.latest_backup_path.stat().st_size
            != source.stat().st_size
        ):
            raise RuntimeError(
                "Latest CSV backup verification failed."
            )

        pd.read_csv(timestamped)
        return timestamped

    def validate_storage(
        self,
        *,
        minimum_rows: int = 0,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "csv_exists": self.config.csv_path.exists(),
            "errors_csv_exists": self.config.errors_path.exists(),
            "csv_size_bytes": 0,
            "csv_rows": 0,
            "csv_columns": 0,
            "missing_columns": [],
            "json_files": 0,
            "valid": False,
        }

        if not report["csv_exists"]:
            return report

        report["csv_size_bytes"] = (
            self.config.csv_path.stat().st_size
        )

        try:
            frame = pd.read_csv(self.config.csv_path)
            report["csv_rows"] = len(frame)
            report["csv_columns"] = len(frame.columns)
            report["missing_columns"] = [
                column
                for column in self.config.experiment_columns
                if column not in frame.columns
            ]
            report["json_files"] = len(
                list(
                    self.config.experiments_path.glob(
                        "*/metadata.json"
                    )
                )
            )
            report["valid"] = (
                report["csv_size_bytes"] > 0
                and report["csv_rows"] >= minimum_rows
                and not report["missing_columns"]
            )
        except Exception as exc:
            report["validation_error"] = str(exc)

        return report

    def dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.config.csv_path)

    def summary(self) -> Dict[str, Any]:
        frame = self.dataframe()

        if frame.empty:
            return {
                "total_rows": 0,
                "successful_rows": 0,
                "failed_rows": 0,
                "skipped_rows": 0,
                "completed_keys": self.completed_count,
            }

        status = (
            frame["status"]
            .fillna("")
            .astype(str)
            .str.upper()
        )

        return {
            "total_rows": int(len(frame)),
            "successful_rows": int(
                np.count_nonzero(status == "SUCCESS")
            ),
            "failed_rows": int(
                np.count_nonzero(status == "FAILED")
            ),
            "skipped_rows": int(
                np.count_nonzero(status == "SKIPPED")
            ),
            "completed_keys": self.completed_count,
            "csv_path": str(self.config.csv_path),
            "errors_path": str(self.config.errors_path),
        }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary_directory:
        config = ExperimentLoggerConfig(
            output_root=temporary_directory,
            backup_interval=2,
            checkpoint_interval=2,
        )
        logger = ExperimentLogger(config)

        for index in range(3):
            logger.log_experiment({
                "experiment_id": f"EXP-TEST-{index + 1:04d}",
                "experiment_key": (
                    f"HOST-1|WM-1|DCT|{10 + index}|none"
                ),
                "benchmark_version": "LOGGER-TEST",
                "benchmark_tag": "SELF_TEST",
                "host": {
                    "image_id": "HOST-1",
                    "file_name": "host.png",
                    "file_hash": "host_hash",
                    "width": 512,
                    "height": 512,
                    "benchmark_tag": "Synthetic",
                },
                "watermark": {
                    "watermark_id": "WM-1",
                    "watermark_type": "Binary Pattern",
                    "file_hash": "wm_hash",
                    "width": 32,
                    "height": 32,
                    "payload_bits": 1024,
                    "density": 0.5,
                },
                "embedding": {
                    "method": "DCT",
                    "alpha": 10 + index,
                    "watermark_width": 32,
                    "watermark_height": 32,
                    "channel": "Grayscale",
                    "block_size": 8,
                    "decomposition_level": 1,
                    "subband": None,
                },
                "attack": {
                    "attack_type": "none",
                    "parameter": None,
                },
                "imperceptibility": {
                    "mse": 0.0,
                    "psnr": 50.0,
                    "ssim": 1.0,
                },
                "robustness": {
                    "ber": 0.0,
                    "correlation": 1.0,
                    "extracted_successfully": True,
                },
                "runtime": {
                    "embedding_time": 0.01,
                    "attack_time": 0.0,
                    "extraction_time": 0.01,
                },
                "status": "SUCCESS",
            })

        validation = logger.validate_storage(minimum_rows=3)
        frame = logger.dataframe()

        assert validation["valid"]
        assert validation["csv_rows"] == 3
        assert validation["json_files"] == 3
        assert frame["ber"].notna().all()
        assert (frame["ber"] == 0.0).all()
        assert frame["mse"].notna().all()
        assert (frame["mse"] == 0.0).all()
        assert frame["attack_time_seconds"].notna().all()
        assert (frame["attack_time_seconds"] == 0.0).all()

        print("=" * 72)
        print("AWSRE EXPERIMENT LOGGER SELF TEST")
        print("=" * 72)
        print("Validation:", validation)
        print("Summary:", logger.summary())
        print("\n✅ Experiment logger self test passed.")
