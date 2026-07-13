"""
AWSRE Safe Experiment Logger

Stores every benchmark experiment immediately and safely.

Storage model:
- one JSON file per experiment;
- one continuously updated CSV index;
- separate CSV error log;
- checkpoint JSON;
- verified backups;
- resume and duplicate-detection support.

The logger does not perform watermark embedding, attacks,
extraction, or metric calculation.
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


# ============================================================
# CONSTANTS
# ============================================================

LOGGER_VERSION = "1.0.0"

DEFAULT_EXPERIMENT_COLUMNS = [
    # Identity
    "experiment_id",
    "experiment_key",
    "timestamp_utc",
    "benchmark_version",
    "benchmark_tag",

    # Host image
    "host_image_id",
    "host_image_name",
    "host_image_hash",
    "host_dataset",
    "host_width",
    "host_height",

    # Watermark
    "watermark_id",
    "watermark_type",
    "watermark_hash",
    "watermark_width",
    "watermark_height",
    "watermark_payload_bits",
    "watermark_density",

    # Embedding
    "method",
    "alpha",
    "watermark_size",
    "channel",
    "block_size",
    "decomposition_level",
    "subband",

    # Attack
    "attack_type",
    "attack_parameter",
    "attack_steps",

    # Imperceptibility
    "mse",
    "psnr",
    "ssim",

    # Robustness
    "ber",
    "correlation",
    "extracted_successfully",
    "detection_result",

    # Runtime
    "embedding_time_seconds",
    "attack_time_seconds",
    "extraction_time_seconds",
    "total_time_seconds",

    # Prediction
    "predicted_psnr",
    "predicted_ssim",
    "predicted_ber",
    "predicted_correlation",
    "prediction_confidence",
    "prediction_source",

    # Files
    "experiment_json_path",
    "experiment_directory",

    # Status
    "status",
    "error_message",
    "notes",
]

DEFAULT_ERROR_COLUMNS = [
    "timestamp_utc",
    "experiment_id",
    "experiment_key",
    "host_image_name",
    "watermark_type",
    "method",
    "alpha",
    "attack_type",
    "attack_parameter",
    "error_type",
    "error_message",
    "traceback",
]

VALID_STATUSES = {
    "PENDING",
    "RUNNING",
    "SUCCESS",
    "FAILED",
    "SKIPPED",
}


# ============================================================
# HELPERS
# ============================================================

def utc_now_iso() -> str:
    """
    Return current UTC time in ISO-8601 format.
    """
    return datetime.now(
        timezone.utc
    ).isoformat()


def make_json_safe(value: Any) -> Any:
    """
    Recursively convert common scientific-Python values
    into JSON-serializable Python objects.
    """
    if value is None:
        return None

    if is_dataclass(value):
        return make_json_safe(
            asdict(value)
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric = float(value)

        if not np.isfinite(numeric):
            return None

        return numeric

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

        return value

    if isinstance(
        value,
        (str, int, bool),
    ):
        return value

    return str(value)


def nested_get(
    data: Dict[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    """
    Read a nested dictionary value using dot notation.

    Example:
        nested_get(data, "host.file_name")
    """
    current: Any = data

    for part in path.split("."):
        if not isinstance(
            current,
            dict,
        ):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def atomic_write_text(
    output_path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """
    Write text atomically by replacing the target only after
    the temporary file has been written successfully.
    """
    destination = Path(
        output_path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file: Optional[
        tempfile.NamedTemporaryFile
    ] = None

    try:
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            delete=False,
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )

        with temporary_file:
            temporary_file.write(
                content
            )

            temporary_file.flush()
            os.fsync(
                temporary_file.fileno()
            )

        os.replace(
            temporary_file.name,
            destination,
        )

    except Exception:
        if temporary_file is not None:
            temporary_path = Path(
                temporary_file.name
            )

            if temporary_path.exists():
                temporary_path.unlink(
                    missing_ok=True
                )

        raise

    return destination


def atomic_write_json(
    output_path: str | Path,
    data: Dict[str, Any],
) -> Path:
    """
    Save JSON atomically.
    """
    json_text = json.dumps(
        make_json_safe(data),
        indent=4,
        ensure_ascii=False,
        sort_keys=False,
    )

    return atomic_write_text(
        output_path,
        json_text,
        encoding="utf-8",
    )


def ensure_csv_header(
    csv_path: str | Path,
    columns: Iterable[str],
) -> Path:
    """
    Create a CSV file containing only its header when missing.
    """
    path = Path(
        csv_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        with path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=list(columns),
                extrasaction="ignore",
            )

            writer.writeheader()
            file_handle.flush()
            os.fsync(
                file_handle.fileno()
            )

    return path


def append_csv_row(
    csv_path: str | Path,
    row: Dict[str, Any],
    columns: Iterable[str],
) -> None:
    """
    Append one row and force the write to disk.
    """
    path = ensure_csv_header(
        csv_path,
        columns,
    )

    normalized_columns = list(
        columns
    )

    normalized_row = {
        column: make_json_safe(
            row.get(column)
        )
        for column in normalized_columns
    }

    for key, value in list(
        normalized_row.items()
    ):
        if isinstance(
            value,
            (list, dict),
        ):
            normalized_row[key] = (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

    with path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=normalized_columns,
            extrasaction="ignore",
        )

        writer.writerow(
            normalized_row
        )

        file_handle.flush()
        os.fsync(
            file_handle.fileno()
        )


# ============================================================
# LOGGER CONFIGURATION
# ============================================================

@dataclass
class ExperimentLoggerConfig:
    """
    Paths and safety options for the experiment logger.
    """

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
        default_factory=lambda: list(
            DEFAULT_EXPERIMENT_COLUMNS
        )
    )

    error_columns: List[str] = field(
        default_factory=lambda: list(
            DEFAULT_ERROR_COLUMNS
        )
    )

    def __post_init__(self) -> None:
        self.output_root = Path(
            self.output_root
        )

        self.backup_interval = max(
            1,
            int(self.backup_interval),
        )

        self.checkpoint_interval = max(
            1,
            int(self.checkpoint_interval),
        )

    @property
    def experiments_path(self) -> Path:
        return (
            self.output_root
            / self.experiments_directory_name
        )

    @property
    def logs_path(self) -> Path:
        return (
            self.output_root
            / self.logs_directory_name
        )

    @property
    def backups_path(self) -> Path:
        return (
            self.output_root
            / self.backups_directory_name
        )

    @property
    def checkpoints_path(self) -> Path:
        return (
            self.output_root
            / self.checkpoints_directory_name
        )

    @property
    def csv_path(self) -> Path:
        return (
            self.output_root
            / self.csv_file_name
        )

    @property
    def errors_path(self) -> Path:
        return (
            self.logs_path
            / self.errors_file_name
        )

    @property
    def checkpoint_path(self) -> Path:
        return (
            self.checkpoints_path
            / self.checkpoint_file_name
        )

    @property
    def latest_backup_path(self) -> Path:
        return (
            self.backups_path
            / "awsre_experiments_latest.csv"
        )

    def create_directories(self) -> None:
        for directory in (
            self.output_root,
            self.experiments_path,
            self.logs_path,
            self.backups_path,
            self.checkpoints_path,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================
# RECORD FLATTENING
# ============================================================

def flatten_experiment_record(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Flatten an ExperimentRecord-style nested dictionary into
    one CSV row.

    The function also accepts already-flat dictionaries.
    """
    safe_record = make_json_safe(
        record
    )

    if not isinstance(
        safe_record,
        dict,
    ):
        raise TypeError(
            "Experiment record must serialize to a dictionary."
        )

    host = safe_record.get(
        "host",
        {},
    ) or {}

    watermark = safe_record.get(
        "watermark",
        {},
    ) or {}

    embedding = safe_record.get(
        "embedding",
        {},
    ) or {}

    attack = safe_record.get(
        "attack",
        {},
    ) or {}

    imperceptibility = safe_record.get(
        "imperceptibility",
        {},
    ) or {}

    robustness = safe_record.get(
        "robustness",
        {},
    ) or {}

    runtime = safe_record.get(
        "runtime",
        {},
    ) or {}

    prediction = safe_record.get(
        "prediction",
        {},
    ) or {}

    watermark_width = (
        watermark.get("width")
        or embedding.get(
            "watermark_width"
        )
    )

    watermark_height = (
        watermark.get("height")
        or embedding.get(
            "watermark_height"
        )
    )

    if (
        watermark_width is not None
        and watermark_height is not None
    ):
        watermark_size = (
            f"{watermark_width}"
            f"x"
            f"{watermark_height}"
        )
    else:
        watermark_size = safe_record.get(
            "watermark_size"
        )

    attack_steps = (
        attack.get("steps")
        or safe_record.get(
            "attack_steps"
        )
        or []
    )

    embedding_time = (
        runtime.get(
            "embedding_time"
        )
        or runtime.get(
            "embedding_time_seconds"
        )
        or 0.0
    )

    attack_time = (
        runtime.get(
            "attack_time"
        )
        or runtime.get(
            "attack_time_seconds"
        )
        or 0.0
    )

    extraction_time = (
        runtime.get(
            "extraction_time"
        )
        or runtime.get(
            "extraction_time_seconds"
        )
        or 0.0
    )

    total_time = (
        runtime.get(
            "total_runtime"
        )
        or runtime.get(
            "total_time_seconds"
        )
        or (
            float(embedding_time)
            + float(attack_time)
            + float(extraction_time)
        )
    )

    row = {
        "experiment_id": safe_record.get(
            "experiment_id"
        ),
        "experiment_key": safe_record.get(
            "experiment_key"
        ),
        "timestamp_utc": safe_record.get(
            "timestamp_utc",
            utc_now_iso(),
        ),
        "benchmark_version": safe_record.get(
            "benchmark_version"
        ),
        "benchmark_tag": safe_record.get(
            "benchmark_tag"
        ),

        "host_image_id": (
            host.get("image_id")
            or safe_record.get(
                "host_image_id"
            )
        ),
        "host_image_name": (
            host.get("file_name")
            or safe_record.get(
                "host_image_name"
            )
        ),
        "host_image_hash": (
            host.get("file_hash")
            or safe_record.get(
                "host_image_hash"
            )
        ),
        "host_dataset": (
            host.get("benchmark_tag")
            or host.get("dataset_name")
            or safe_record.get(
                "host_dataset"
            )
        ),
        "host_width": (
            host.get("width")
            or host.get(
                "processed_width"
            )
            or safe_record.get(
                "host_width"
            )
        ),
        "host_height": (
            host.get("height")
            or host.get(
                "processed_height"
            )
            or safe_record.get(
                "host_height"
            )
        ),

        "watermark_id": (
            watermark.get(
                "watermark_id"
            )
            or safe_record.get(
                "watermark_id"
            )
        ),
        "watermark_type": (
            watermark.get(
                "watermark_type"
            )
            or safe_record.get(
                "watermark_type"
            )
        ),
        "watermark_hash": (
            watermark.get(
                "file_hash"
            )
            or safe_record.get(
                "watermark_hash"
            )
        ),
        "watermark_width": watermark_width,
        "watermark_height": watermark_height,
        "watermark_payload_bits": (
            watermark.get(
                "payload_bits"
            )
            or safe_record.get(
                "watermark_payload_bits"
            )
        ),
        "watermark_density": (
            watermark.get("density")
            or safe_record.get(
                "watermark_density"
            )
        ),

        "method": (
            embedding.get("method")
            or safe_record.get(
                "method"
            )
        ),
        "alpha": (
            embedding.get("alpha")
            or safe_record.get(
                "alpha"
            )
        ),
        "watermark_size": watermark_size,
        "channel": (
            embedding.get("channel")
            or safe_record.get(
                "channel"
            )
        ),
        "block_size": (
            embedding.get(
                "block_size"
            )
            or safe_record.get(
                "block_size"
            )
        ),
        "decomposition_level": (
            embedding.get(
                "decomposition_level"
            )
            or safe_record.get(
                "decomposition_level"
            )
        ),
        "subband": (
            embedding.get("subband")
            or safe_record.get(
                "subband"
            )
        ),

        "attack_type": (
            attack.get(
                "attack_type"
            )
            or safe_record.get(
                "attack_type"
            )
            or "none"
        ),
        "attack_parameter": (
            attack.get("parameter")
            if "parameter" in attack
            else safe_record.get(
                "attack_parameter"
            )
        ),
        "attack_steps": attack_steps,

        "mse": (
            imperceptibility.get("mse")
            or safe_record.get("mse")
        ),
        "psnr": (
            imperceptibility.get("psnr")
            or safe_record.get("psnr")
        ),
        "ssim": (
            imperceptibility.get("ssim")
            or safe_record.get("ssim")
        ),

        "ber": (
            robustness.get("ber")
            or safe_record.get("ber")
        ),
        "correlation": (
            robustness.get(
                "correlation"
            )
            or safe_record.get(
                "correlation"
            )
        ),
        "extracted_successfully": (
            robustness.get(
                "extracted_successfully"
            )
            if "extracted_successfully"
            in robustness
            else safe_record.get(
                "extracted_successfully"
            )
        ),
        "detection_result": (
            safe_record.get(
                "detection_result"
            )
        ),

        "embedding_time_seconds": embedding_time,
        "attack_time_seconds": attack_time,
        "extraction_time_seconds": extraction_time,
        "total_time_seconds": total_time,

        "predicted_psnr": prediction.get(
            "expected_psnr"
        ),
        "predicted_ssim": prediction.get(
            "expected_ssim"
        ),
        "predicted_ber": prediction.get(
            "expected_ber"
        ),
        "predicted_correlation": prediction.get(
            "expected_correlation"
        ),
        "prediction_confidence": prediction.get(
            "confidence"
        ),
        "prediction_source": prediction.get(
            "prediction_source"
        ),

        "experiment_json_path": safe_record.get(
            "experiment_json_path"
        ),
        "experiment_directory": safe_record.get(
            "experiment_directory"
        ),

        "status": safe_record.get(
            "status",
            "SUCCESS",
        ),
        "error_message": safe_record.get(
            "error_message",
            "",
        ),
        "notes": safe_record.get(
            "notes",
            "",
        ),
    }

    # Preserve explicitly supplied flat values.
    for column in DEFAULT_EXPERIMENT_COLUMNS:
        if (
            column in safe_record
            and safe_record[column]
            is not None
        ):
            row[column] = safe_record[
                column
            ]

    return make_json_safe(
        row
    )


# ============================================================
# EXPERIMENT LOGGER
# ============================================================

class ExperimentLogger:
    """
    Safe persistent logger for AWSRE benchmark experiments.
    """

    def __init__(
        self,
        config: ExperimentLoggerConfig,
    ) -> None:
        if not isinstance(
            config,
            ExperimentLoggerConfig,
        ):
            raise TypeError(
                "config must be ExperimentLoggerConfig."
            )

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

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    def _load_completed_keys(
        self,
    ) -> Set[str]:
        """
        Load successful experiment keys from the CSV index.
        """
        csv_path = self.config.csv_path

        if (
            not csv_path.exists()
            or csv_path.stat().st_size == 0
        ):
            return set()

        try:
            dataframe = pd.read_csv(
                csv_path,
                usecols=[
                    "experiment_key",
                    "status",
                ],
            )

        except (
            ValueError,
            pd.errors.EmptyDataError,
        ):
            return set()

        if dataframe.empty:
            return set()

        status_series = dataframe[
            "status"
        ].fillna("").astype(str).str.upper()

        key_series = dataframe[
            "experiment_key"
        ].fillna("").astype(str)

        return {
            key
            for key, status
            in zip(
                key_series,
                status_series,
            )
            if (
                key
                and status == "SUCCESS"
            )
        }

    def already_completed(
        self,
        experiment_key: str,
    ) -> bool:
        """
        Return whether a successful result already exists.
        """
        return str(
            experiment_key
        ) in self._completed_keys

    @property
    def completed_count(self) -> int:
        return len(
            self._completed_keys
        )

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    def experiment_directory(
        self,
        experiment_id: str,
    ) -> Path:
        """
        Return the experiment-specific directory.
        """
        safe_id = (
            str(experiment_id)
            .strip()
            .replace("/", "_")
            .replace("\\", "_")
        )

        if not safe_id:
            raise ValueError(
                "experiment_id cannot be empty."
            )

        return (
            self.config.experiments_path
            / safe_id
        )

    def experiment_json_path(
        self,
        experiment_id: str,
    ) -> Path:
        return (
            self.experiment_directory(
                experiment_id
            )
            / "metadata.json"
        )

    # --------------------------------------------------------
    # RECORD WRITING
    # --------------------------------------------------------

    def log_experiment(
        self,
        record: Any,
        *,
        allow_duplicate: bool = False,
    ) -> Dict[str, Any]:
        """
        Save one experiment to JSON and CSV.

        Returns the flattened CSV row.
        """
        with self._lock:
            raw_record = make_json_safe(
                record
            )

            if not isinstance(
                raw_record,
                dict,
            ):
                raise TypeError(
                    "Experiment record must serialize "
                    "to a dictionary."
                )

            experiment_id = raw_record.get(
                "experiment_id"
            )

            experiment_key = raw_record.get(
                "experiment_key"
            )

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
                and self.already_completed(
                    experiment_key
                )
            ):
                raise ValueError(
                    "Experiment has already been completed: "
                    f"{experiment_key}"
                )

            status = str(
                raw_record.get(
                    "status",
                    "SUCCESS",
                )
            ).upper()

            if status not in VALID_STATUSES:
                raise ValueError(
                    f"Unsupported experiment status: {status}."
                )

            raw_record[
                "status"
            ] = status

            raw_record.setdefault(
                "timestamp_utc",
                utc_now_iso(),
            )

            experiment_directory = (
                self.experiment_directory(
                    experiment_id
                )
            )

            experiment_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            json_path = (
                self.experiment_json_path(
                    experiment_id
                )
            )

            raw_record[
                "experiment_directory"
            ] = str(
                experiment_directory
            )

            raw_record[
                "experiment_json_path"
            ] = str(
                json_path
            )

            try:
                if (
                    self.config
                    .save_json_per_experiment
                ):
                    atomic_write_json(
                        json_path,
                        raw_record,
                    )

                    self._verify_json(
                        json_path,
                        expected_experiment_id=(
                            experiment_id
                        ),
                    )

                flattened_row = (
                    flatten_experiment_record(
                        raw_record
                    )
                )

                append_csv_row(
                    self.config.csv_path,
                    flattened_row,
                    self.config.experiment_columns,
                )

                self._verify_csv_last_record(
                    experiment_id=experiment_id
                )

                if status == "SUCCESS":
                    self._completed_keys.add(
                        str(experiment_key)
                    )

                self.successful_writes += 1

                if (
                    self.successful_writes
                    % self.config
                    .checkpoint_interval
                    == 0
                ):
                    self.save_checkpoint()

                if (
                    self.successful_writes
                    % self.config
                    .backup_interval
                    == 0
                ):
                    self.create_backup()

                return flattened_row

            except Exception:
                self.failed_writes += 1
                raise

    def _verify_json(
        self,
        json_path: Path,
        *,
        expected_experiment_id: str,
    ) -> None:
        """
        Re-open and validate a saved experiment JSON.
        """
        if (
            not json_path.exists()
            or json_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Experiment JSON was not written correctly."
            )

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            saved = json.load(
                file_handle
            )

        if (
            str(
                saved.get(
                    "experiment_id"
                )
            )
            != str(
                expected_experiment_id
            )
        ):
            raise RuntimeError(
                "Saved experiment JSON failed ID verification."
            )

    def _verify_csv_last_record(
        self,
        *,
        experiment_id: str,
    ) -> None:
        """
        Verify that the experiment appears in the CSV.
        """
        dataframe = pd.read_csv(
            self.config.csv_path
        )

        if dataframe.empty:
            raise RuntimeError(
                "CSV is empty after experiment write."
            )

        experiment_ids = (
            dataframe[
                "experiment_id"
            ]
            .fillna("")
            .astype(str)
        )

        if str(
            experiment_id
        ) not in set(
            experiment_ids
        ):
            raise RuntimeError(
                "Experiment was not found after CSV write."
            )

    # --------------------------------------------------------
    # ERROR LOGGING
    # --------------------------------------------------------

    def log_error(
        self,
        error_record: Dict[str, Any],
    ) -> None:
        """
        Append one error record immediately.
        """
        normalized = {
            "timestamp_utc": (
                error_record.get(
                    "timestamp_utc"
                )
                or utc_now_iso()
            ),
            "experiment_id": error_record.get(
                "experiment_id"
            ),
            "experiment_key": error_record.get(
                "experiment_key"
            ),
            "host_image_name": error_record.get(
                "host_image_name"
            ),
            "watermark_type": error_record.get(
                "watermark_type"
            ),
            "method": error_record.get(
                "method"
            ),
            "alpha": error_record.get(
                "alpha"
            ),
            "attack_type": error_record.get(
                "attack_type"
            ),
            "attack_parameter": error_record.get(
                "attack_parameter"
            ),
            "error_type": error_record.get(
                "error_type"
            ),
            "error_message": error_record.get(
                "error_message"
            ),
            "traceback": error_record.get(
                "traceback"
            ),
        }

        with self._lock:
            append_csv_row(
                self.config.errors_path,
                normalized,
                self.config.error_columns,
            )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    def save_checkpoint(
        self,
        *,
        extra_data: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Path:
        """
        Save current logger progress.
        """
        checkpoint = {
            "logger_version": LOGGER_VERSION,
            "timestamp_utc": utc_now_iso(),
            "completed_count": (
                self.completed_count
            ),
            "successful_writes": (
                self.successful_writes
            ),
            "failed_writes": (
                self.failed_writes
            ),
            "csv_path": str(
                self.config.csv_path
            ),
            "errors_path": str(
                self.config.errors_path
            ),
            "completed_experiment_keys": sorted(
                self._completed_keys
            ),
            "extra_data": (
                make_json_safe(
                    extra_data
                )
                if extra_data
                else {}
            ),
        }

        return atomic_write_json(
            self.config.checkpoint_path,
            checkpoint,
        )

    def load_checkpoint(
        self,
    ) -> Dict[str, Any]:
        """
        Load the current checkpoint if it exists.
        """
        path = self.config.checkpoint_path

        if not path.exists():
            return {}

        with path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            checkpoint = json.load(
                file_handle
            )

        saved_keys = checkpoint.get(
            "completed_experiment_keys",
            [],
        )

        self._completed_keys.update(
            str(key)
            for key in saved_keys
            if key
        )

        return checkpoint

    # --------------------------------------------------------
    # BACKUP
    # --------------------------------------------------------

    def create_backup(
        self,
    ) -> Path:
        """
        Create and verify a backup of the main CSV.
        """
        source = self.config.csv_path

        if (
            not source.exists()
            or source.stat().st_size == 0
        ):
            raise RuntimeError(
                "Cannot back up a missing or empty CSV."
            )

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d-%H%M%S"
        )

        timestamped_backup = (
            self.config.backups_path
            / (
                "awsre_experiments_"
                f"{timestamp}.csv"
            )
        )

        shutil.copy2(
            source,
            timestamped_backup,
        )

        shutil.copy2(
            source,
            self.config.latest_backup_path,
        )

        if (
            timestamped_backup.stat().st_size
            != source.stat().st_size
        ):
            raise RuntimeError(
                "Timestamped CSV backup verification failed."
            )

        if (
            self.config
            .latest_backup_path
            .stat()
            .st_size
            != source.stat().st_size
        ):
            raise RuntimeError(
                "Latest CSV backup verification failed."
            )

        pd.read_csv(
            timestamped_backup
        )

        return timestamped_backup

    # --------------------------------------------------------
    # VALIDATION AND SUMMARY
    # --------------------------------------------------------

    def validate_storage(
        self,
        *,
        minimum_rows: int = 0,
    ) -> Dict[str, Any]:
        """
        Re-open logger outputs and return a validation report.
        """
        report: Dict[str, Any] = {
            "csv_exists": (
                self.config.csv_path.exists()
            ),
            "errors_csv_exists": (
                self.config.errors_path.exists()
            ),
            "csv_size_bytes": 0,
            "csv_rows": 0,
            "csv_columns": 0,
            "missing_columns": [],
            "json_files": 0,
            "valid": False,
        }

        if not report[
            "csv_exists"
        ]:
            return report

        report[
            "csv_size_bytes"
        ] = self.config.csv_path.stat().st_size

        try:
            dataframe = pd.read_csv(
                self.config.csv_path
            )

            report[
                "csv_rows"
            ] = len(
                dataframe
            )

            report[
                "csv_columns"
            ] = len(
                dataframe.columns
            )

            report[
                "missing_columns"
            ] = [
                column
                for column
                in self.config
                .experiment_columns
                if column
                not in dataframe.columns
            ]

            report[
                "json_files"
            ] = len(
                list(
                    self.config
                    .experiments_path
                    .glob(
                        "*/metadata.json"
                    )
                )
            )

            report["valid"] = (
                report[
                    "csv_size_bytes"
                ] > 0
                and report[
                    "csv_rows"
                ] >= minimum_rows
                and not report[
                    "missing_columns"
                ]
            )

        except Exception as exc:
            report[
                "validation_error"
            ] = str(
                exc
            )

        return report

    def dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return the current experiment index.
        """
        return pd.read_csv(
            self.config.csv_path
        )

    def summary(
        self,
    ) -> Dict[str, Any]:
        """
        Return lightweight experiment statistics.
        """
        dataframe = self.dataframe()

        if dataframe.empty:
            return {
                "total_rows": 0,
                "successful_rows": 0,
                "failed_rows": 0,
                "completed_keys": (
                    self.completed_count
                ),
            }

        status = (
            dataframe["status"]
            .fillna("")
            .astype(str)
            .str.upper()
        )

        return {
            "total_rows": int(
                len(dataframe)
            ),
            "successful_rows": int(
                np.count_nonzero(
                    status == "SUCCESS"
                )
            ),
            "failed_rows": int(
                np.count_nonzero(
                    status == "FAILED"
                )
            ),
            "skipped_rows": int(
                np.count_nonzero(
                    status == "SKIPPED"
                )
            ),
            "completed_keys": (
                self.completed_count
            ),
            "csv_path": str(
                self.config.csv_path
            ),
            "errors_path": str(
                self.config.errors_path
            ),
        }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary_directory:
        config = ExperimentLoggerConfig(
            output_root=temporary_directory,
            backup_interval=2,
            checkpoint_interval=2,
        )

        logger = ExperimentLogger(
            config
        )

        for index in range(3):
            experiment_id = (
                f"EXP-TEST-{index + 1:04d}"
            )

            experiment_key = (
                f"HOST-1|WM-1|DCT|"
                f"{10 + index}|none"
            )

            logger.log_experiment({
                "experiment_id": experiment_id,
                "experiment_key": experiment_key,
                "timestamp_utc": utc_now_iso(),
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
                    "mse": 0.1,
                    "psnr": 50.0,
                    "ssim": 0.999,
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

        validation = logger.validate_storage(
            minimum_rows=3
        )

        assert validation["valid"]
        assert validation["csv_rows"] == 3
        assert validation["json_files"] == 3

        assert logger.already_completed(
            "HOST-1|WM-1|DCT|10|none"
        )

        print("=" * 72)
        print("AWSRE EXPERIMENT LOGGER SELF TEST")
        print("=" * 72)

        print(
            "Validation:",
            validation,
        )

        print(
            "Summary:",
            logger.summary(),
        )

        print(
            "\n✅ Experiment logger self test passed."
        )
