"""
AWSRE Benchmark Runner

Orchestrates the complete benchmark workflow:

Host dataset
    -> Watermark generation
    -> Watermark embedding
    -> Attack simulation
    -> Watermark extraction
    -> Metric calculation
    -> Safe JSON/CSV logging

The runner currently works with every algorithm registered in
watermarking.registry. At this stage, DCT is the first fully
implemented method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import time
import traceback
import uuid

import numpy as np

from attacks.pipeline import AttackPipeline
from benchmark.config import (
    AttackSpec,
    BenchmarkConfig,
)
from benchmark.dataset_loader import (
    DatasetLoadResult,
    HostImageItem,
    load_host_dataset,
    validate_loaded_dataset,
)
from benchmark.experiment_logger import (
    ExperimentLogger,
    ExperimentLoggerConfig,
)
from benchmark.watermark_generator.registry import (
    generate_watermark,
    load_builtin_generators,
)
from watermarking.metrics import (
    calculate_ber,
    calculate_correlation,
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
)
from watermarking.registry import (
    create_watermarker,
    load_builtin_watermarkers,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def generate_experiment_id(
    sequence_number: int,
) -> str:
    """
    Generate a readable experiment ID.
    """
    return (
        f"EXP-"
        f"{sequence_number:08d}-"
        f"{uuid.uuid4().hex[:6].upper()}"
    )


def stable_text_hash(
    value: str,
) -> str:
    """
    Generate a stable SHA-256 hash from text.
    """
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def ensure_finite_metric(
    value: Any,
    *,
    name: str,
) -> float:
    """Return a required metric as a finite Python float."""
    if value is None:
        raise ValueError(f"Required metric '{name}' is missing.")

    result = float(value)

    if not np.isfinite(result):
        raise ValueError(
            f"Required metric '{name}' is not finite: {value!r}."
        )

    return result


def build_experiment_key(
    *,
    host_hash: str,
    watermark_hash: str,
    method: str,
    alpha: float,
    watermark_size: Tuple[int, int],
    attack_type: str,
    attack_parameter: Any,
) -> str:
    """
    Create a deterministic experiment key for resume logic.
    """
    width, height = watermark_size

    raw_key = (
        f"{host_hash}|"
        f"{watermark_hash}|"
        f"{method}|"
        f"{float(alpha):.8f}|"
        f"{width}x{height}|"
        f"{attack_type}|"
        f"{attack_parameter}"
    )

    return stable_text_hash(
        raw_key
    )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class BenchmarkProgress:
    expected_experiments: int

    attempted_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    skipped_experiments: int = 0

    started_at_utc: str = field(
        default_factory=utc_now_iso
    )

    elapsed_seconds: float = 0.0

    @property
    def completed_experiments(self) -> int:
        return (
            self.successful_experiments
            + self.failed_experiments
            + self.skipped_experiments
        )

    @property
    def completion_ratio(self) -> float:
        if self.expected_experiments <= 0:
            return 0.0

        return min(
            1.0,
            self.completed_experiments
            / self.expected_experiments,
        )

    @property
    def completion_percent(self) -> float:
        return (
            self.completion_ratio
            * 100.0
        )

    @property
    def average_seconds_per_experiment(
        self,
    ) -> float:
        if self.attempted_experiments <= 0:
            return 0.0

        return (
            self.elapsed_seconds
            / self.attempted_experiments
        )

    @property
    def estimated_remaining_seconds(
        self,
    ) -> float:
        remaining = max(
            0,
            self.expected_experiments
            - self.completed_experiments,
        )

        return (
            remaining
            * self.average_seconds_per_experiment
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        data["completed_experiments"] = (
            self.completed_experiments
        )

        data["completion_percent"] = (
            self.completion_percent
        )

        data[
            "average_seconds_per_experiment"
        ] = self.average_seconds_per_experiment

        data[
            "estimated_remaining_seconds"
        ] = self.estimated_remaining_seconds

        return data


@dataclass
class BenchmarkRunResult:
    progress: BenchmarkProgress

    csv_path: str
    errors_path: str
    checkpoint_path: str

    started_at_utc: str
    finished_at_utc: str

    runtime_seconds: float

    storage_validation: Dict[str, Any]
    logger_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "progress": self.progress.to_dict(),
            "csv_path": self.csv_path,
            "errors_path": self.errors_path,
            "checkpoint_path": self.checkpoint_path,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "runtime_seconds": self.runtime_seconds,
            "storage_validation": (
                self.storage_validation
            ),
            "logger_summary": (
                self.logger_summary
            ),
        }


# ============================================================
# WATERMARK GENERATION OPTIONS
# ============================================================

def default_generation_options(
    watermark_type: str,
    *,
    seed: int,
) -> Dict[str, Any]:
    """
    Return default benchmark options for each watermark type.

    Logo and Signature require external sources. Their source
    paths must be supplied through BenchmarkRunner generation_options.
    """
    normalized = (
        watermark_type
        .strip()
        .lower()
    )

    if normalized == "binary pattern":
        return {
            "pattern": "random",
            "seed": seed,
            "density": 0.5,
        }

    if normalized == "text":
        return {
            "text": "AWSRE",
            "seed": seed,
            "thickness": 1,
        }

    if normalized == "qr code":
        return {
            "text": "AWSRE-BENCH",
            "seed": seed,
            "error_correction": "M",
        }

    if normalized in {
        "logo",
        "signature",
    }:
        return {
            "seed": seed,
        }

    return {
        "seed": seed,
    }


# ============================================================
# BENCHMARK RUNNER
# ============================================================

class BenchmarkRunner:
    """
    Main AWSRE benchmark orchestration engine.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        generation_options: Optional[
            Dict[str, Dict[str, Any]]
        ] = None,
        logger: Optional[
            ExperimentLogger
        ] = None,
    ) -> None:
        if not isinstance(
            config,
            BenchmarkConfig,
        ):
            raise TypeError(
                "config must be BenchmarkConfig."
            )

        self.config = config

        self.generation_options = (
            generation_options
            or {}
        )

        self.config.create_directories()
        self.config.save_json()

        load_builtin_generators(
            strict=True
        )

        load_builtin_watermarkers()

        if logger is None:
            logger_config = (
                ExperimentLoggerConfig(
                    output_root=(
                        self.config.output_path
                    ),
                    backup_interval=(
                        self.config.backup_interval
                    ),
                    checkpoint_interval=(
                        self.config
                        .checkpoint_interval
                    ),
                    resume_enabled=(
                        self.config.resume_enabled
                    ),
                    save_json_per_experiment=True,
                )
            )

            logger = ExperimentLogger(
                logger_config
            )

        self.logger = logger

        self.attack_pipeline = (
            AttackPipeline(
                default_seed=(
                    self.config.random_seed
                )
            )
        )

        self.dataset_result: Optional[
            DatasetLoadResult
        ] = None

        self.progress = BenchmarkProgress(
            expected_experiments=(
                self.config
                .expected_experiment_count
            )
        )

        self._sequence_number = 0

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    def load_dataset(
        self,
    ) -> DatasetLoadResult:
        """
        Load and validate host images.
        """
        result = load_host_dataset(
            directory=(
                self.config.host_images_path
            ),
            target_size=(
                self.config.host_target_size
            ),
            max_images=(
                self.config.max_host_images
            ),
            dataset_name=(
                self.config.benchmark_tag
            ),
            color_mode="grayscale",
            recursive=True,
            remove_duplicates=True,
            strict_count=(
                self.config.strict_validation
            ),
        )

        validate_loaded_dataset(
            result,
            minimum_images=1,
            expected_size=(
                self.config.host_target_size
            ),
        )

        self.dataset_result = result

        return result

    # --------------------------------------------------------
    # WATERMARK GENERATION
    # --------------------------------------------------------

    def generate_benchmark_watermark(
        self,
        *,
        watermark_type: str,
        watermark_size: Tuple[int, int],
        seed: int,
    ):
        """
        Generate one benchmark watermark.
        """
        options = default_generation_options(
            watermark_type,
            seed=seed,
        )

        custom_options = (
            self.generation_options.get(
                watermark_type,
                self.generation_options.get(
                    watermark_type.lower(),
                    {},
                ),
            )
        )

        options.update(
            custom_options
        )

        if watermark_type in {
            "Logo",
            "Signature",
        }:
            if "source" not in options:
                raise ValueError(
                    f"{watermark_type} generation "
                    "requires a 'source' image."
                )

        return generate_watermark(
            watermark_type=watermark_type,
            size=watermark_size,
            default_seed=seed,
            **options,
        )

    # --------------------------------------------------------
    # SINGLE EMBEDDING
    # --------------------------------------------------------

    def embed_once(
        self,
        *,
        host: HostImageItem,
        watermark_result,
        method: str,
        alpha: float,
    ):
        """
        Perform one embedding operation.
        """
        algorithm = create_watermarker(
            method=method,
            alpha=alpha,
        )

        embedding_result = (
            algorithm.embed(
                host.image,
                watermark_result.image,
            )
        )

        mse = ensure_finite_metric(
            calculate_mse(
                host.image,
                embedding_result.watermarked_image,
            ),
            name="mse",
        )

        psnr = ensure_finite_metric(
            calculate_psnr(
                host.image,
                embedding_result.watermarked_image,
            ),
            name="psnr",
        )

        ssim = ensure_finite_metric(
            calculate_ssim(
                host.image,
                embedding_result.watermarked_image,
            ),
            name="ssim",
        )

        return (
            algorithm,
            embedding_result,
            {
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
            },
        )

    # --------------------------------------------------------
    # SINGLE ATTACK EXPERIMENT
    # --------------------------------------------------------

    def run_attack_experiment(
        self,
        *,
        host: HostImageItem,
        watermark_result,
        algorithm,
        embedding_result,
        imperceptibility_metrics: Dict[
            str,
            float,
        ],
        method: str,
        alpha: float,
        watermark_size: Tuple[int, int],
        attack_spec: AttackSpec,
    ) -> None:
        """
        Apply attack, extract watermark, calculate metrics
        and save the experiment immediately.
        """
        self._sequence_number += 1

        experiment_id = (
            generate_experiment_id(
                self._sequence_number
            )
        )

        experiment_key = (
            build_experiment_key(
                host_hash=host.file_hash,
                watermark_hash=(
                    watermark_result.file_hash
                ),
                method=method,
                alpha=alpha,
                watermark_size=(
                    watermark_size
                ),
                attack_type=(
                    attack_spec.attack_type
                ),
                attack_parameter=(
                    attack_spec.parameter
                ),
            )
        )

        if (
            self.config.resume_enabled
            and self.logger.already_completed(
                experiment_key
            )
        ):
            self.progress.skipped_experiments += 1

            return

        self.progress.attempted_experiments += 1

        started_at = time.perf_counter()

        try:
            attack_result = (
                self.attack_pipeline.apply(
                    embedding_result
                    .watermarked_image,
                    attack_spec.attack_type,
                    attack_spec.parameter,
                )
            )

            extraction_result = (
                algorithm.extract(
                    host.image,
                    attack_result.image,
                    watermark_result.image.shape,
                )
            )

            extracted_watermark = (
                extraction_result
                .extracted_watermark
            )

            ber = ensure_finite_metric(
                calculate_ber(
                    watermark_result.image,
                    extracted_watermark,
                ),
                name="ber",
            )

            correlation = ensure_finite_metric(
                calculate_correlation(
                    watermark_result.image,
                    extracted_watermark,
                ),
                name="correlation",
            )

            extracted_successfully = bool(
                np.isfinite(ber)
                and np.isfinite(correlation)
            )

            detection_result = (
                "VERIFIED"
                if (
                    ber <= 0.10
                    and correlation >= 0.80
                )
                else "NOT_VERIFIED"
            )

            total_time = (
                embedding_result.runtime
                + attack_result.runtime_seconds
                + extraction_result.runtime
            )

            record = {
                "experiment_id": experiment_id,
                "experiment_key": experiment_key,
                "timestamp_utc": utc_now_iso(),
                "benchmark_version": (
                    self.config
                    .benchmark_version
                ),
                "benchmark_tag": (
                    self.config.benchmark_tag
                ),

                "host": {
                    "image_id": host.image_id,
                    "file_name": (
                        host.file_name
                    ),
                    "file_path": (
                        host.file_path
                    ),
                    "file_hash": (
                        host.file_hash
                    ),
                    "dataset_name": (
                        host.dataset_name
                    ),
                    "width": (
                        host.processed_width
                    ),
                    "height": (
                        host.processed_height
                    ),
                },

                "watermark": {
                    **watermark_result
                    .metadata_dict(),
                },

                "embedding": {
                    "method": method,
                    "alpha": float(alpha),
                    "watermark_width": (
                        watermark_result.width
                    ),
                    "watermark_height": (
                        watermark_result.height
                    ),
                    "channel": "Grayscale",
                    "block_size": (
                        embedding_result
                        .metadata
                        .get(
                            "block_size"
                        )
                    ),
                    "decomposition_level": 1,
                    "subband": None,
                    "metadata": (
                        embedding_result
                        .metadata
                    ),
                },

                "attack": {
                    "attack_type": (
                        attack_result.attack_type
                    ),
                    "parameter": (
                        attack_spec.parameter
                    ),
                    "steps": (
                        attack_result.steps
                    ),
                    "metadata": (
                        attack_result.metadata
                    ),
                },

                "imperceptibility": (
                    imperceptibility_metrics
                ),

                "robustness": {
                    "ber": ber,
                    "correlation": correlation,
                    "extracted_successfully": (
                        extracted_successfully
                    ),
                },

                "runtime": {
                    "embedding_time": (
                        embedding_result.runtime
                    ),
                    "attack_time": (
                        attack_result
                        .runtime_seconds
                    ),
                    "extraction_time": (
                        extraction_result.runtime
                    ),
                    "total_runtime": (
                        total_time
                    ),
                },

                "detection_result": (
                    detection_result
                ),

                # Flat metric copies make zero values explicit and
                # protect them through every serialization layer.
                "mse": imperceptibility_metrics["mse"],
                "psnr": imperceptibility_metrics["psnr"],
                "ssim": imperceptibility_metrics["ssim"],
                "ber": ber,
                "correlation": correlation,

                "status": "SUCCESS",
                "error_message": "",
                "notes": "",
            }

            self.logger.log_experiment(
                record
            )

            self.progress.successful_experiments += 1

        except Exception as exc:
            self.progress.failed_experiments += 1

            error_traceback = (
                traceback.format_exc()
            )

            self.logger.log_error({
                "timestamp_utc": utc_now_iso(),
                "experiment_id": experiment_id,
                "experiment_key": experiment_key,
                "host_image_name": (
                    host.file_name
                ),
                "watermark_type": (
                    watermark_result
                    .watermark_type
                ),
                "method": method,
                "alpha": alpha,
                "attack_type": (
                    attack_spec.attack_type
                ),
                "attack_parameter": (
                    attack_spec.parameter
                ),
                "error_type": (
                    type(exc).__name__
                ),
                "error_message": str(exc),
                "traceback": (
                    error_traceback
                ),
            })

            if self.config.stop_on_error:
                raise

        finally:
            self.progress.elapsed_seconds += (
                time.perf_counter()
                - started_at
            )

            self.print_progress()

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    def run(
        self,
    ) -> BenchmarkRunResult:
        """
        Execute the complete configured benchmark.
        """
        started_at_utc = utc_now_iso()
        started_at = time.perf_counter()

        if self.dataset_result is None:
            self.load_dataset()

        if self.dataset_result is None:
            raise RuntimeError(
                "Dataset could not be loaded."
            )

        for host_index, host in enumerate(
            self.dataset_result.images
        ):
            for watermark_type in (
                self.config.watermark_types
            ):
                for watermark_size in (
                    self.config.watermark_sizes
                ):
                    watermark_seed = (
                        self.config.random_seed
                        + host_index
                    )

                    try:
                        watermark_result = (
                            self
                            .generate_benchmark_watermark(
                                watermark_type=(
                                    watermark_type
                                ),
                                watermark_size=(
                                    watermark_size
                                ),
                                seed=(
                                    watermark_seed
                                ),
                            )
                        )

                    except Exception as exc:
                        self.log_generation_error(
                            host=host,
                            watermark_type=(
                                watermark_type
                            ),
                            watermark_size=(
                                watermark_size
                            ),
                            error=exc,
                        )

                        if (
                            self.config.stop_on_error
                        ):
                            raise

                        continue

                    for method in (
                        self.config.methods
                    ):
                        for alpha in (
                            self.config.alpha_values
                        ):
                            try:
                                (
                                    algorithm,
                                    embedding_result,
                                    imperceptibility,
                                ) = self.embed_once(
                                    host=host,
                                    watermark_result=(
                                        watermark_result
                                    ),
                                    method=method,
                                    alpha=alpha,
                                )

                            except Exception as exc:
                                self.log_embedding_error(
                                    host=host,
                                    watermark_result=(
                                        watermark_result
                                    ),
                                    method=method,
                                    alpha=alpha,
                                    error=exc,
                                )

                                if (
                                    self.config
                                    .stop_on_error
                                ):
                                    raise

                                continue

                            for attack_spec in (
                                self.config.attacks
                            ):
                                self.run_attack_experiment(
                                    host=host,
                                    watermark_result=(
                                        watermark_result
                                    ),
                                    algorithm=algorithm,
                                    embedding_result=(
                                        embedding_result
                                    ),
                                    imperceptibility_metrics=(
                                        imperceptibility
                                    ),
                                    method=method,
                                    alpha=alpha,
                                    watermark_size=(
                                        watermark_size
                                    ),
                                    attack_spec=(
                                        attack_spec
                                    ),
                                )

        runtime_seconds = (
            time.perf_counter()
            - started_at
        )

        self.progress.elapsed_seconds = (
            runtime_seconds
        )

        self.logger.save_checkpoint(
            extra_data={
                "progress": (
                    self.progress.to_dict()
                )
            }
        )

        if (
            self.logger.summary()[
                "total_rows"
            ] > 0
        ):
            self.logger.create_backup()

        storage_validation = (
            self.logger.validate_storage(
                minimum_rows=(
                    1
                    if self.progress
                    .successful_experiments
                    > 0
                    else 0
                )
            )
        )

        result = BenchmarkRunResult(
            progress=self.progress,
            csv_path=str(
                self.logger.config.csv_path
            ),
            errors_path=str(
                self.logger.config.errors_path
            ),
            checkpoint_path=str(
                self.logger
                .config
                .checkpoint_path
            ),
            started_at_utc=(
                started_at_utc
            ),
            finished_at_utc=utc_now_iso(),
            runtime_seconds=runtime_seconds,
            storage_validation=(
                storage_validation
            ),
            logger_summary=(
                self.logger.summary()
            ),
        )

        self.print_final_summary(
            result
        )

        return result

    # --------------------------------------------------------
    # ERROR HELPERS
    # --------------------------------------------------------

    def log_generation_error(
        self,
        *,
        host: HostImageItem,
        watermark_type: str,
        watermark_size: Tuple[int, int],
        error: Exception,
    ) -> None:
        self.logger.log_error({
            "timestamp_utc": utc_now_iso(),
            "experiment_id": "",
            "experiment_key": "",
            "host_image_name": (
                host.file_name
            ),
            "watermark_type": (
                watermark_type
            ),
            "method": "",
            "alpha": None,
            "attack_type": "",
            "attack_parameter": None,
            "error_type": (
                type(error).__name__
            ),
            "error_message": (
                "Watermark generation failed: "
                f"{error}"
            ),
            "traceback": (
                traceback.format_exc()
            ),
        })

    def log_embedding_error(
        self,
        *,
        host: HostImageItem,
        watermark_result,
        method: str,
        alpha: float,
        error: Exception,
    ) -> None:
        self.logger.log_error({
            "timestamp_utc": utc_now_iso(),
            "experiment_id": "",
            "experiment_key": "",
            "host_image_name": (
                host.file_name
            ),
            "watermark_type": (
                watermark_result
                .watermark_type
            ),
            "method": method,
            "alpha": alpha,
            "attack_type": "",
            "attack_parameter": None,
            "error_type": (
                type(error).__name__
            ),
            "error_message": (
                "Embedding failed: "
                f"{error}"
            ),
            "traceback": (
                traceback.format_exc()
            ),
        })

    # --------------------------------------------------------
    # CONSOLE OUTPUT
    # --------------------------------------------------------

    def print_progress(
        self,
    ) -> None:
        print(
            "[AWSRE] "
            f"{self.progress.completed_experiments}"
            "/"
            f"{self.progress.expected_experiments}"
            " | "
            f"{self.progress.completion_percent:.2f}%"
            " | "
            f"Success: "
            f"{self.progress.successful_experiments}"
            " | "
            f"Failed: "
            f"{self.progress.failed_experiments}"
            " | "
            f"Skipped: "
            f"{self.progress.skipped_experiments}"
        )

    @staticmethod
    def print_final_summary(
        result: BenchmarkRunResult,
    ) -> None:
        progress = result.progress

        print("\n" + "=" * 72)
        print("AWSRE BENCHMARK COMPLETED")
        print("=" * 72)

        print(
            "Expected experiments:",
            progress.expected_experiments,
        )

        print(
            "Successful:",
            progress.successful_experiments,
        )

        print(
            "Failed:",
            progress.failed_experiments,
        )

        print(
            "Skipped:",
            progress.skipped_experiments,
        )

        print(
            "Runtime:",
            f"{result.runtime_seconds:.2f} seconds",
        )

        print(
            "CSV:",
            result.csv_path,
        )

        print(
            "Error log:",
            result.errors_path,
        )

        print(
            "Storage valid:",
            result.storage_validation.get(
                "valid"
            ),
        )


# ============================================================
# CONVENIENCE WRAPPER
# ============================================================

def run_benchmark(
    config: BenchmarkConfig,
    *,
    generation_options: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
) -> BenchmarkRunResult:
    """
    Execute a benchmark through one function call.
    """
    runner = BenchmarkRunner(
        config=config,
        generation_options=(
            generation_options
        ),
    )

    return runner.run()
