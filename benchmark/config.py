"""
AWSRE Benchmark Configuration

Central configuration for AWSRE-Bench.

This module contains no Colab-specific, Streamlit-specific,
or watermarking algorithm code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


# ============================================================
# ATTACK CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class AttackSpec:
    """
    One attack configuration used by AWSRE-Bench.
    """

    attack_type: str
    parameter: Optional[float | int | str] = None

    def __post_init__(self) -> None:
        normalized = self.attack_type.strip().lower()

        if not normalized:
            raise ValueError(
                "attack_type cannot be empty."
            )

        object.__setattr__(
            self,
            "attack_type",
            normalized,
        )

    @property
    def key(self) -> str:
        return (
            f"{self.attack_type}|"
            f"{self.parameter}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# BENCHMARK CONFIGURATION
# ============================================================

@dataclass
class BenchmarkConfig:
    """
    Complete AWSRE-Bench configuration.
    """

    project_name: str = "AWSRE-Bench"
    benchmark_version: str = "0.1.0"

    benchmark_tag: str = "SMOKE_TEST"

    random_seed: int = 42

    max_host_images: int = 2

    host_target_size: Tuple[int, int] = (
        512,
        512,
    )

    methods: List[str] = field(
        default_factory=lambda: [
            "DCT",
        ]
    )

    alpha_values: List[float] = field(
        default_factory=lambda: [
            10.0,
            20.0,
        ]
    )

    watermark_sizes: List[
        Tuple[int, int]
    ] = field(
        default_factory=lambda: [
            (32, 32),
        ]
    )

    watermark_types: List[str] = field(
        default_factory=lambda: [
            "Binary Pattern",
            "Text",
        ]
    )

    attacks: List[AttackSpec] = field(
        default_factory=lambda: [
            AttackSpec(
                attack_type="none",
                parameter=None,
            ),
            AttackSpec(
                attack_type="jpeg",
                parameter=70,
            ),
            AttackSpec(
                attack_type="gaussian_noise",
                parameter=0.01,
            ),
            AttackSpec(
                attack_type="gaussian_blur",
                parameter=3,
            ),
        ]
    )

    save_every_experiment: bool = True

    backup_interval: int = 10

    checkpoint_interval: int = 10

    save_experiment_images: bool = True

    stop_on_error: bool = False

    resume_enabled: bool = True

    strict_validation: bool = True

    dataset_root: str = (
        "/content/drive/MyDrive/"
        "AWSRE_Benchmark/datasets"
    )

    output_root: str = (
        "/content/drive/MyDrive/"
        "AWSRE_Benchmark/outputs"
    )

    host_images_subdirectory: str = (
        "host_images"
    )

    watermarks_subdirectory: str = (
        "watermarks"
    )

    experiments_subdirectory: str = (
        "experiments"
    )

    logs_subdirectory: str = (
        "logs"
    )

    backups_subdirectory: str = (
        "backups"
    )

    checkpoints_subdirectory: str = (
        "checkpoints"
    )

    csv_file_name: str = (
        "awsre_experiments.csv"
    )

    errors_file_name: str = (
        "awsre_errors.csv"
    )

    checkpoint_file_name: str = (
        "benchmark_checkpoint.json"
    )

    config_file_name: str = (
        "benchmark_config.json"
    )

    def __post_init__(self) -> None:
        self.random_seed = int(
            self.random_seed
        )

        self.max_host_images = max(
            1,
            int(self.max_host_images),
        )

        width = int(
            self.host_target_size[0]
        )

        height = int(
            self.host_target_size[1]
        )

        if width <= 0 or height <= 0:
            raise ValueError(
                "host_target_size must contain "
                "positive values."
            )

        self.host_target_size = (
            width,
            height,
        )

        if not self.methods:
            raise ValueError(
                "At least one watermarking method "
                "must be configured."
            )

        if not self.alpha_values:
            raise ValueError(
                "At least one alpha value "
                "must be configured."
            )

        if not self.watermark_sizes:
            raise ValueError(
                "At least one watermark size "
                "must be configured."
            )

        if not self.watermark_types:
            raise ValueError(
                "At least one watermark type "
                "must be configured."
            )

        if not self.attacks:
            raise ValueError(
                "At least one attack configuration "
                "must be provided."
            )

        self.backup_interval = max(
            1,
            int(self.backup_interval),
        )

        self.checkpoint_interval = max(
            1,
            int(self.checkpoint_interval),
        )

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    @property
    def dataset_path(self) -> Path:
        return Path(
            self.dataset_root
        )

    @property
    def output_path(self) -> Path:
        return Path(
            self.output_root
        )

    @property
    def host_images_path(self) -> Path:
        return (
            self.dataset_path
            / self.host_images_subdirectory
        )

    @property
    def watermarks_path(self) -> Path:
        return (
            self.dataset_path
            / self.watermarks_subdirectory
        )

    @property
    def experiments_path(self) -> Path:
        return (
            self.output_path
            / self.experiments_subdirectory
        )

    @property
    def logs_path(self) -> Path:
        return (
            self.output_path
            / self.logs_subdirectory
        )

    @property
    def backups_path(self) -> Path:
        return (
            self.output_path
            / self.backups_subdirectory
        )

    @property
    def checkpoints_path(self) -> Path:
        return (
            self.output_path
            / self.checkpoints_subdirectory
        )

    @property
    def csv_path(self) -> Path:
        return (
            self.output_path
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
    def saved_config_path(self) -> Path:
        return (
            self.output_path
            / self.config_file_name
        )

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    @property
    def strategy_count_per_host(self) -> int:
        return (
            len(self.methods)
            * len(self.alpha_values)
            * len(self.watermark_sizes)
            * len(self.watermark_types)
        )

    @property
    def attack_count(self) -> int:
        return len(
            self.attacks
        )

    @property
    def expected_experiment_count(self) -> int:
        return (
            self.max_host_images
            * self.strategy_count_per_host
            * self.attack_count
        )

    # --------------------------------------------------------
    # FILE SYSTEM
    # --------------------------------------------------------

    def create_directories(self) -> None:
        directories = [
            self.dataset_path,
            self.output_path,
            self.host_images_path,
            self.watermarks_path,
            self.experiments_path,
            self.logs_path,
            self.backups_path,
            self.checkpoints_path,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        data["host_target_size"] = list(
            self.host_target_size
        )

        data["watermark_sizes"] = [
            list(size)
            for size in self.watermark_sizes
        ]

        data["attacks"] = [
            attack.to_dict()
            for attack in self.attacks
        ]

        data["expected_experiment_count"] = (
            self.expected_experiment_count
        )

        return data

    def save_json(
        self,
        path: Optional[
            str | Path
        ] = None,
    ) -> Path:
        output_path = (
            Path(path)
            if path is not None
            else self.saved_config_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                self.to_dict(),
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return output_path


# ============================================================
# DEFAULT CONFIGURATIONS
# ============================================================

def create_smoke_test_config() -> BenchmarkConfig:
    """
    Small configuration for validating the full pipeline.
    """
    return BenchmarkConfig(
        benchmark_version="0.1.0-smoke",
        benchmark_tag="SMOKE_TEST",
        max_host_images=2,
        methods=[
            "DCT",
        ],
        alpha_values=[
            10.0,
            20.0,
        ],
        watermark_sizes=[
            (32, 32),
        ],
        watermark_types=[
            "Binary Pattern",
            "Text",
        ],
        attacks=[
            AttackSpec(
                "none",
                None,
            ),
            AttackSpec(
                "jpeg",
                70,
            ),
        ],
        save_experiment_images=True,
        stop_on_error=True,
        strict_validation=True,
    )


def create_development_config() -> BenchmarkConfig:
    """
    Medium-sized development benchmark.
    """
    return BenchmarkConfig(
        benchmark_version="0.2.0-dev",
        benchmark_tag="DEVELOPMENT",
        max_host_images=10,
        methods=[
            "DCT",
        ],
        alpha_values=[
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
        ],
        watermark_sizes=[
            (32, 32),
            (64, 64),
        ],
        watermark_types=[
            "Binary Pattern",
            "Text",
            "Logo",
        ],
        attacks=[
            AttackSpec(
                "none",
                None,
            ),
            AttackSpec(
                "jpeg",
                90,
            ),
            AttackSpec(
                "jpeg",
                70,
            ),
            AttackSpec(
                "gaussian_noise",
                0.01,
            ),
            AttackSpec(
                "gaussian_blur",
                3,
            ),
        ],
        save_experiment_images=True,
        stop_on_error=False,
        strict_validation=True,
    )


def create_full_config() -> BenchmarkConfig:
    """
    Full benchmark profile.

    Additional methods can be added after they are implemented.
    """
    return BenchmarkConfig(
        benchmark_version="1.0.0",
        benchmark_tag="AWSRE_BENCH_FULL",
        max_host_images=100,
        methods=[
            "DCT",
        ],
        alpha_values=[
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
        ],
        watermark_sizes=[
            (32, 32),
            (64, 64),
        ],
        watermark_types=[
            "Binary Pattern",
            "Text",
            "Logo",
            "QR Code",
            "Signature",
        ],
        attacks=[
            AttackSpec(
                "none",
                None,
            ),
            AttackSpec(
                "jpeg",
                90,
            ),
            AttackSpec(
                "jpeg",
                70,
            ),
            AttackSpec(
                "jpeg",
                50,
            ),
            AttackSpec(
                "gaussian_noise",
                0.01,
            ),
            AttackSpec(
                "gaussian_noise",
                0.03,
            ),
            AttackSpec(
                "salt_pepper",
                0.05,
            ),
            AttackSpec(
                "gaussian_blur",
                3,
            ),
            AttackSpec(
                "rotation",
                5,
            ),
            AttackSpec(
                "crop",
                0.90,
            ),
            AttackSpec(
                "brightness",
                20,
            ),
            AttackSpec(
                "contrast",
                20,
            ),
            AttackSpec(
                "gamma",
                1.2,
            ),
            AttackSpec(
                "sharpen",
                1.0,
            ),
        ],
        save_experiment_images=False,
        stop_on_error=False,
        strict_validation=True,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    config = create_smoke_test_config()

    config.create_directories()

    saved_path = config.save_json()

    print("=" * 72)
    print("AWSRE BENCHMARK CONFIGURATION")
    print("=" * 72)

    print(
        "Project:",
        config.project_name,
    )

    print(
        "Version:",
        config.benchmark_version,
    )

    print(
        "Methods:",
        config.methods,
    )

    print(
        "Host images:",
        config.max_host_images,
    )

    print(
        "Expected experiments:",
        config.expected_experiment_count,
    )

    print(
        "Configuration saved to:",
        saved_path,
    )

    print("\n✅ Benchmark configuration test passed.")
