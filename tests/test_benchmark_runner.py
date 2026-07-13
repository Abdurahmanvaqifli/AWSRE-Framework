"""
AWSRE Benchmark Runner End-to-End Test
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from benchmark.benchmark_runner import (
    BenchmarkRunner,
)
from benchmark.config import (
    AttackSpec,
    BenchmarkConfig,
)


def run_test():
    print("=" * 72)
    print("AWSRE BENCHMARK RUNNER END-TO-END TEST")
    print("=" * 72)

    rng = np.random.default_rng(
        seed=2026
    )

    with tempfile.TemporaryDirectory() as temp_directory:
        root = Path(
            temp_directory
        )

        dataset_root = (
            root / "datasets"
        )

        host_directory = (
            dataset_root / "host_images"
        )

        output_root = (
            root / "outputs"
        )

        host_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index in range(2):
            image = rng.integers(
                0,
                256,
                size=(512, 512),
                dtype=np.uint8,
            )

            cv2.imwrite(
                str(
                    host_directory
                    / f"host_{index}.png"
                ),
                image,
            )

        config = BenchmarkConfig(
            project_name="AWSRE-Test",
            benchmark_version=(
                "0.1.0-runner-test"
            ),
            benchmark_tag="RUNNER_TEST",
            random_seed=42,
            max_host_images=2,
            host_target_size=(512, 512),
            methods=["DCT"],
            alpha_values=[10.0, 20.0],
            watermark_sizes=[(32, 32)],
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
            save_every_experiment=True,
            backup_interval=4,
            checkpoint_interval=4,
            save_experiment_images=False,
            stop_on_error=True,
            resume_enabled=True,
            strict_validation=True,
            dataset_root=str(
                dataset_root
            ),
            output_root=str(
                output_root
            ),
        )

        assert (
            config.expected_experiment_count
            == 16
        )

        runner = BenchmarkRunner(
            config=config
        )

        result = runner.run()

        assert (
            result.progress
            .successful_experiments
            == 16
        )

        assert (
            result.progress
            .failed_experiments
            == 0
        )

        assert result.storage_validation[
            "valid"
        ]

        csv_path = Path(
            result.csv_path
        )

        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

        dataframe = pd.read_csv(
            csv_path
        )

        assert len(dataframe) == 16
        assert dataframe[
            "experiment_id"
        ].nunique() == 16

        assert set(
            dataframe["method"]
        ) == {
            "DCT"
        }

        assert set(
            dataframe["attack_type"]
        ) == {
            "none",
            "jpeg",
        }

        assert set(
            dataframe["watermark_type"]
        ) == {
            "Binary Pattern",
            "Text",
        }

        assert dataframe[
            "psnr"
        ].notna().all()

        assert dataframe[
            "ssim"
        ].notna().all()

        assert dataframe[
            "ber"
        ].notna().all()

        assert dataframe[
            "correlation"
        ].notna().all()

        # Resume test.
        resumed_runner = BenchmarkRunner(
            config=config
        )

        resumed_result = (
            resumed_runner.run()
        )

        assert (
            resumed_result.progress
            .successful_experiments
            == 0
        )

        assert (
            resumed_result.progress
            .skipped_experiments
            == 16
        )

        resumed_dataframe = pd.read_csv(
            csv_path
        )

        assert len(
            resumed_dataframe
        ) == 16

        print(
            "\nExperiments:",
            len(dataframe),
        )

        print(
            "Successful:",
            result.progress
            .successful_experiments,
        )

        print(
            "Failed:",
            result.progress
            .failed_experiments,
        )

        print(
            "Resume skipped:",
            resumed_result.progress
            .skipped_experiments,
        )

        print(
            "CSV:",
            csv_path,
        )

        print(
            "\n✅ BENCHMARK RUNNER END-TO-END TEST PASSED"
        )


if __name__ == "__main__":
    run_test()
