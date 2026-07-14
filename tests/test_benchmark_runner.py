"""AWSRE benchmark runner end-to-end and resume test."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from benchmark.benchmark_runner import BenchmarkRunner
from benchmark.config import AttackSpec, BenchmarkConfig


def run_test():
    print("=" * 72)
    print("AWSRE BENCHMARK RUNNER END-TO-END TEST")
    print("=" * 72)

    rng = np.random.default_rng(seed=2026)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset_root = root / "datasets"
        hosts = dataset_root / "host_images"
        output_root = root / "outputs"
        hosts.mkdir(parents=True, exist_ok=True)

        for index in range(2):
            image = rng.integers(
                0,
                256,
                size=(512, 512),
                dtype=np.uint8,
            )
            assert cv2.imwrite(
                str(hosts / f"host_{index}.png"),
                image,
            )

        config = BenchmarkConfig(
            project_name="AWSRE-Test",
            benchmark_version="1.1.0-test",
            benchmark_tag="RUNNER_TEST",
            random_seed=42,
            max_host_images=2,
            host_target_size=(512, 512),
            methods=["DCT"],
            alpha_values=[10.0, 20.0],
            watermark_sizes=[(32, 32)],
            watermark_types=["Binary Pattern", "Text"],
            attacks=[
                AttackSpec("none", None),
                AttackSpec("jpeg", 70),
            ],
            save_every_experiment=True,
            backup_interval=4,
            checkpoint_interval=4,
            save_experiment_images=False,
            stop_on_error=True,
            resume_enabled=True,
            strict_validation=True,
            dataset_root=str(dataset_root),
            output_root=str(output_root),
        )

        first = BenchmarkRunner(config=config).run()
        assert first.progress.successful_experiments == 16
        assert first.progress.failed_experiments == 0
        assert first.storage_validation["valid"]

        csv_path = Path(first.csv_path)
        frame = pd.read_csv(csv_path)
        metrics = ["mse", "psnr", "ssim", "ber", "correlation"]

        assert len(frame) == 16
        assert frame[metrics].notna().all().all()
        assert np.isfinite(
            frame[metrics].to_numpy(dtype=float)
        ).all()
        assert frame["ber"].between(0.0, 1.0).all()

        clean = frame["attack_type"] == "none"
        assert clean.any()
        assert (frame.loc[clean, "ber"] == 0.0).all()

        resumed = BenchmarkRunner(config=config).run()
        assert resumed.progress.successful_experiments == 0
        assert resumed.progress.failed_experiments == 0
        assert resumed.progress.skipped_experiments == 16
        assert len(pd.read_csv(csv_path)) == 16

        print("\nExperiments:", len(frame))
        print("Clean zero-BER rows:", int(clean.sum()))
        print("Resume skipped:", resumed.progress.skipped_experiments)
        print("\n✅ BENCHMARK RUNNER END-TO-END TEST PASSED")


if __name__ == "__main__":
    run_test()
