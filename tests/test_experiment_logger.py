"""
AWSRE Experiment Logger Test
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from benchmark.experiment_logger import (
    ExperimentLogger,
    ExperimentLoggerConfig,
    utc_now_iso,
)


def build_test_record(
    index: int,
) -> dict:
    experiment_id = (
        f"EXP-UNIT-{index:04d}"
    )

    experiment_key = (
        f"HOST-{index}|WM-{index}|"
        f"DCT|20|jpeg|70"
    )

    return {
        "experiment_id": experiment_id,
        "experiment_key": experiment_key,
        "timestamp_utc": utc_now_iso(),
        "benchmark_version": "1.0-test",
        "benchmark_tag": "UNIT_TEST",

        "host": {
            "image_id": f"HOST-{index}",
            "file_name": (
                f"host_{index}.png"
            ),
            "file_hash": (
                f"host_hash_{index}"
            ),
            "width": 512,
            "height": 512,
            "benchmark_tag": "Synthetic",
        },

        "watermark": {
            "watermark_id": (
                f"WM-{index}"
            ),
            "watermark_type": (
                "Binary Pattern"
            ),
            "file_hash": (
                f"wm_hash_{index}"
            ),
            "width": 32,
            "height": 32,
            "payload_bits": 1024,
            "density": 0.5,
        },

        "embedding": {
            "method": "DCT",
            "alpha": 20,
            "watermark_width": 32,
            "watermark_height": 32,
            "channel": "Grayscale",
            "block_size": 8,
            "decomposition_level": 1,
            "subband": None,
        },

        "attack": {
            "attack_type": "jpeg",
            "parameter": 70,
            "steps": [],
        },

        "imperceptibility": {
            "mse": 0.25,
            "psnr": 48.0,
            "ssim": 0.998,
        },

        "robustness": {
            "ber": 0.01,
            "correlation": 0.98,
            "extracted_successfully": True,
        },

        "runtime": {
            "embedding_time": 0.02,
            "attack_time": 0.01,
            "extraction_time": 0.02,
        },

        "prediction": {
            "expected_psnr": 47.5,
            "expected_ssim": 0.997,
            "expected_ber": 0.02,
            "expected_correlation": 0.97,
            "confidence": 80.0,
            "prediction_source": (
                "Strategy Prior"
            ),
        },

        "status": "SUCCESS",
        "notes": "Unit-test record",
    }


def run_test():
    print("=" * 72)
    print("AWSRE EXPERIMENT LOGGER TEST")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as temp_directory:
        root = Path(
            temp_directory
        )

        config = ExperimentLoggerConfig(
            output_root=root,
            backup_interval=2,
            checkpoint_interval=2,
            resume_enabled=True,
        )

        logger = ExperimentLogger(
            config
        )

        records = [
            build_test_record(index)
            for index in range(
                1,
                4,
            )
        ]

        for record in records:
            flattened = logger.log_experiment(
                record
            )

            assert (
                flattened["experiment_id"]
                == record["experiment_id"]
            )

            assert (
                flattened["method"]
                == "DCT"
            )

            assert (
                flattened["attack_type"]
                == "jpeg"
            )

        assert config.csv_path.exists()
        assert config.errors_path.exists()
        assert config.checkpoint_path.exists()
        assert (
            config.latest_backup_path
            .exists()
        )

        dataframe = pd.read_csv(
            config.csv_path
        )

        assert len(dataframe) == 3
        assert set(
            dataframe["status"]
        ) == {
            "SUCCESS"
        }

        assert dataframe[
            "experiment_id"
        ].nunique() == 3

        for record in records:
            json_path = (
                logger.experiment_json_path(
                    record["experiment_id"]
                )
            )

            assert json_path.exists()
            assert json_path.stat().st_size > 0

            saved = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )

            assert (
                saved["experiment_id"]
                == record[
                    "experiment_id"
                ]
            )

            assert logger.already_completed(
                record["experiment_key"]
            )

        error_record = {
            "experiment_id": (
                "EXP-ERROR-0001"
            ),
            "experiment_key": (
                "ERROR-KEY"
            ),
            "host_image_name": (
                "broken.png"
            ),
            "watermark_type": (
                "Logo"
            ),
            "method": "DCT",
            "alpha": 20,
            "attack_type": "jpeg",
            "attack_parameter": 70,
            "error_type": "ValueError",
            "error_message": (
                "Synthetic logger error"
            ),
            "traceback": (
                "Synthetic traceback"
            ),
        }

        logger.log_error(
            error_record
        )

        errors_dataframe = pd.read_csv(
            config.errors_path
        )

        assert len(
            errors_dataframe
        ) == 1

        validation = (
            logger.validate_storage(
                minimum_rows=3
            )
        )

        assert validation["valid"]
        assert validation[
            "csv_rows"
        ] == 3

        checkpoint = (
            logger.load_checkpoint()
        )

        assert checkpoint[
            "completed_count"
        ] >= 2

        # Resume check with a new logger instance.
        resumed_logger = ExperimentLogger(
            config
        )

        assert (
            resumed_logger.completed_count
            == 3
        )

        for record in records:
            assert (
                resumed_logger
                .already_completed(
                    record[
                        "experiment_key"
                    ]
                )
            )

        print(
            "\nCSV rows:",
            validation["csv_rows"],
        )

        print(
            "JSON files:",
            validation["json_files"],
        )

        print(
            "Completed keys:",
            resumed_logger.completed_count,
        )

        print(
            "Error rows:",
            len(errors_dataframe),
        )

        print(
            "Backup exists:",
            config.latest_backup_path.exists(),
        )

        print(
            "\n✅ EXPERIMENT LOGGER TEST PASSED"
        )


if __name__ == "__main__":
    run_test()
