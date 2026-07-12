"""
AWSRE Dataset Loader Test
"""

from pathlib import Path
import tempfile

import cv2
import numpy as np

from benchmark.dataset_loader import (
    dataset_metadata_rows,
    load_host_dataset,
    validate_loaded_dataset,
)


def run_test():
    print("=" * 72)
    print("AWSRE DATASET LOADER TEST")
    print("=" * 72)

    rng = np.random.default_rng(
        seed=2026
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        image_1 = rng.integers(
            0,
            256,
            size=(320, 480),
            dtype=np.uint8,
        )

        image_2 = rng.integers(
            0,
            256,
            size=(600, 400),
            dtype=np.uint8,
        )

        cv2.imwrite(
            str(root / "image_1.png"),
            image_1,
        )

        cv2.imwrite(
            str(root / "image_2.jpg"),
            image_2,
        )

        # Duplicate file
        cv2.imwrite(
            str(root / "duplicate.png"),
            image_1,
        )

        # Corrupted image-like file
        (root / "broken.jpg").write_text(
            "not a real image",
            encoding="utf-8",
        )

        result = load_host_dataset(
            root,
            target_size=(512, 512),
            max_images=2,
            dataset_name="UNIT_TEST",
            color_mode="grayscale",
            remove_duplicates=True,
            strict_count=True,
        )

        validate_loaded_dataset(
            result,
            minimum_images=2,
            expected_size=(512, 512),
        )

        assert len(result.images) == 2
        assert result.summary.successfully_loaded == 2
        assert result.summary.duplicate_files >= 1
        assert result.summary.corrupted_files >= 1

        metadata = dataset_metadata_rows(
            result
        )

        assert len(metadata) == 2
        assert metadata[0]["processed_width"] == 512
        assert metadata[0]["processed_height"] == 512
        assert metadata[0]["dataset_name"] == "UNIT_TEST"

        for item in result.images:
            assert item.image.shape == (512, 512)
            assert len(item.file_hash) == 64
            assert item.image_id.startswith("HOST-")

        print(
            "\nSuccessfully loaded:",
            result.summary.successfully_loaded,
        )

        print(
            "Duplicates skipped:",
            result.summary.duplicate_files,
        )

        print(
            "Corrupted detected:",
            result.summary.corrupted_files,
        )

        print(
            "\n✅ DATASET LOADER TEST PASSED"
        )


if __name__ == "__main__":
    run_test()
