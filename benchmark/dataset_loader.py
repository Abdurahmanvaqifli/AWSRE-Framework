"""
AWSRE Dataset Loader

Safely loads host images for AWSRE-Bench.

Responsibilities:
- scan a directory for supported image files;
- reject unreadable/corrupted files;
- resize images to a standard size;
- compute SHA-256 hashes;
- remove duplicates;
- return structured host-image records;
- provide clear loading statistics.

No Streamlit code is used here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib

import cv2
import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class HostImageItem:
    """
    One successfully loaded host image.
    """

    image_id: str
    file_name: str
    file_path: str
    file_hash: str

    original_width: int
    original_height: int
    processed_width: int
    processed_height: int

    channels: int
    extension: str
    dataset_name: str

    image: np.ndarray

    def metadata_dict(self) -> Dict[str, Any]:
        """
        Return serializable metadata without the NumPy image.
        """
        data = asdict(self)
        data.pop("image", None)
        return data


@dataclass
class DatasetLoadSummary:
    """
    Summary of one dataset-loading operation.
    """

    requested_limit: Optional[int]
    discovered_files: int
    successfully_loaded: int
    corrupted_files: int
    duplicate_files: int
    unsupported_files: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetLoadResult:
    """
    Complete dataset-loading result.
    """

    images: List[HostImageItem]
    summary: DatasetLoadSummary
    errors: List[Dict[str, str]]


# ============================================================
# HASHING
# ============================================================

def calculate_file_hash(
    file_path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 hash of a file.
    """
    path = Path(file_path)

    sha256 = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# DISCOVERY
# ============================================================

def discover_image_files(
    directory: str | Path,
    *,
    recursive: bool = True,
    supported_extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
) -> List[Path]:
    """
    Discover supported image files in a directory.
    """
    root = Path(directory)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: {root}"
        )

    normalized_extensions = {
        extension.lower()
        for extension in supported_extensions
    }

    iterator = (
        root.rglob("*")
        if recursive
        else root.glob("*")
    )

    files = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in normalized_extensions
    ]

    return sorted(
        files,
        key=lambda item: str(item).lower(),
    )


# ============================================================
# IMAGE READING
# ============================================================

def read_image(
    file_path: str | Path,
    *,
    color_mode: str = "grayscale",
) -> np.ndarray:
    """
    Read an image safely with OpenCV.

    color_mode:
    - grayscale
    - color
    """
    path = Path(file_path)

    normalized_mode = color_mode.strip().lower()

    if normalized_mode == "grayscale":
        flag = cv2.IMREAD_GRAYSCALE
    elif normalized_mode == "color":
        flag = cv2.IMREAD_COLOR
    else:
        raise ValueError(
            "color_mode must be 'grayscale' or 'color'."
        )

    image = cv2.imread(
        str(path),
        flag,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not read image: {path}"
        )

    if image.size == 0:
        raise ValueError(
            f"Image is empty: {path}"
        )

    return image


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize image to (width, height).
    """
    width = int(target_size[0])
    height = int(target_size[1])

    if width <= 0 or height <= 0:
        raise ValueError(
            "Target dimensions must be positive."
        )

    interpolation = (
        cv2.INTER_AREA
        if image.shape[1] > width
        or image.shape[0] > height
        else cv2.INTER_CUBIC
    )

    return cv2.resize(
        image,
        (width, height),
        interpolation=interpolation,
    )


# ============================================================
# ID CREATION
# ============================================================

def build_image_id(
    index: int,
    file_hash: str,
) -> str:
    """
    Build a deterministic readable host-image ID.
    """
    short_hash = file_hash[:10].upper()

    return (
        f"HOST-"
        f"{index:06d}-"
        f"{short_hash}"
    )


# ============================================================
# DATASET LOADING
# ============================================================

def load_host_dataset(
    directory: str | Path,
    *,
    target_size: Tuple[int, int] = (512, 512),
    max_images: Optional[int] = None,
    dataset_name: str = "Custom",
    color_mode: str = "grayscale",
    recursive: bool = True,
    remove_duplicates: bool = True,
    strict_count: bool = False,
) -> DatasetLoadResult:
    """
    Load host images from a directory.

    Parameters
    ----------
    directory:
        Dataset directory.
    target_size:
        Output image size as (width, height).
    max_images:
        Maximum number of valid images to load.
    dataset_name:
        Name stored in metadata.
    color_mode:
        'grayscale' or 'color'.
    recursive:
        Search nested directories.
    remove_duplicates:
        Ignore files with identical SHA-256 hashes.
    strict_count:
        Raise an error if fewer valid images than requested
        are loaded.

    Returns
    -------
    DatasetLoadResult
        Loaded images, summary and error information.
    """
    if max_images is not None:
        max_images = int(max_images)

        if max_images <= 0:
            raise ValueError(
                "max_images must be positive or None."
            )

    discovered = discover_image_files(
        directory,
        recursive=recursive,
    )

    loaded_images: List[HostImageItem] = []
    errors: List[Dict[str, str]] = []

    seen_hashes = set()

    corrupted_count = 0
    duplicate_count = 0

    for file_path in discovered:
        if (
            max_images is not None
            and len(loaded_images) >= max_images
        ):
            break

        try:
            file_hash = calculate_file_hash(
                file_path
            )

            if (
                remove_duplicates
                and file_hash in seen_hashes
            ):
                duplicate_count += 1
                continue

            image = read_image(
                file_path,
                color_mode=color_mode,
            )

            original_height = int(
                image.shape[0]
            )

            original_width = int(
                image.shape[1]
            )

            processed = resize_image(
                image,
                target_size,
            )

            processed_height = int(
                processed.shape[0]
            )

            processed_width = int(
                processed.shape[1]
            )

            channels = (
                1
                if processed.ndim == 2
                else int(processed.shape[2])
            )

            item = HostImageItem(
                image_id=build_image_id(
                    len(loaded_images) + 1,
                    file_hash,
                ),
                file_name=file_path.name,
                file_path=str(file_path),
                file_hash=file_hash,
                original_width=original_width,
                original_height=original_height,
                processed_width=processed_width,
                processed_height=processed_height,
                channels=channels,
                extension=file_path.suffix.lower(),
                dataset_name=dataset_name,
                image=processed,
            )

            loaded_images.append(item)
            seen_hashes.add(file_hash)

        except Exception as exc:
            corrupted_count += 1

            errors.append({
                "file_path": str(file_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

    if (
        strict_count
        and max_images is not None
        and len(loaded_images) < max_images
    ):
        raise RuntimeError(
            f"Requested {max_images} valid images, "
            f"but only {len(loaded_images)} were loaded."
        )

    summary = DatasetLoadSummary(
        requested_limit=max_images,
        discovered_files=len(discovered),
        successfully_loaded=len(loaded_images),
        corrupted_files=corrupted_count,
        duplicate_files=duplicate_count,
        unsupported_files=0,
    )

    return DatasetLoadResult(
        images=loaded_images,
        summary=summary,
        errors=errors,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_loaded_dataset(
    result: DatasetLoadResult,
    *,
    minimum_images: int = 1,
    expected_size: Optional[
        Tuple[int, int]
    ] = None,
) -> None:
    """
    Validate a loaded dataset before benchmark execution.
    """
    if not isinstance(
        result,
        DatasetLoadResult,
    ):
        raise TypeError(
            "result must be DatasetLoadResult."
        )

    if len(result.images) < minimum_images:
        raise RuntimeError(
            f"Dataset contains {len(result.images)} valid images; "
            f"minimum required: {minimum_images}."
        )

    image_ids = [
        item.image_id
        for item in result.images
    ]

    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError(
            "Duplicate image IDs detected."
        )

    hashes = [
        item.file_hash
        for item in result.images
    ]

    if len(hashes) != len(set(hashes)):
        raise RuntimeError(
            "Duplicate image hashes detected."
        )

    for item in result.images:
        if item.image is None:
            raise RuntimeError(
                f"Missing image array: {item.file_name}"
            )

        if item.image.size == 0:
            raise RuntimeError(
                f"Empty image array: {item.file_name}"
            )

        if expected_size is not None:
            expected_width = int(
                expected_size[0]
            )

            expected_height = int(
                expected_size[1]
            )

            if (
                item.processed_width != expected_width
                or item.processed_height
                != expected_height
            ):
                raise RuntimeError(
                    f"Unexpected image size for "
                    f"{item.file_name}: "
                    f"{item.processed_width}x"
                    f"{item.processed_height}."
                )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def dataset_metadata_rows(
    result: DatasetLoadResult,
) -> List[Dict[str, Any]]:
    """
    Return serializable metadata rows.
    """
    return [
        item.metadata_dict()
        for item in result.images
    ]


def print_dataset_summary(
    result: DatasetLoadResult,
) -> None:
    """
    Print dataset loading summary.
    """
    summary = result.summary

    print("=" * 72)
    print("AWSRE DATASET LOADER SUMMARY")
    print("=" * 72)

    print(
        "Requested limit:",
        summary.requested_limit,
    )

    print(
        "Discovered files:",
        summary.discovered_files,
    )

    print(
        "Successfully loaded:",
        summary.successfully_loaded,
    )

    print(
        "Corrupted/unreadable:",
        summary.corrupted_files,
    )

    print(
        "Duplicates skipped:",
        summary.duplicate_files,
    )

    print(
        "Errors recorded:",
        len(result.errors),
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temporary_path = Path(temp_dir)

        rng = np.random.default_rng(
            seed=42
        )

        for index in range(3):
            image = rng.integers(
                0,
                256,
                size=(300, 400),
                dtype=np.uint8,
            )

            cv2.imwrite(
                str(
                    temporary_path
                    / f"sample_{index}.png"
                ),
                image,
            )

        result = load_host_dataset(
            temporary_path,
            target_size=(512, 512),
            max_images=2,
            dataset_name="SELF_TEST",
            color_mode="grayscale",
            strict_count=True,
        )

        validate_loaded_dataset(
            result,
            minimum_images=2,
            expected_size=(512, 512),
        )

        print_dataset_summary(
            result
        )

        if len(result.images) != 2:
            raise RuntimeError(
                "Dataset loader self test failed."
            )

        print(
            "\n✅ Dataset loader self test passed."
        )
