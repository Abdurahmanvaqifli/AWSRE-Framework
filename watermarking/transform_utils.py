"""
AWSRE Transform Utilities

Shared transform and image-processing utilities used by:

- DCTWatermarker
- DWTWatermarker
- DCTSVDWatermarker
- DWTSVDWatermarker
- BlockSVDWatermarker
- AWSRE-Bench

This module contains no Streamlit code and does not calculate
evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np
import pywt


# ============================================================
# TYPE ALIASES
# ============================================================

BlockCoordinate = Tuple[int, int, int, int]
Coefficient = Tuple[int, int]


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class DWTResult:
    """
    Result of a single-level 2D discrete wavelet transform.
    """

    ll: np.ndarray
    lh: np.ndarray
    hl: np.ndarray
    hh: np.ndarray
    wavelet: str
    original_shape: Tuple[int, int]


@dataclass(frozen=True)
class SVDResult:
    """
    Result of singular value decomposition.
    """

    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray


@dataclass(frozen=True)
class CapacityInfo:
    """
    Block-based embedding capacity information.
    """

    host_height: int
    host_width: int
    block_size: int
    available_blocks_y: int
    available_blocks_x: int
    available_bits: int
    requested_bits: int
    capacity_ratio: float
    fits: bool


# ============================================================
# ARRAY VALIDATION
# ============================================================

def validate_array(
    array: np.ndarray,
    *,
    name: str = "array",
    allowed_dimensions: Sequence[int] = (2, 3),
) -> None:
    """
    Validate a NumPy image-like array.
    """
    if array is None:
        raise ValueError(f"{name} cannot be None.")

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if array.ndim not in allowed_dimensions:
        raise ValueError(
            f"{name} must have dimensions {tuple(allowed_dimensions)}; "
            f"received ndim={array.ndim}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )


def validate_same_shape(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str = "first",
    second_name: str = "second",
) -> None:
    """
    Ensure two arrays have identical dimensions.
    """
    validate_array(first, name=first_name)
    validate_array(second, name=second_name)

    if first.shape != second.shape:
        raise ValueError(
            f"Shape mismatch: {first_name}={first.shape}, "
            f"{second_name}={second.shape}."
        )


def validate_block_size(block_size: int) -> int:
    """
    Validate and normalize block size.
    """
    try:
        normalized = int(block_size)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "block_size must be an integer."
        ) from exc

    if normalized <= 0:
        raise ValueError(
            "block_size must be greater than zero."
        )

    return normalized


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def ensure_float32(array: np.ndarray) -> np.ndarray:
    """
    Convert an array to float32 safely.
    """
    validate_array(array)

    return np.asarray(
        array,
        dtype=np.float32,
    )


def ensure_float64(array: np.ndarray) -> np.ndarray:
    """
    Convert an array to float64 safely.
    """
    validate_array(array)

    return np.asarray(
        array,
        dtype=np.float64,
    )


def ensure_uint8(array: np.ndarray) -> np.ndarray:
    """
    Clip values to [0, 255] and convert to uint8.
    """
    validate_array(array)

    if array.dtype == np.uint8:
        return array.copy()

    return np.uint8(
        np.clip(
            np.rint(array),
            0,
            255,
        )
    )


def clip_image(
    array: np.ndarray,
    minimum: float = 0.0,
    maximum: float = 255.0,
) -> np.ndarray:
    """
    Clip an image without changing its current dtype.
    """
    validate_array(array)

    return np.clip(
        array,
        minimum,
        maximum,
    )


def normalize_image(
    array: np.ndarray,
    output_min: float = 0.0,
    output_max: float = 255.0,
) -> np.ndarray:
    """
    Min-max normalize an array.
    """
    validate_array(array)

    array_float = array.astype(
        np.float32
    )

    minimum = float(
        np.min(array_float)
    )

    maximum = float(
        np.max(array_float)
    )

    if maximum - minimum <= 1e-12:
        return np.full_like(
            array_float,
            output_min,
            dtype=np.float32,
        )

    normalized = (
        array_float - minimum
    ) / (
        maximum - minimum
    )

    return (
        normalized
        * (output_max - output_min)
        + output_min
    ).astype(np.float32)


# ============================================================
# IMAGE CONVERSION
# ============================================================

def ensure_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert a grayscale, RGB or BGR-like image to grayscale.

    For three-channel input, OpenCV BGR ordering is assumed.
    """
    validate_array(image)

    if image.ndim == 2:
        return image.copy()

    if image.shape[2] != 3:
        raise ValueError(
            "Three-channel image required for grayscale conversion."
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def split_bgr_channels(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a BGR image into blue, green and red channels.
    """
    validate_array(
        image,
        allowed_dimensions=(3,),
    )

    if image.shape[2] != 3:
        raise ValueError(
            "BGR image must contain exactly three channels."
        )

    blue, green, red = cv2.split(
        image
    )

    return blue, green, red


def merge_bgr_channels(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
) -> np.ndarray:
    """
    Merge blue, green and red channels.
    """
    validate_same_shape(
        blue,
        green,
        first_name="blue",
        second_name="green",
    )

    validate_same_shape(
        blue,
        red,
        first_name="blue",
        second_name="red",
    )

    return cv2.merge(
        [blue, green, red]
    )


# ============================================================
# WATERMARK PREPARATION
# ============================================================

def prepare_binary_watermark(
    watermark: np.ndarray,
    *,
    threshold: float = 127.5,
) -> np.ndarray:
    """
    Convert watermark arrays to binary uint8 values 0 and 1.

    Supports:
    - 0/1 arrays
    - 0/255 arrays
    - grayscale arrays
    - three-channel images
    """
    validate_array(
        watermark,
        name="watermark",
    )

    grayscale = ensure_grayscale(
        watermark
    ).astype(np.float32)

    if float(np.max(grayscale)) <= 1.0:
        binary = grayscale >= 0.5
    else:
        binary = grayscale >= threshold

    return binary.astype(
        np.uint8
    )


def resize_binary_watermark(
    watermark: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a watermark while preserving binary values.

    Parameters
    ----------
    size:
        Output size as (width, height).
    """
    binary = prepare_binary_watermark(
        watermark
    )

    width = int(size[0])
    height = int(size[1])

    if width <= 0 or height <= 0:
        raise ValueError(
            "Watermark dimensions must be positive."
        )

    resized = cv2.resize(
        binary,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return prepare_binary_watermark(
        resized
    )


def flatten_watermark_bits(
    watermark: np.ndarray,
) -> np.ndarray:
    """
    Return binary watermark bits as a one-dimensional array.
    """
    return prepare_binary_watermark(
        watermark
    ).reshape(-1)


def reconstruct_watermark_bits(
    bits: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reconstruct a binary watermark from flattened bits.
    """
    bits_array = np.asarray(
        bits
    ).reshape(-1)

    height = int(shape[0])
    width = int(shape[1])

    if height <= 0 or width <= 0:
        raise ValueError(
            "Watermark shape must contain positive values."
        )

    expected_size = height * width

    if bits_array.size != expected_size:
        raise ValueError(
            f"Bit count mismatch: received {bits_array.size}, "
            f"expected {expected_size}."
        )

    return prepare_binary_watermark(
        bits_array.reshape(
            height,
            width,
        )
    )


# ============================================================
# BLOCK OPERATIONS
# ============================================================

def calculate_block_grid(
    image_shape: Tuple[int, int],
    block_size: int,
) -> Tuple[int, int]:
    """
    Calculate the number of complete blocks in both directions.
    """
    block_size = validate_block_size(
        block_size
    )

    height = int(image_shape[0])
    width = int(image_shape[1])

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image dimensions must be positive."
        )

    return (
        height // block_size,
        width // block_size,
    )


def iter_block_coordinates(
    image_shape: Tuple[int, int],
    block_size: int,
    *,
    max_blocks: Optional[int] = None,
) -> Iterator[BlockCoordinate]:
    """
    Yield complete non-overlapping block coordinates.

    Coordinates are returned as:

        y_start, y_end, x_start, x_end
    """
    blocks_y, blocks_x = calculate_block_grid(
        image_shape,
        block_size,
    )

    emitted = 0

    for row in range(blocks_y):
        for column in range(blocks_x):
            if (
                max_blocks is not None
                and emitted >= max_blocks
            ):
                return

            y_start = row * block_size
            x_start = column * block_size

            yield (
                y_start,
                y_start + block_size,
                x_start,
                x_start + block_size,
            )

            emitted += 1


def extract_block(
    image: np.ndarray,
    coordinate: BlockCoordinate,
) -> np.ndarray:
    """
    Extract one image block.
    """
    validate_array(
        image,
        name="image",
    )

    y_start, y_end, x_start, x_end = coordinate

    block = image[
        y_start:y_end,
        x_start:x_end,
    ]

    if block.size == 0:
        raise ValueError(
            f"Empty block extracted at {coordinate}."
        )

    return block.copy()


def insert_block(
    image: np.ndarray,
    block: np.ndarray,
    coordinate: BlockCoordinate,
) -> None:
    """
    Insert one block into an existing image in place.
    """
    validate_array(
        image,
        name="image",
    )

    validate_array(
        block,
        name="block",
    )

    y_start, y_end, x_start, x_end = coordinate

    expected_shape = (
        y_end - y_start,
        x_end - x_start,
    )

    if block.shape[:2] != expected_shape:
        raise ValueError(
            f"Block shape mismatch: {block.shape[:2]} "
            f"!= {expected_shape}."
        )

    image[
        y_start:y_end,
        x_start:x_end,
    ] = block


def split_into_blocks(
    image: np.ndarray,
    block_size: int,
    *,
    max_blocks: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Split an image into complete non-overlapping blocks.
    """
    coordinates = iter_block_coordinates(
        image.shape[:2],
        block_size,
        max_blocks=max_blocks,
    )

    return tuple(
        extract_block(
            image,
            coordinate,
        )
        for coordinate in coordinates
    )


def reconstruct_from_blocks(
    blocks: Sequence[np.ndarray],
    image_shape: Tuple[int, int],
    block_size: int,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Reconstruct an image from row-major non-overlapping blocks.
    """
    block_size = validate_block_size(
        block_size
    )

    height = int(image_shape[0])
    width = int(image_shape[1])

    output = np.zeros(
        (height, width),
        dtype=dtype,
    )

    coordinates = list(
        iter_block_coordinates(
            image_shape,
            block_size,
            max_blocks=len(blocks),
        )
    )

    if len(blocks) > len(coordinates):
        raise ValueError(
            "More blocks were supplied than the image can contain."
        )

    for block, coordinate in zip(
        blocks,
        coordinates,
    ):
        insert_block(
            output,
            block,
            coordinate,
        )

    return output


# ============================================================
# CAPACITY OPERATIONS
# ============================================================

def calculate_block_capacity(
    host_shape: Tuple[int, int],
    watermark_shape: Tuple[int, int],
    block_size: int,
) -> CapacityInfo:
    """
    Calculate whether one-bit-per-block embedding is possible.
    """
    block_size = validate_block_size(
        block_size
    )

    host_height = int(
        host_shape[0]
    )

    host_width = int(
        host_shape[1]
    )

    watermark_height = int(
        watermark_shape[0]
    )

    watermark_width = int(
        watermark_shape[1]
    )

    available_y, available_x = calculate_block_grid(
        (host_height, host_width),
        block_size,
    )

    available_bits = (
        available_y * available_x
    )

    requested_bits = (
        watermark_height
        * watermark_width
    )

    return CapacityInfo(
        host_height=host_height,
        host_width=host_width,
        block_size=block_size,
        available_blocks_y=available_y,
        available_blocks_x=available_x,
        available_bits=available_bits,
        requested_bits=requested_bits,
        capacity_ratio=(
            requested_bits
            / max(available_bits, 1)
        ),
        fits=requested_bits <= available_bits,
    )


def validate_block_capacity(
    host_shape: Tuple[int, int],
    watermark_shape: Tuple[int, int],
    block_size: int,
) -> CapacityInfo:
    """
    Validate one-bit-per-block capacity.
    """
    capacity = calculate_block_capacity(
        host_shape,
        watermark_shape,
        block_size,
    )

    if not capacity.fits:
        raise ValueError(
            "Host image does not have enough complete blocks. "
            f"Available bits: {capacity.available_bits}; "
            f"requested bits: {capacity.requested_bits}."
        )

    return capacity


def maximum_watermark_shape(
    host_shape: Tuple[int, int],
    block_size: int,
) -> Tuple[int, int]:
    """
    Return maximum watermark shape as (height, width).
    """
    return calculate_block_grid(
        host_shape,
        block_size,
    )


# ============================================================
# DCT OPERATIONS
# ============================================================

def apply_dct(
    block: np.ndarray,
) -> np.ndarray:
    """
    Apply the two-dimensional discrete cosine transform.
    """
    validate_array(
        block,
        name="DCT block",
        allowed_dimensions=(2,),
    )

    return cv2.dct(
        ensure_float32(block)
    )


def inverse_dct(
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Apply the inverse two-dimensional DCT.
    """
    validate_array(
        coefficients,
        name="DCT coefficients",
        allowed_dimensions=(2,),
    )

    return cv2.idct(
        ensure_float32(
            coefficients
        )
    )


def validate_coefficients(
    coefficients: Iterable[Coefficient],
    block_size: int,
) -> Tuple[Coefficient, ...]:
    """
    Validate transform coefficient coordinates.
    """
    block_size = validate_block_size(
        block_size
    )

    validated = []

    for item in coefficients:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
        ):
            raise ValueError(
                "Each coefficient must be a (row, column) tuple."
            )

        row = int(item[0])
        column = int(item[1])

        if not (
            0 <= row < block_size
            and 0 <= column < block_size
        ):
            raise ValueError(
                f"Coefficient {(row, column)} is outside "
                f"a {block_size}x{block_size} block."
            )

        validated.append(
            (row, column)
        )

    if not validated:
        raise ValueError(
            "At least one coefficient is required."
        )

    if len(set(validated)) != len(validated):
        raise ValueError(
            "Duplicate coefficient coordinates are not allowed."
        )

    return tuple(validated)


# ============================================================
# DWT OPERATIONS
# ============================================================

def apply_dwt(
    image: np.ndarray,
    *,
    wavelet: str = "haar",
    mode: str = "symmetric",
) -> DWTResult:
    """
    Apply a single-level two-dimensional wavelet transform.
    """
    validate_array(
        image,
        name="DWT image",
        allowed_dimensions=(2,),
    )

    image_float = ensure_float32(
        image
    )

    ll, (
        lh,
        hl,
        hh,
    ) = pywt.dwt2(
        image_float,
        wavelet=wavelet,
        mode=mode,
    )

    return DWTResult(
        ll=np.asarray(
            ll,
            dtype=np.float32,
        ),
        lh=np.asarray(
            lh,
            dtype=np.float32,
        ),
        hl=np.asarray(
            hl,
            dtype=np.float32,
        ),
        hh=np.asarray(
            hh,
            dtype=np.float32,
        ),
        wavelet=wavelet,
        original_shape=image.shape,
    )


def inverse_dwt(
    transform: DWTResult,
    *,
    mode: str = "symmetric",
) -> np.ndarray:
    """
    Reconstruct an image from a single-level DWT result.
    """
    if not isinstance(
        transform,
        DWTResult,
    ):
        raise TypeError(
            "transform must be a DWTResult."
        )

    reconstructed = pywt.idwt2(
        (
            transform.ll,
            (
                transform.lh,
                transform.hl,
                transform.hh,
            ),
        ),
        wavelet=transform.wavelet,
        mode=mode,
    )

    original_height = int(
        transform.original_shape[0]
    )

    original_width = int(
        transform.original_shape[1]
    )

    return np.asarray(
        reconstructed[
            :original_height,
            :original_width,
        ],
        dtype=np.float32,
    )


def replace_dwt_subband(
    transform: DWTResult,
    *,
    ll: Optional[np.ndarray] = None,
    lh: Optional[np.ndarray] = None,
    hl: Optional[np.ndarray] = None,
    hh: Optional[np.ndarray] = None,
) -> DWTResult:
    """
    Return a new DWT result with selected sub-bands replaced.
    """
    return DWTResult(
        ll=(
            transform.ll
            if ll is None
            else ensure_float32(ll)
        ),
        lh=(
            transform.lh
            if lh is None
            else ensure_float32(lh)
        ),
        hl=(
            transform.hl
            if hl is None
            else ensure_float32(hl)
        ),
        hh=(
            transform.hh
            if hh is None
            else ensure_float32(hh)
        ),
        wavelet=transform.wavelet,
        original_shape=transform.original_shape,
    )


# ============================================================
# SVD OPERATIONS
# ============================================================

def apply_svd(
    matrix: np.ndarray,
    *,
    full_matrices: bool = False,
) -> SVDResult:
    """
    Apply singular value decomposition.
    """
    validate_array(
        matrix,
        name="SVD matrix",
        allowed_dimensions=(2,),
    )

    u, s, vt = np.linalg.svd(
        ensure_float64(matrix),
        full_matrices=full_matrices,
    )

    return SVDResult(
        u=np.asarray(
            u,
            dtype=np.float64,
        ),
        s=np.asarray(
            s,
            dtype=np.float64,
        ),
        vt=np.asarray(
            vt,
            dtype=np.float64,
        ),
    )


def inverse_svd(
    transform: SVDResult,
) -> np.ndarray:
    """
    Reconstruct a matrix from an SVD result.
    """
    if not isinstance(
        transform,
        SVDResult,
    ):
        raise TypeError(
            "transform must be an SVDResult."
        )

    return (
        transform.u
        @ np.diag(transform.s)
        @ transform.vt
    )


def replace_singular_values(
    transform: SVDResult,
    singular_values: np.ndarray,
) -> SVDResult:
    """
    Return a new SVD result with modified singular values.
    """
    new_values = np.asarray(
        singular_values,
        dtype=np.float64,
    ).reshape(-1)

    if new_values.shape != transform.s.shape:
        raise ValueError(
            f"Singular-value shape mismatch: {new_values.shape} "
            f"!= {transform.s.shape}."
        )

    return SVDResult(
        u=transform.u.copy(),
        s=new_values.copy(),
        vt=transform.vt.copy(),
    )


def modify_primary_singular_value(
    transform: SVDResult,
    delta: float,
) -> SVDResult:
    """
    Add a delta to the largest singular value.
    """
    if transform.s.size == 0:
        raise ValueError(
            "SVD result contains no singular values."
        )

    singular_values = transform.s.copy()
    singular_values[0] += float(delta)

    # Singular values must remain non-negative.
    singular_values[0] = max(
        singular_values[0],
        0.0,
    )

    return replace_singular_values(
        transform,
        singular_values,
    )


# ============================================================
# FFT OPERATIONS
# ============================================================

def apply_fft(
    image: np.ndarray,
) -> np.ndarray:
    """
    Apply the two-dimensional fast Fourier transform.
    """
    validate_array(
        image,
        name="FFT image",
        allowed_dimensions=(2,),
    )

    return np.fft.fft2(
        ensure_float32(image)
    )


def inverse_fft(
    spectrum: np.ndarray,
) -> np.ndarray:
    """
    Apply the inverse two-dimensional FFT.
    """
    validate_array(
        spectrum,
        name="FFT spectrum",
        allowed_dimensions=(2,),
    )

    return np.real(
        np.fft.ifft2(
            spectrum
        )
    ).astype(np.float32)


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(
        seed=42
    )

    image = rng.integers(
        0,
        256,
        size=(512, 512),
        dtype=np.uint8,
    )

    watermark = rng.integers(
        0,
        2,
        size=(32, 32),
        dtype=np.uint8,
    )

    # DCT test
    block = image[:8, :8]
    dct_result = apply_dct(block)
    dct_reconstructed = inverse_dct(
        dct_result
    )

    dct_error = float(
        np.mean(
            np.abs(
                block.astype(np.float32)
                - dct_reconstructed
            )
        )
    )

    # DWT test
    dwt_result = apply_dwt(image)
    dwt_reconstructed = inverse_dwt(
        dwt_result
    )

    dwt_error = float(
        np.mean(
            np.abs(
                image.astype(np.float32)
                - dwt_reconstructed
            )
        )
    )

    # SVD test
    svd_result = apply_svd(block)
    svd_reconstructed = inverse_svd(
        svd_result
    )

    svd_error = float(
        np.mean(
            np.abs(
                block.astype(np.float64)
                - svd_reconstructed
            )
        )
    )

    capacity = validate_block_capacity(
        image.shape,
        watermark.shape,
        8,
    )

    print("=" * 72)
    print("AWSRE TRANSFORM UTILITIES SELF TEST")
    print("=" * 72)

    print(f"DCT reconstruction error: {dct_error:.10f}")
    print(f"DWT reconstruction error: {dwt_error:.10f}")
    print(f"SVD reconstruction error: {svd_error:.10f}")
    print(f"Available bits: {capacity.available_bits}")
    print(f"Requested bits: {capacity.requested_bits}")
    print(f"Capacity fits: {capacity.fits}")

    if dct_error > 1e-3:
        raise RuntimeError(
            "DCT reconstruction test failed."
        )

    if dwt_error > 1e-3:
        raise RuntimeError(
            "DWT reconstruction test failed."
        )

    if svd_error > 1e-8:
        raise RuntimeError(
            "SVD reconstruction test failed."
        )

    print("\n✅ Transform utilities self test passed.")
