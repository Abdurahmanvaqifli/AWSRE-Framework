"""
AWSRE Transform Utilities Test
"""

import numpy as np

from watermarking.transform_utils import (
    apply_dct,
    apply_dwt,
    apply_fft,
    apply_svd,
    calculate_block_capacity,
    flatten_watermark_bits,
    inverse_dct,
    inverse_dwt,
    inverse_fft,
    inverse_svd,
    iter_block_coordinates,
    maximum_watermark_shape,
    prepare_binary_watermark,
    reconstruct_watermark_bits,
    resize_binary_watermark,
    split_into_blocks,
    validate_block_capacity,
)


def run_test():
    print("=" * 72)
    print("AWSRE TRANSFORM UTILITIES TEST")
    print("=" * 72)

    rng = np.random.default_rng(
        seed=2026
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

    # Binary watermark tests
    binary = prepare_binary_watermark(
        watermark
    )

    resized = resize_binary_watermark(
        binary,
        (64, 64),
    )

    bits = flatten_watermark_bits(
        binary
    )

    reconstructed_watermark = reconstruct_watermark_bits(
        bits,
        binary.shape,
    )

    assert binary.shape == (32, 32)
    assert resized.shape == (64, 64)
    assert np.array_equal(
        binary,
        reconstructed_watermark,
    )

    # Block tests
    blocks = split_into_blocks(
        image,
        block_size=8,
    )

    coordinates = list(
        iter_block_coordinates(
            image.shape,
            block_size=8,
        )
    )

    assert len(blocks) == 4096
    assert len(coordinates) == 4096
    assert blocks[0].shape == (8, 8)

    capacity = calculate_block_capacity(
        image.shape,
        binary.shape,
        block_size=8,
    )

    validated_capacity = validate_block_capacity(
        image.shape,
        binary.shape,
        block_size=8,
    )

    assert capacity.fits
    assert validated_capacity.fits
    assert capacity.available_bits == 4096
    assert capacity.requested_bits == 1024

    assert maximum_watermark_shape(
        image.shape,
        8,
    ) == (64, 64)

    # DCT test
    test_block = image[:8, :8]

    dct = apply_dct(
        test_block
    )

    dct_inverse = inverse_dct(
        dct
    )

    dct_error = np.mean(
        np.abs(
            test_block.astype(np.float32)
            - dct_inverse
        )
    )

    assert dct_error < 1e-3

    # DWT test
    dwt = apply_dwt(
        image
    )

    dwt_inverse = inverse_dwt(
        dwt
    )

    dwt_error = np.mean(
        np.abs(
            image.astype(np.float32)
            - dwt_inverse
        )
    )

    assert dwt_error < 1e-3

    # SVD test
    svd = apply_svd(
        test_block
    )

    svd_inverse = inverse_svd(
        svd
    )

    svd_error = np.mean(
        np.abs(
            test_block.astype(np.float64)
            - svd_inverse
        )
    )

    assert svd_error < 1e-8

    # FFT test
    fft = apply_fft(
        test_block
    )

    fft_inverse = inverse_fft(
        fft
    )

    fft_error = np.mean(
        np.abs(
            test_block.astype(np.float32)
            - fft_inverse
        )
    )

    assert fft_error < 1e-3

    print(f"\nDCT error: {dct_error:.10f}")
    print(f"DWT error: {dwt_error:.10f}")
    print(f"SVD error: {svd_error:.10f}")
    print(f"FFT error: {fft_error:.10f}")
    print(
        "Maximum watermark shape:",
        maximum_watermark_shape(
            image.shape,
            8,
        ),
    )

    print("\n✅ TRANSFORM UTILITIES TEST PASSED")


if __name__ == "__main__":
    run_test()
