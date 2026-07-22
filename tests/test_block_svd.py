import numpy as np
import pytest

from watermarking.block_svd import (
    BlockSVDMetadata,
    BlockSVDWatermarker,
    embed_block_svd,
    extract_block_svd,
)


def create_test_image(
    height: int = 256,
    width: int = 256,
) -> np.ndarray:
    """Create a deterministic grayscale test image."""

    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]

    image = (
        rows
        + columns
        + ((rows // 8) * 5)
        + ((columns // 8) * 3)
    ) % 256

    return image.astype(np.uint8)


def create_color_test_image(
    height: int = 256,
    width: int = 256,
) -> np.ndarray:
    """Create a deterministic BGR test image."""

    gray = create_test_image(height, width)

    return np.dstack(
        (
            gray,
            np.roll(gray, 20, axis=0),
            np.roll(gray, 30, axis=1),
        )
    )


def calculate_ber(
    expected: np.ndarray,
    recovered: np.ndarray,
) -> float:
    """Calculate bit error rate."""

    expected_bits = expected.astype(np.uint8).reshape(-1)
    recovered_bits = recovered.astype(np.uint8).reshape(-1)

    return float(np.mean(expected_bits != recovered_bits))


def test_grayscale_embedding_and_extraction():
    image = create_test_image()

    watermark = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )

    watermarker = BlockSVDWatermarker(
        block_size=8,
        alpha=30.0,
        repeat_watermark=True,
    )

    watermarked, metadata = watermarker.embed(
        image,
        watermark,
    )

    recovered = watermarker.extract(
        image,
        watermarked,
        metadata=metadata,
    )

    assert watermarked.shape == image.shape
    assert watermarked.dtype == np.uint8
    assert recovered.shape == watermark.shape
    assert calculate_ber(watermark, recovered) <= 0.05


def test_color_embedding_and_extraction():
    image = create_color_test_image()

    watermark = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    watermarker = BlockSVDWatermarker(
        block_size=8,
        alpha=40.0,
        repeat_watermark=True,
    )

    watermarked, metadata = watermarker.embed(
        image,
        watermark,
    )

    recovered = watermarker.extract(
        image,
        watermarked,
        metadata=metadata,
    )

    assert watermarked.shape == image.shape
    assert watermarked.dtype == np.uint8
    assert recovered.shape == watermark.shape
    assert calculate_ber(watermark, recovered) <= 0.10


def test_capacity():
    image = np.zeros((256, 256), dtype=np.uint8)

    watermarker = BlockSVDWatermarker(block_size=8)

    assert watermarker.capacity(image) == 1024


def test_metadata_serialization():
    metadata = BlockSVDMetadata(
        algorithm="Block-SVD",
        watermark_shape=(4, 4),
        watermark_length=16,
        host_shape=(256, 256),
        block_size=8,
        alpha=20.0,
        channel="grayscale",
        usable_blocks=1024,
        embedded_bits=1024,
        repeated=True,
        embedding_time_seconds=0.1,
    )

    restored = BlockSVDMetadata.from_dict(
        metadata.to_dict()
    )

    assert restored.algorithm == metadata.algorithm
    assert restored.watermark_shape == metadata.watermark_shape
    assert restored.watermark_length == metadata.watermark_length
    assert restored.block_size == metadata.block_size


def test_functional_api():
    image = create_test_image()

    watermark = np.array(
        [1, 0, 1, 1, 0, 0, 1, 0],
        dtype=np.uint8,
    )

    watermarked, metadata = embed_block_svd(
        image,
        watermark,
        alpha=30.0,
    )

    recovered = extract_block_svd(
        image,
        watermarked,
        metadata=metadata,
    )

    assert np.array_equal(recovered, watermark)


def test_watermark_capacity_error():
    image = np.zeros((16, 16), dtype=np.uint8)

    # 16x16 with 8x8 blocks has a capacity of only 4 bits.
    watermark = np.ones(5, dtype=np.uint8)

    watermarker = BlockSVDWatermarker(block_size=8)

    with pytest.raises(ValueError, match="capacity"):
        watermarker.embed(image, watermark)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        BlockSVDWatermarker(alpha=0)

    with pytest.raises(ValueError):
        BlockSVDWatermarker(alpha=-10)


def test_original_and_watermarked_shape_mismatch():
    original = np.zeros((128, 128), dtype=np.uint8)
    watermarked = np.zeros((256, 256), dtype=np.uint8)

    watermarker = BlockSVDWatermarker()

    with pytest.raises(ValueError, match="same shape"):
        watermarker.extract(
            original,
            watermarked,
            watermark_length=16,
        )


def test_input_image_is_not_modified():
    image = create_test_image()
    image_before = image.copy()

    watermark = np.array(
        [1, 0, 1, 0],
        dtype=np.uint8,
    )

    watermarker = BlockSVDWatermarker(alpha=30)

    watermarker.embed(image, watermark)

    assert np.array_equal(image, image_before)


def test_one_dimensional_watermark_shape():
    image = create_test_image()

    watermark = np.array(
        [1, 0, 0, 1, 1, 0, 1, 0],
        dtype=np.uint8,
    )

    watermarker = BlockSVDWatermarker(alpha=30)

    watermarked, metadata = watermarker.embed(
        image,
        watermark,
    )

    recovered = watermarker.extract(
        image,
        watermarked,
        metadata=metadata,
    )

    assert recovered.shape == watermark.shape
    assert calculate_ber(watermark, recovered) <= 0.05
