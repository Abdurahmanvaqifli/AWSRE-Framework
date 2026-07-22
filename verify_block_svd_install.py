from pathlib import Path

import cv2
import numpy as np

from watermarking.block_svd import BlockSVDWatermarker


def calculate_psnr(
    original: np.ndarray,
    modified: np.ndarray,
) -> float:
    original_float = original.astype(np.float64)
    modified_float = modified.astype(np.float64)

    mse = np.mean(
        (original_float - modified_float) ** 2
    )

    if mse == 0:
        return float("inf")

    return float(
        10.0 * np.log10((255.0 ** 2) / mse)
    )


def calculate_ber(
    original_watermark: np.ndarray,
    extracted_watermark: np.ndarray,
) -> float:
    expected = original_watermark.reshape(-1)
    recovered = extracted_watermark.reshape(-1)

    return float(np.mean(expected != recovered))


def create_demo_image() -> np.ndarray:
    height = 256
    width = 256

    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]

    channel_1 = (rows + columns) % 256
    channel_2 = (2 * rows + columns) % 256
    channel_3 = (rows + 2 * columns) % 256

    return np.dstack(
        (channel_1, channel_2, channel_3)
    ).astype(np.uint8)


def main() -> None:
    output_directory = Path("outputs/block_svd")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    host_image = create_demo_image()

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
        alpha=40.0,
        repeat_watermark=True,
    )

    watermarked_image, metadata = watermarker.embed(
        host_image,
        watermark,
    )

    extracted_watermark = watermarker.extract(
        host_image,
        watermarked_image,
        metadata=metadata,
    )

    psnr = calculate_psnr(
        host_image,
        watermarked_image,
    )

    ber = calculate_ber(
        watermark,
        extracted_watermark,
    )

    cv2.imwrite(
        str(output_directory / "host.png"),
        host_image,
    )

    cv2.imwrite(
        str(output_directory / "watermarked.png"),
        watermarked_image,
    )

    watermark_visual = watermark * 255
    extracted_visual = extracted_watermark * 255

    cv2.imwrite(
        str(output_directory / "watermark.png"),
        watermark_visual,
    )

    cv2.imwrite(
        str(output_directory / "extracted.png"),
        extracted_visual,
    )

    print("=" * 60)
    print("BLOCK-SVD INSTALLATION VERIFICATION")
    print("=" * 60)
    print(f"Host shape        : {host_image.shape}")
    print(f"Watermark shape   : {watermark.shape}")
    print(f"Capacity          : {watermarker.capacity(host_image)} bits")
    print(f"Embedded bits     : {metadata['embedded_bits']}")
    print(f"Block size        : {metadata['block_size']}")
    print(f"Alpha             : {metadata['alpha']}")
    print(f"PSNR              : {psnr:.4f} dB")
    print(f"BER               : {ber:.6f}")
    print(
        "Embedding time    : "
        f"{watermarker.last_embedding_time:.6f} seconds"
    )
    print(
        "Extraction time   : "
        f"{watermarker.last_extraction_time:.6f} seconds"
    )
    print(f"Output directory  : {output_directory}")
    print("=" * 60)

    if ber <= 0.10:
        print("SUCCESS: Block-SVD is working.")
    else:
        raise RuntimeError(
            f"Verification failed because BER={ber:.6f}."
        )


if __name__ == "__main__":
    main()
