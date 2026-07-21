"""AWSRE DCT-SVD Watermarker Test."""

import cv2
import numpy as np

from watermarking.dct_svd import DCTSVDWatermarker
from watermarking.metrics import (
    calculate_ber,
    calculate_correlation,
    calculate_psnr,
    calculate_ssim,
)
from watermarking.registry import (
    create_watermarker,
    is_registered,
    list_registered_methods,
)


def run_test():
    print("=" * 72)
    print("AWSRE DCT-SVD WATERMARKER TEST")
    print("=" * 72)

    rng = np.random.default_rng(seed=2026)
    host = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)

    watermark = np.zeros((32, 32), dtype=np.uint8)
    cv2.putText(
        watermark,
        "A",
        (5, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        1,
        2,
    )

    algorithm = DCTSVDWatermarker(alpha=20)
    embedding_result = algorithm.embed(host, watermark)
    watermarked = embedding_result.watermarked_image

    extraction_result = algorithm.extract(host, watermarked, watermark.shape)
    extracted = extraction_result.extracted_watermark

    psnr = calculate_psnr(host, watermarked)
    ssim = calculate_ssim(host, watermarked)
    ber = calculate_ber(watermark, extracted)
    correlation = calculate_correlation(watermark, extracted)

    assert watermarked.shape == host.shape
    assert watermarked.dtype == np.uint8
    assert extracted.shape == watermark.shape
    assert extracted.dtype == np.uint8

    assert psnr > 25
    assert 0 <= ssim <= 1
    assert ber <= 0.02
    assert 0 <= correlation <= 1

    assert is_registered("DCT-SVD")
    assert "DCT-SVD" in list_registered_methods()

    factory_algorithm = create_watermarker(method="dct_svd", alpha=20)
    assert isinstance(factory_algorithm, DCTSVDWatermarker)

    assert embedding_result.metadata["method"] == "DCT-SVD"
    assert embedding_result.metadata["transform_pipeline"] == ["DCT", "SVD"]
    assert extraction_result.metadata["non_blind_extraction"] is True

    print(f"\nPSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.6f}")
    print(f"BER: {ber:.8f}")
    print(f"Correlation: {correlation:.6f}")
    print(f"Embedding runtime: {embedding_result.runtime:.6f} s")
    print(f"Extraction runtime: {extraction_result.runtime:.6f} s")
    print("Registered methods:", list_registered_methods())
    print("\n✅ DCT-SVD WATERMARKER TEST PASSED")


if __name__ == "__main__":
    run_test()
