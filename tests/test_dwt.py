"""AWSRE DWT Watermarker Test."""

import numpy as np

from watermarking.dwt import DWTWatermarker
from watermarking.metrics import (
    calculate_ber,
    calculate_correlation,
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
)
from watermarking.registry import (
    create_watermarker,
    list_registered_methods,
)


def run_test():
    print("=" * 72)
    print("AWSRE DWT WATERMARKER TEST")
    print("=" * 72)

    rng = np.random.default_rng(seed=2026)

    host = rng.integers(
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

    for subband in ("LL", "LH", "HL", "HH"):
        algorithm = DWTWatermarker(
            alpha=10,
            wavelet="haar",
            level=1,
            subband=subband,
        )

        embedding = algorithm.embed(host, watermark)

        extraction = algorithm.extract(
            host,
            embedding.watermarked_image,
            watermark.shape,
        )

        mse = calculate_mse(
            host,
            embedding.watermarked_image,
        )
        psnr = calculate_psnr(
            host,
            embedding.watermarked_image,
        )
        ssim = calculate_ssim(
            host,
            embedding.watermarked_image,
        )
        ber = calculate_ber(
            watermark,
            extraction.extracted_watermark,
        )
        correlation = calculate_correlation(
            watermark,
            extraction.extracted_watermark,
        )

        assert embedding.watermarked_image.shape == host.shape
        assert embedding.watermarked_image.dtype == np.uint8
        assert extraction.extracted_watermark.shape == watermark.shape
        assert extraction.extracted_watermark.dtype == np.uint8
        assert np.isfinite(mse)
        assert np.isfinite(psnr)
        assert np.isfinite(ssim)
        assert np.isfinite(ber)
        assert np.isfinite(correlation)
        assert ber == 0.0
        assert correlation == 1.0
        assert embedding.runtime >= 0.0
        assert extraction.runtime >= 0.0

        print(
            f"{subband}: "
            f"PSNR={psnr:.4f}, "
            f"SSIM={ssim:.6f}, "
            f"BER={ber:.8f}, "
            f"Corr={correlation:.6f}"
        )

    registry_algorithm = create_watermarker(
        method="DWT",
        alpha=10,
    )

    assert isinstance(
        registry_algorithm,
        DWTWatermarker,
    )

    registered = list_registered_methods()
    assert "DWT" in registered

    print("\nRegistered methods:", registered)
    print("\n✅ DWT WATERMARKER TEST PASSED")


if __name__ == "__main__":
    run_test()
