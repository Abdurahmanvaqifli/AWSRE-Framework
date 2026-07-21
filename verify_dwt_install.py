"""AWSRE DWT installation verification."""

import importlib
import inspect

import numpy as np


def run_verification():
    print("=" * 72)
    print("AWSRE DWT INSTALLATION VERIFICATION")
    print("=" * 72)

    module = importlib.import_module(
        "watermarking.dwt"
    )

    DWTWatermarker = getattr(
        module,
        "DWTWatermarker",
    )

    print(
        "Module:",
        inspect.getfile(DWTWatermarker),
    )

    algorithm = DWTWatermarker(
        alpha=10,
        wavelet="haar",
        subband="LL",
    )

    rng = np.random.default_rng(seed=42)

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

    embedded = algorithm.embed(host, watermark)

    extracted = algorithm.extract(
        host,
        embedded.watermarked_image,
        watermark.shape,
    )

    ber = float(
        np.mean(
            watermark
            != extracted.extracted_watermark
        )
    )

    print("BER:", ber)
    print("Metadata:", embedded.metadata)

    assert ber == 0.0

    print("\n✅ DWT installation verification passed.")


if __name__ == "__main__":
    run_verification()
