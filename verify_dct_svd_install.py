"""Verify AWSRE DCT-SVD installation and registry integration."""

from watermarking.dct_svd import DCTSVDWatermarker
from watermarking.registry import create_watermarker, list_registered_methods


def main():
    algorithm = create_watermarker(method="DCT-SVD", alpha=20)

    assert isinstance(algorithm, DCTSVDWatermarker)
    assert "DCT-SVD" in list_registered_methods()

    print("✅ DCT-SVD installation verified")
    print("Registered methods:", list_registered_methods())
    print("Algorithm info:", algorithm.info())


if __name__ == "__main__":
    main()
