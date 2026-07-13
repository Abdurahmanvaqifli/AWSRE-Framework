"""
AWSRE Benchmark Watermark Generator Package
"""

from benchmark.watermark_generator.base_generator import (
    BaseWatermarkGenerator,
    GeneratedWatermark,
    WatermarkType,
)

from benchmark.watermark_generator.binary_pattern import (
    BinaryPatternGenerator,
    BinaryPatternType,
    generate_binary_pattern,
)

from benchmark.watermark_generator.text_generator import (
    TextFont,
    TextWatermarkGenerator,
    generate_text_watermark,
)

from benchmark.watermark_generator.qr_generator import (
    QRErrorCorrection,
    QRCodeWatermarkGenerator,
    generate_qr_watermark,
)
from benchmark.watermark_generator.image_based_generator import (
    ImageBasedWatermarkGenerator,
)

from benchmark.watermark_generator.logo_generator import (
    LogoWatermarkGenerator,
    generate_logo_watermark,
)

from benchmark.watermark_generator.signature_generator import (
    SignatureWatermarkGenerator,
    generate_signature_watermark,
)
__all__ = [
    "BaseWatermarkGenerator",
    "GeneratedWatermark",
    "WatermarkType",
    "BinaryPatternGenerator",
    "BinaryPatternType",
    "generate_binary_pattern",
    "TextFont",
    "TextWatermarkGenerator",
    "generate_text_watermark",
    "QRErrorCorrection",
    "QRCodeWatermarkGenerator",
    "generate_qr_watermark",
    "ImageBasedWatermarkGenerator",
    "LogoWatermarkGenerator",
    "generate_logo_watermark",
    "SignatureWatermarkGenerator",
    "generate_signature_watermark",
]
