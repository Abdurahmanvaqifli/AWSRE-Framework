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

from benchmark.watermark_generator.registry import (
    create_generator,
    generate_watermark,
    generator_registry_diagnostics,
    generator_registry_size,
    get_generator_class,
    get_generator_registry_info,
    is_generator_registered,
    list_registered_generators,
    load_builtin_generators,
    register_generator,
    watermark_generator,
)

from benchmark.watermark_generator.registry import (
    create_generator,
    generate_watermark,
    generator_registry_diagnostics,
    generator_registry_size,
    get_generator_class,
    get_generator_registry_info,
    is_generator_registered,
    list_registered_generators,
    load_builtin_generators,
    register_generator,
    watermark_generator,
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
    "create_generator",
    "generate_watermark",
    "register_generator",
    "watermark_generator",
    "load_builtin_generators",
    "is_generator_registered",
    "get_generator_class",
    "list_registered_generators",
    "generator_registry_size",
    "get_generator_registry_info",
    "generator_registry_diagnostics",
]
load_builtin_generators()
