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
]
