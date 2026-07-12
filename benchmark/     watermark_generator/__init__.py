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

__all__ = [
    "BaseWatermarkGenerator",
    "GeneratedWatermark",
    "WatermarkType",
    "BinaryPatternGenerator",
    "BinaryPatternType",
    "generate_binary_pattern",
]
