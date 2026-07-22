"""
AWSRE Watermarking Package
"""

from watermarking.base import (
    BaseWatermarker,
    EmbeddingResult,
    ExtractionResult,
)

from watermarking.metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    calculate_ber,
    calculate_correlation,
    evaluate_embedding,
)

from watermarking.registry import (
    create_watermarker,
    embed,
    extract,
    get_registry_info,
    is_registered,
    list_registered_methods,
    load_builtin_watermarkers,
    register_watermarker,
    registry_size,
    watermarker,
)

from watermarking.dct_svd import DCTSVDWatermarker

from watermarking.block_svd import (
    BlockSVDMetadata,
    BlockSVDWatermarker,
    embed_block_svd,
    extract_block_svd,
)

__all__ = [
    "BaseWatermarker",
    "EmbeddingResult",
    "ExtractionResult",
    "create_watermarker",
    "embed",
    "extract",
    "register_watermarker",
    "watermarker",
    "is_registered",
    "list_registered_methods",
    "registry_size",
    "get_registry_info",
    "calculate_mse",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_ber",
    "calculate_correlation",
    "evaluate_embedding",
    "DCTSVDWatermarker",
    "BlockSVDMetadata",
    "BlockSVDWatermarker",
    "embed_block_svd",
    "extract_block_svd",
]
