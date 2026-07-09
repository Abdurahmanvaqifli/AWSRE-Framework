"""
AWSRE Strategy Library Module

This module defines the knowledge base of supported watermarking strategies.

It contains:
- Strategy data classes
- Supported watermarking methods
- Alpha ranges
- Recommended watermark sizes
- suitability profiles
- expected performance priors
- explanation templates

No Streamlit code is used here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PerformancePrior:
    expected_psnr: float
    expected_ber: float
    expected_correlation: float
    runtime_score: float
    capacity_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategySuitability:
    preferred_entropy: List[str]
    preferred_texture: List[str]
    preferred_edge_density: List[str]
    preferred_complexity: List[str]
    suitable_host_profiles: List[str]
    suitable_objectives: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatermarkStrategy:
    method: str
    display_name: str
    description: str
    advantages: List[str]
    disadvantages: List[str]
    alpha_range: Tuple[int, int]
    alpha_step: int
    supported_watermark_sizes: List[Tuple[int, int]]
    typical_runtime: str
    computational_complexity: str
    imperceptibility_level: str
    robustness_level: str
    capacity_level: str
    performance_prior: PerformancePrior
    suitability: StrategySuitability
    explanation_templates: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    def alpha_values(self) -> List[int]:
        start, end = self.alpha_range
        return list(range(start, end + 1, self.alpha_step))

    def supports_size(self, size: Tuple[int, int]) -> bool:
        return size in self.supported_watermark_sizes

    def is_suitable_for_objective(self, objective: str) -> bool:
        return objective in self.suitability.suitable_objectives


# ============================================================
# OBJECTIVES
# ============================================================

OBJECTIVES = [
    "Maximum Image Quality",
    "Maximum Robustness",
    "Balanced",
    "Maximum Capacity",
    "Fast Embedding",
]


OBJECTIVE_PROFILES = {
    "Maximum Image Quality": {
        "psnr": 0.42,
        "ber": 0.12,
        "correlation": 0.18,
        "runtime": 0.08,
        "capacity": 0.20,
    },
    "Maximum Robustness": {
        "psnr": 0.15,
        "ber": 0.35,
        "correlation": 0.32,
        "runtime": 0.05,
        "capacity": 0.13,
    },
    "Balanced": {
        "psnr": 0.25,
        "ber": 0.25,
        "correlation": 0.25,
        "runtime": 0.10,
        "capacity": 0.15,
    },
    "Maximum Capacity": {
        "psnr": 0.14,
        "ber": 0.16,
        "correlation": 0.15,
        "runtime": 0.10,
        "capacity": 0.45,
    },
    "Fast Embedding": {
        "psnr": 0.20,
        "ber": 0.15,
        "correlation": 0.18,
        "runtime": 0.37,
        "capacity": 0.10,
    },
}


# ============================================================
# STRATEGY LIBRARY
# ============================================================

STRATEGY_LIBRARY: List[WatermarkStrategy] = [

    WatermarkStrategy(
        method="DCT",
        display_name="Discrete Cosine Transform",
        description=(
            "DCT embeds watermark information into frequency coefficients, "
            "usually in mid-frequency bands to balance robustness and invisibility."
        ),
        advantages=[
            "Fast embedding and extraction",
            "Relevant for JPEG-like compression",
            "Simple and computationally efficient",
            "Suitable for large-scale batch processing",
        ],
        disadvantages=[
            "Less stable against geometric attacks",
            "May lose robustness under strong rotation or cropping",
            "Performance depends strongly on coefficient selection",
        ],
        alpha_range=(5, 30),
        alpha_step=5,
        supported_watermark_sizes=[
            (32, 32),
            (64, 64),
        ],
        typical_runtime="Fast",
        computational_complexity="Low",
        imperceptibility_level="Medium-High",
        robustness_level="Medium",
        capacity_level="High",
        performance_prior=PerformancePrior(
            expected_psnr=44.5,
            expected_ber=0.060,
            expected_correlation=0.930,
            runtime_score=0.25,
            capacity_score=0.85,
        ),
        suitability=StrategySuitability(
            preferred_entropy=["Medium", "High"],
            preferred_texture=["Medium", "High"],
            preferred_edge_density=["Medium", "High"],
            preferred_complexity=["Medium", "High"],
            suitable_host_profiles=[
                "Balanced",
                "Highly Textured / Natural",
                "High-frequency Rich",
            ],
            suitable_objectives=[
                "Maximum Capacity",
                "Fast Embedding",
                "Balanced",
            ],
        ),
        explanation_templates=[
            "DCT is recommended when fast embedding is important.",
            "Mid-frequency embedding is suitable for images with moderate or high texture.",
            "DCT is appropriate when capacity and runtime are prioritized.",
        ],
    ),

    WatermarkStrategy(
        method="DWT",
        display_name="Discrete Wavelet Transform",
        description=(
            "DWT decomposes the image into multi-resolution sub-bands and embeds "
            "watermark data into selected frequency components."
        ),
        advantages=[
            "Good multi-resolution representation",
            "Often provides strong imperceptibility",
            "Suitable for smooth and moderately textured images",
            "Good balance between image quality and robustness",
        ],
        disadvantages=[
            "Sub-band choice affects extraction stability",
            "May be weaker under severe compression if not tuned",
            "Extraction may be sensitive to resizing or geometric changes",
        ],
        alpha_range=(5, 25),
        alpha_step=5,
        supported_watermark_sizes=[
            (32, 32),
            (64, 64),
        ],
        typical_runtime="Fast",
        computational_complexity="Low-Medium",
        imperceptibility_level="High",
        robustness_level="Medium",
        capacity_level="Medium",
        performance_prior=PerformancePrior(
            expected_psnr=46.0,
            expected_ber=0.045,
            expected_correlation=0.950,
            runtime_score=0.35,
            capacity_score=0.75,
        ),
        suitability=StrategySuitability(
            preferred_entropy=["Low", "Medium"],
            preferred_texture=["Low", "Medium"],
            preferred_edge_density=["Low", "Medium"],
            preferred_complexity=["Low", "Medium"],
            suitable_host_profiles=[
                "Smooth / Medical-like",
                "Balanced",
                "Low Contrast / Bright",
            ],
            suitable_objectives=[
                "Maximum Image Quality",
                "Balanced",
                "Fast Embedding",
            ],
        ),
        explanation_templates=[
            "DWT is suitable for preserving visual quality in smooth images.",
            "Wavelet decomposition helps balance imperceptibility and robustness.",
            "DWT is recommended when the objective emphasizes image quality.",
        ],
    ),
    WatermarkStrategy(
        method="DCT-SVD",
        display_name="Hybrid DCT-SVD",
        description=(
            "Hybrid watermarking using Discrete Cosine Transform and Singular "
            "Value Decomposition for improved imperceptibility and robustness."
        ),
        advantages=[
            "Very high visual quality",
            "Good robustness against common attacks",
            "Stable singular values",
            "Suitable for research applications",
        ],
        disadvantages=[
            "Higher computational cost",
            "Longer embedding time",
            "More complex implementation",
        ],
        alpha_range=(5, 40),
        alpha_step=5,
        supported_watermark_sizes=[
            (32, 32),
            (64, 64),
        ],
        typical_runtime="Medium",
        computational_complexity="Medium",
        imperceptibility_level="Very High",
        robustness_level="High",
        capacity_level="Medium",
        performance_prior=PerformancePrior(
            expected_psnr=48.0,
            expected_ber=0.025,
            expected_correlation=0.980,
            runtime_score=0.55,
            capacity_score=0.68,
        ),
        suitability=StrategySuitability(
            preferred_entropy=["Medium", "High"],
            preferred_texture=["Medium", "High"],
            preferred_edge_density=["Medium", "High"],
            preferred_complexity=["Medium", "High"],
            suitable_host_profiles=[
                "Balanced",
                "Highly Textured / Natural",
                "High-frequency Rich",
            ],
            suitable_objectives=[
                "Balanced",
                "Maximum Image Quality",
                "Maximum Robustness",
            ],
        ),
        explanation_templates=[
            "DCT-SVD provides strong robustness while maintaining excellent image quality.",
            "Suitable when both PSNR and robustness are important.",
            "Recommended for balanced optimization.",
        ],
    ),

    WatermarkStrategy(
        method="DWT-SVD",
        display_name="Hybrid DWT-SVD",
        description=(
            "Hybrid watermarking using Wavelet Transform and Singular Value "
            "Decomposition."
        ),
        advantages=[
            "Excellent robustness",
            "High imperceptibility",
            "Very stable extraction",
            "Well suited for medical and scientific images",
        ],
        disadvantages=[
            "Slower than DCT",
            "Higher computational complexity",
        ],
        alpha_range=(5, 40),
        alpha_step=5,
        supported_watermark_sizes=[
            (32, 32),
            (64, 64),
        ],
        typical_runtime="Medium",
        computational_complexity="High",
        imperceptibility_level="Very High",
        robustness_level="Very High",
        capacity_level="Medium",
        performance_prior=PerformancePrior(
            expected_psnr=49.0,
            expected_ber=0.015,
            expected_correlation=0.988,
            runtime_score=0.60,
            capacity_score=0.62,
        ),
        suitability=StrategySuitability(
            preferred_entropy=["Low", "Medium"],
            preferred_texture=["Low", "Medium"],
            preferred_edge_density=["Low", "Medium"],
            preferred_complexity=["Low", "Medium"],
            suitable_host_profiles=[
                "Smooth / Medical-like",
                "Balanced",
                "Low Contrast / Bright",
            ],
            suitable_objectives=[
                "Maximum Robustness",
                "Maximum Image Quality",
                "Balanced",
            ],
        ),
        explanation_templates=[
            "DWT-SVD performs exceptionally well for smooth images.",
            "Recommended when robustness is the primary objective.",
            "Hybrid transform improves resistance against common attacks.",
        ],
    ),

    WatermarkStrategy(
        method="BLOCK-SVD",
        display_name="Block-based SVD",
        description=(
            "Local block-based singular value decomposition watermarking."
        ),
        advantages=[
            "Good local robustness",
            "Flexible embedding",
            "Suitable for adaptive watermarking",
        ],
        disadvantages=[
            "Runtime increases with image size",
            "Parameter tuning is important",
        ],
        alpha_range=(5, 35),
        alpha_step=5,
        supported_watermark_sizes=[
            (32, 32),
            (64, 64),
        ],
        typical_runtime="Medium",
        computational_complexity="Medium",
        imperceptibility_level="High",
        robustness_level="High",
        capacity_level="Medium",
        performance_prior=PerformancePrior(
            expected_psnr=47.0,
            expected_ber=0.020,
            expected_correlation=0.985,
            runtime_score=0.58,
            capacity_score=0.66,
        ),
        suitability=StrategySuitability(
            preferred_entropy=["Medium", "High"],
            preferred_texture=["Medium", "High"],
            preferred_edge_density=["Medium", "High"],
            preferred_complexity=["Medium", "High"],
            suitable_host_profiles=[
                "Balanced",
                "Highly Textured / Natural",
            ],
            suitable_objectives=[
                "Balanced",
                "Maximum Robustness",
            ],
        ),
        explanation_templates=[
            "Block-wise embedding provides strong local robustness.",
            "Recommended for images with rich local structures.",
            "Suitable for adaptive watermark embedding.",
        ],
    ),
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_strategy(method: str) -> Optional[WatermarkStrategy]:
    """
    Return a strategy object by method name.
    """
    for strategy in STRATEGY_LIBRARY:
        if strategy.method.upper() == method.upper():
            return strategy
    return None


def list_methods() -> List[str]:
    """
    Return supported method names.
    """
    return [s.method for s in STRATEGY_LIBRARY]


def list_display_names() -> List[str]:
    """
    Return display names.
    """
    return [s.display_name for s in STRATEGY_LIBRARY]


def strategies_for_objective(objective: str) -> List[WatermarkStrategy]:
    """
    Return strategies suitable for the selected objective.
    """
    return [
        s for s in STRATEGY_LIBRARY
        if objective in s.suitability.suitable_objectives
    ]


def strategy_to_dataframe():
    """
    Convert the strategy library to a pandas DataFrame.
    """
    import pandas as pd

    rows = []

    for s in STRATEGY_LIBRARY:

        rows.append({
            "Method": s.method,
            "Display Name": s.display_name,
            "Runtime": s.typical_runtime,
            "Complexity": s.computational_complexity,
            "Imperceptibility": s.imperceptibility_level,
            "Robustness": s.robustness_level,
            "Capacity": s.capacity_level,
            "Alpha Range": f"{s.alpha_range[0]}-{s.alpha_range[1]}",
            "Recommended Sizes": ", ".join(
                [f"{w}×{h}" for w, h in s.supported_watermark_sizes]
            ),
            "Typical PSNR": s.performance_prior.expected_psnr,
            "Typical BER": s.performance_prior.expected_ber,
            "Typical Correlation": s.performance_prior.expected_correlation,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    print("=" * 60)
    print("AWSRE Strategy Library")
    print("=" * 60)

    print("\nSupported methods:")

    for strategy in STRATEGY_LIBRARY:
        print(f"- {strategy.method}")

    print("\nTotal strategies:", len(STRATEGY_LIBRARY))
