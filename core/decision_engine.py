"""
AWSRE Decision Engine

This module ranks watermarking algorithms according to:

- Imperceptibility
- Robustness
- Runtime efficiency
- Capacity

The engine is independent from Streamlit and can later be reused
by FastAPI, desktop software, experiments and automated benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class PriorityWeights:
    """
    User-defined recommendation priorities.

    All values may initially be supplied on any positive scale,
    such as 0-100 or 1-5. They are normalized automatically.
    """

    imperceptibility: float
    robustness: float
    speed: float
    capacity: float

    def normalized(self) -> Dict[str, float]:
        """
        Normalize priority values so their sum becomes 1.
        """
        raw_weights = {
            "imperceptibility": max(
                float(self.imperceptibility),
                0.0,
            ),
            "robustness": max(
                float(self.robustness),
                0.0,
            ),
            "speed": max(
                float(self.speed),
                0.0,
            ),
            "capacity": max(
                float(self.capacity),
                0.0,
            ),
        }

        total = sum(raw_weights.values())

        if total <= 0:
            return {
                "imperceptibility": 0.25,
                "robustness": 0.25,
                "speed": 0.25,
                "capacity": 0.25,
            }

        return {
            name: value / total
            for name, value in raw_weights.items()
        }


@dataclass
class RecommendationResult:
    """
    Final output returned by the decision engine.
    """

    recommended_method: str
    confidence: float
    ranking: pd.DataFrame
    explanation: List[str]
    weights: Dict[str, float]


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _safe_numeric_series(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """
    Return a numeric pandas Series without NaN or infinite values.
    """
    if column not in dataframe.columns:
        return pd.Series(
            np.full(
                len(dataframe),
                default,
                dtype=np.float64,
            ),
            index=dataframe.index,
        )

    series = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).astype(np.float64)

    series = series.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if series.notna().any():
        replacement = float(
            series.dropna().median()
        )
    else:
        replacement = float(default)

    return series.fillna(
        replacement
    )


def normalize_higher_is_better(
    values: pd.Series,
) -> pd.Series:
    """
    Min-max normalization where larger values are better.

    Output range:
        0.0 to 1.0
    """
    numeric = values.astype(
        np.float64
    )

    minimum = float(
        numeric.min()
    )

    maximum = float(
        numeric.max()
    )

    difference = maximum - minimum

    if abs(difference) < 1e-12:
        return pd.Series(
            np.ones(
                len(numeric),
                dtype=np.float64,
            ),
            index=numeric.index,
        )

    return (
        numeric - minimum
    ) / difference


def normalize_lower_is_better(
    values: pd.Series,
) -> pd.Series:
    """
    Min-max normalization where smaller values are better.
    """
    higher_score = normalize_higher_is_better(
        values
    )

    return 1.0 - higher_score


# ============================================================
# METRIC GROUP SCORES
# ============================================================

def calculate_metric_scores(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw benchmark values into comparable 0-1 scores.

    Expected columns include:

    - Method
    - PSNR (dB)
    - SSIM
    - BER
    - Correlation
    - Total Time (s)
    - Capacity (bits)

    Missing columns are handled safely.
    """
    if results.empty:
        raise ValueError(
            "The benchmark result table is empty."
        )

    if "Method" not in results.columns:
        raise ValueError(
            "The result table must contain a 'Method' column."
        )

    scored = results.copy()

    psnr = _safe_numeric_series(
        scored,
        "PSNR (dB)",
    )

    ssim = _safe_numeric_series(
        scored,
        "SSIM",
    )

    ber = _safe_numeric_series(
        scored,
        "BER",
        default=1.0,
    )

    correlation = _safe_numeric_series(
        scored,
        "Correlation",
    )

    runtime = _safe_numeric_series(
        scored,
        "Total Time (s)",
    )

    capacity = _safe_numeric_series(
        scored,
        "Capacity (bits)",
        default=1.0,
    )

    scored["PSNR Score"] = normalize_higher_is_better(
        psnr
    )

    scored["SSIM Score"] = normalize_higher_is_better(
        ssim
    )

    scored["BER Score"] = normalize_lower_is_better(
        ber
    )

    scored["Correlation Score"] = (
        normalize_higher_is_better(
            correlation
        )
    )

    scored["Runtime Score"] = normalize_lower_is_better(
        runtime
    )

    scored["Capacity Score"] = (
        normalize_higher_is_better(
            capacity
        )
    )

    # Imperceptibility:
    # PSNR has 50% importance and SSIM has 50%.
    scored["Imperceptibility Score"] = (
        0.50 * scored["PSNR Score"]
        + 0.50 * scored["SSIM Score"]
    )

    # Robustness:
    # Lower BER and higher correlation are preferred.
    scored["Robustness Score"] = (
        0.50 * scored["BER Score"]
        + 0.50 * scored["Correlation Score"]
    )

    scored["Speed Score"] = scored[
        "Runtime Score"
    ]

    return scored


# ============================================================
# EXPLANATION ENGINE
# ============================================================

def build_explanation(
    ranking: pd.DataFrame,
) -> List[str]:
    """
    Produce human-readable reasons for the first-ranked method.
    """
    if ranking.empty:
        return [
            "No valid algorithm result was available."
        ]

    winner = ranking.iloc[0]

    reasons: List[str] = []

    imperceptibility = float(
        winner["Imperceptibility Score"]
    )

    robustness = float(
        winner["Robustness Score"]
    )

    speed = float(
        winner["Speed Score"]
    )

    capacity = float(
        winner["Capacity Score"]
    )

    if imperceptibility >= 0.80:
        reasons.append(
            "It achieved excellent visual imperceptibility."
        )
    elif imperceptibility >= 0.50:
        reasons.append(
            "It provided competitive visual quality."
        )

    if robustness >= 0.80:
        reasons.append(
            "It achieved strong watermark recovery quality."
        )
    elif robustness >= 0.50:
        reasons.append(
            "It provided a balanced watermark recovery result."
        )

    if speed >= 0.80:
        reasons.append(
            "It had one of the lowest processing times."
        )
    elif speed >= 0.50:
        reasons.append(
            "Its processing time was acceptable."
        )

    if capacity >= 0.80:
        reasons.append(
            "It supported high watermark capacity."
        )
    elif capacity >= 0.50:
        reasons.append(
            "It supported a practical watermark capacity."
        )

    if not reasons:
        reasons.append(
            "It obtained the highest weighted score under "
            "the selected priorities."
        )

    return reasons


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    ranking: pd.DataFrame,
) -> float:
    """
    Calculate recommendation confidence from the score margin.

    The confidence is not a statistical probability. It describes
    how clearly the first method outranked the remaining methods.
    """
    if ranking.empty:
        return 0.0

    best_score = float(
        ranking.iloc[0]["Final Score"]
    )

    if len(ranking) == 1:
        return round(
            min(
                100.0,
                max(
                    0.0,
                    best_score,
                ),
            ),
            2,
        )

    second_score = float(
        ranking.iloc[1]["Final Score"]
    )

    margin = max(
        best_score - second_score,
        0.0,
    )

    # Base confidence + score margin contribution.
    confidence = (
        55.0
        + 0.30 * best_score
        + 0.70 * margin
    )

    return round(
        min(
            99.0,
            max(
                0.0,
                confidence,
            ),
        ),
        2,
    )


# ============================================================
# MAIN RECOMMENDATION FUNCTION
# ============================================================

def recommend_strategy(
    benchmark_results: pd.DataFrame,
    priorities: PriorityWeights,
) -> RecommendationResult:
    """
    Rank algorithms and recommend the best watermarking strategy.
    """
    scored = calculate_metric_scores(
        benchmark_results
    )

    weights = priorities.normalized()

    scored["Final Score"] = 100.0 * (
        weights["imperceptibility"]
        * scored["Imperceptibility Score"]
        + weights["robustness"]
        * scored["Robustness Score"]
        + weights["speed"]
        * scored["Speed Score"]
        + weights["capacity"]
        * scored["Capacity Score"]
    )

    scored = scored.sort_values(
        by="Final Score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    scored.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(scored) + 1,
        ),
    )

    recommended_method = str(
        scored.iloc[0]["Method"]
    )

    confidence = calculate_confidence(
        scored
    )

    explanation = build_explanation(
        scored
    )

    return RecommendationResult(
        recommended_method=recommended_method,
        confidence=confidence,
        ranking=scored,
        explanation=explanation,
        weights=weights,
    )


# ============================================================
# OPTIONAL TEST
# ============================================================

if __name__ == "__main__":
    sample_results = pd.DataFrame({
        "Method": [
            "DCT",
            "DWT",
            "DCT-SVD",
        ],
        "PSNR (dB)": [
            42.10,
            47.50,
            51.20,
        ],
        "SSIM": [
            0.9700,
            0.9910,
            0.9980,
        ],
        "BER": [
            0.0200,
            0.0800,
            0.0100,
        ],
        "Correlation": [
            0.9500,
            0.8800,
            0.9900,
        ],
        "Total Time (s)": [
            0.12,
            0.20,
            0.45,
        ],
        "Capacity (bits)": [
            1024,
            1024,
            1024,
        ],
    })

    sample_priorities = PriorityWeights(
        imperceptibility=80,
        robustness=80,
        speed=30,
        capacity=20,
    )

    recommendation = recommend_strategy(
        sample_results,
        sample_priorities,
    )

    print(
        recommendation.ranking[
            [
                "Rank",
                "Method",
                "Final Score",
            ]
        ]
    )

    print(
        "\nRecommended:",
        recommendation.recommended_method,
    )

    print(
        "Confidence:",
        recommendation.confidence,
    )

    print(
        "Reasons:",
        recommendation.explanation,
    )
