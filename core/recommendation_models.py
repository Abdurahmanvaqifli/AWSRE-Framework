"""
AWSRE Recommendation and Experiment Data Models

This module defines the shared data structures used by:

- AWSRE-Bench
- Recommendation Engine
- Performance Predictor
- Watermark Embedding
- Attack Simulation
- Verification
- Streamlit UI
- FastAPI backend
- Desktop application

The module contains no Streamlit code and no watermarking algorithms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math
import uuid


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_METHODS: Tuple[str, ...] = (
    "DCT",
    "DWT",
    "DCT-SVD",
    "DWT-SVD",
    "BLOCK-SVD",
)

SUPPORTED_OBJECTIVES: Tuple[str, ...] = (
    "Maximum Image Quality",
    "Maximum Robustness",
    "Balanced",
    "Maximum Capacity",
    "Fast Embedding",
)

SUPPORTED_PREDICTION_SOURCES: Tuple[str, ...] = (
    "Strategy Prior",
    "Historical KNN",
    "Interpolation",
    "Random Forest",
    "XGBoost",
    "Neural Network",
)

SUPPORTED_EXPERIMENT_STATUSES: Tuple[str, ...] = (
    "PENDING",
    "RUNNING",
    "SUCCESS",
    "FAILED",
    "SKIPPED",
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now_iso() -> str:
    """
    Return the current UTC time in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def generate_identifier(prefix: str) -> str:
    """
    Generate a readable unique identifier.

    Example
    -------
    EXP-20260712-184533-A12BC34D
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8].upper()

    return f"{prefix}-{timestamp}-{short_uuid}"


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.
    """
    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Convert a value to integer.
    """
    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Clamp a numeric value to a fixed interval.
    """
    return max(minimum, min(maximum, value))


def calculate_file_sha256(file_path: str | Path) -> str:
    """
    Calculate SHA-256 hash of a file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    sha256 = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# SERIALIZABLE BASE MODEL
# ============================================================

@dataclass
class SerializableModel:
    """
    Base class for AWSRE data objects.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dataclass into a dictionary.
        """
        return asdict(self)

    def to_json(
        self,
        indent: int = 4,
    ) -> str:
        """
        Serialize the model as JSON text.
        """
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            default=str,
        )

    def save_json(
        self,
        output_path: str | Path,
        indent: int = 4,
    ) -> Path:
        """
        Save the model as a JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            self.to_json(indent=indent),
            encoding="utf-8",
        )

        return path


# ============================================================
# HOST IMAGE MODEL
# ============================================================

@dataclass
class HostImageRecord(SerializableModel):
    """
    Metadata and extracted features of a host image.
    """

    image_id: str = field(
        default_factory=lambda: generate_identifier("HOST")
    )

    file_name: str = ""
    file_path: Optional[str] = None
    file_hash: str = ""

    width: int = 0
    height: int = 0
    resolution: int = 0
    aspect_ratio: float = 1.0

    brightness: float = 0.0
    contrast: float = 0.0
    entropy: float = 0.0
    edge_density: float = 0.0
    texture_variance: float = 0.0
    frequency_energy: float = 0.0
    high_frequency_ratio: float = 0.0
    dynamic_range: float = 0.0
    noise_level: float = 0.0
    sharpness: float = 0.0
    gradient_complexity: float = 0.0
    histogram_uniformity: float = 0.0
    smooth_area_ratio: float = 0.0
    image_complexity_score: float = 0.0

    host_profile: str = "Unknown"
    benchmark_tag: str = "General"

    created_at_utc: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        self.width = max(0, safe_int(self.width))
        self.height = max(0, safe_int(self.height))

        if self.resolution <= 0 and self.width > 0 and self.height > 0:
            self.resolution = self.width * self.height
        else:
            self.resolution = max(0, safe_int(self.resolution))

        if self.height > 0:
            self.aspect_ratio = safe_float(
                self.width / self.height,
                default=1.0,
            )
        else:
            self.aspect_ratio = safe_float(
                self.aspect_ratio,
                default=1.0,
            )

        self.brightness = clamp(
            safe_float(self.brightness),
            0.0,
            255.0,
        )

        self.contrast = max(0.0, safe_float(self.contrast))
        self.entropy = max(0.0, safe_float(self.entropy))

        self.edge_density = clamp(
            safe_float(self.edge_density),
            0.0,
            1.0,
        )

        self.texture_variance = max(
            0.0,
            safe_float(self.texture_variance),
        )

        self.frequency_energy = max(
            0.0,
            safe_float(self.frequency_energy),
        )

        self.high_frequency_ratio = clamp(
            safe_float(self.high_frequency_ratio),
            0.0,
            1.0,
        )

        self.dynamic_range = clamp(
            safe_float(self.dynamic_range),
            0.0,
            255.0,
        )

        self.noise_level = max(
            0.0,
            safe_float(self.noise_level),
        )

        self.sharpness = max(
            0.0,
            safe_float(self.sharpness),
        )

        self.gradient_complexity = max(
            0.0,
            safe_float(self.gradient_complexity),
        )

        self.histogram_uniformity = clamp(
            safe_float(self.histogram_uniformity),
            0.0,
            1.0,
        )

        self.smooth_area_ratio = clamp(
            safe_float(self.smooth_area_ratio),
            0.0,
            1.0,
        )

        self.image_complexity_score = clamp(
            safe_float(self.image_complexity_score),
            0.0,
            100.0,
        )

    @classmethod
    def from_feature_analysis(
        cls,
        file_name: str,
        analysis: Dict[str, Any],
        file_path: Optional[str] = None,
        file_hash: str = "",
        benchmark_tag: str = "General",
    ) -> "HostImageRecord":
        """
        Construct the record from the result returned by
        feature_extraction.analyze_host_image().
        """
        features = analysis.get("features", {})
        interpretation = analysis.get("interpretation", {})

        return cls(
            file_name=file_name,
            file_path=file_path,
            file_hash=file_hash,
            width=features.get("width", 0),
            height=features.get("height", 0),
            resolution=features.get("resolution", 0),
            aspect_ratio=features.get("aspect_ratio", 1.0),
            brightness=features.get("brightness", 0.0),
            contrast=features.get("contrast", 0.0),
            entropy=features.get("entropy", 0.0),
            edge_density=features.get("edge_density", 0.0),
            texture_variance=features.get(
                "texture_variance",
                0.0,
            ),
            frequency_energy=features.get(
                "frequency_energy",
                0.0,
            ),
            high_frequency_ratio=features.get(
                "high_frequency_ratio",
                0.0,
            ),
            dynamic_range=features.get(
                "dynamic_range",
                0.0,
            ),
            noise_level=features.get(
                "noise_level",
                0.0,
            ),
            sharpness=features.get(
                "sharpness",
                0.0,
            ),
            gradient_complexity=features.get(
                "gradient_complexity",
                0.0,
            ),
            histogram_uniformity=features.get(
                "histogram_uniformity",
                0.0,
            ),
            smooth_area_ratio=features.get(
                "smooth_area_ratio",
                0.0,
            ),
            image_complexity_score=features.get(
                "image_complexity_score",
                0.0,
            ),
            host_profile=interpretation.get(
                "host_profile",
                "Unknown",
            ),
            benchmark_tag=benchmark_tag,
        )

    def feature_vector(self) -> List[float]:
        """
        Return the numeric feature vector used by predictors.
        """
        return [
            self.brightness,
            self.contrast,
            self.entropy,
            self.edge_density,
            self.texture_variance,
            self.frequency_energy,
            self.high_frequency_ratio,
            self.dynamic_range,
            self.noise_level,
            self.sharpness,
            self.gradient_complexity,
            self.histogram_uniformity,
            self.smooth_area_ratio,
            self.image_complexity_score,
            float(self.resolution),
            self.aspect_ratio,
        ]


# ============================================================
# WATERMARK MODEL
# ============================================================

@dataclass
class WatermarkRecord(SerializableModel):
    """
    Metadata and extracted features of a watermark.
    """

    watermark_id: str = field(
        default_factory=lambda: generate_identifier("WM")
    )

    file_name: str = ""
    file_path: Optional[str] = None
    file_hash: str = ""

    watermark_type: str = "Binary Pattern"
    owner_label: Optional[str] = None

    width: int = 0
    height: int = 0
    payload_bits: int = 0

    density: float = 0.0
    foreground_ratio: float = 0.0
    entropy: float = 0.0
    edge_complexity: float = 0.0
    connected_components: int = 0
    compactness: float = 0.0
    fill_ratio: float = 0.0
    symmetry_score: float = 0.0
    stroke_complexity: float = 0.0
    structural_complexity: float = 0.0
    watermark_complexity_score: float = 0.0

    density_level: str = "Unknown"
    complexity_level: str = "Unknown"

    created_at_utc: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        self.width = max(0, safe_int(self.width))
        self.height = max(0, safe_int(self.height))

        if self.payload_bits <= 0 and self.width > 0 and self.height > 0:
            self.payload_bits = self.width * self.height
        else:
            self.payload_bits = max(
                0,
                safe_int(self.payload_bits),
            )

        self.density = clamp(
            safe_float(self.density),
            0.0,
            1.0,
        )

        self.foreground_ratio = clamp(
            safe_float(self.foreground_ratio),
            0.0,
            1.0,
        )

        self.entropy = max(
            0.0,
            safe_float(self.entropy),
        )

        self.edge_complexity = clamp(
            safe_float(self.edge_complexity),
            0.0,
            1.0,
        )

        self.connected_components = max(
            0,
            safe_int(self.connected_components),
        )

        self.compactness = clamp(
            safe_float(self.compactness),
            0.0,
            1.0,
        )

        self.fill_ratio = clamp(
            safe_float(self.fill_ratio),
            0.0,
            1.0,
        )

        self.symmetry_score = clamp(
            safe_float(self.symmetry_score),
            0.0,
            1.0,
        )

        self.stroke_complexity = max(
            0.0,
            safe_float(self.stroke_complexity),
        )

        self.structural_complexity = max(
            0.0,
            safe_float(self.structural_complexity),
        )

        self.watermark_complexity_score = clamp(
            safe_float(self.watermark_complexity_score),
            0.0,
            100.0,
        )

    @classmethod
    def from_feature_analysis(
        cls,
        file_name: str,
        analysis: Dict[str, Any],
        watermark_type: str,
        file_path: Optional[str] = None,
        file_hash: str = "",
        owner_label: Optional[str] = None,
    ) -> "WatermarkRecord":
        """
        Construct a watermark record from
        feature_extraction.analyze_watermark().
        """
        features = analysis.get("features", {})
        interpretation = analysis.get("interpretation", {})

        return cls(
            file_name=file_name,
            file_path=file_path,
            file_hash=file_hash,
            watermark_type=watermark_type,
            owner_label=owner_label,
            width=features.get("width", 0),
            height=features.get("height", 0),
            payload_bits=features.get("payload_bits", 0),
            density=features.get("density", 0.0),
            foreground_ratio=features.get(
                "foreground_ratio",
                0.0,
            ),
            entropy=features.get("entropy", 0.0),
            edge_complexity=features.get(
                "edge_complexity",
                0.0,
            ),
            connected_components=features.get(
                "connected_components",
                0,
            ),
            compactness=features.get("compactness", 0.0),
            fill_ratio=features.get("fill_ratio", 0.0),
            symmetry_score=features.get(
                "symmetry_score",
                0.0,
            ),
            stroke_complexity=features.get(
                "stroke_complexity",
                0.0,
            ),
            structural_complexity=features.get(
                "structural_complexity",
                0.0,
            ),
            watermark_complexity_score=features.get(
                "watermark_complexity_score",
                0.0,
            ),
            density_level=interpretation.get(
                "density_level",
                "Unknown",
            ),
            complexity_level=interpretation.get(
                "complexity_level",
                "Unknown",
            ),
        )

    def size_tuple(self) -> Tuple[int, int]:
        """
        Return watermark size as width and height.
        """
        return self.width, self.height

    def feature_vector(self) -> List[float]:
        """
        Return the watermark numeric feature vector.
        """
        return [
            float(self.width),
            float(self.height),
            float(self.payload_bits),
            self.density,
            self.foreground_ratio,
            self.entropy,
            self.edge_complexity,
            float(self.connected_components),
            self.compactness,
            self.fill_ratio,
            self.symmetry_score,
            self.stroke_complexity,
            self.structural_complexity,
            self.watermark_complexity_score,
        ]


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

@dataclass
class EmbeddingConfiguration(SerializableModel):
    """
    Configuration of one candidate embedding strategy.
    """

    method: str
    alpha: float
    watermark_width: int
    watermark_height: int

    channel: str = "Grayscale"
    block_size: int = 8
    decomposition_level: int = 1
    subband: Optional[str] = None

    configuration_id: str = field(
        default_factory=lambda: generate_identifier("CFG")
    )

    def __post_init__(self) -> None:
        normalized_method = self.method.upper()

        if normalized_method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported embedding method: {self.method}"
            )

        self.method = normalized_method
        self.alpha = max(0.0, safe_float(self.alpha))

        self.watermark_width = max(
            1,
            safe_int(self.watermark_width, default=32),
        )

        self.watermark_height = max(
            1,
            safe_int(self.watermark_height, default=32),
        )

        self.block_size = max(
            1,
            safe_int(self.block_size, default=8),
        )

        self.decomposition_level = max(
            1,
            safe_int(self.decomposition_level, default=1),
        )

    @property
    def watermark_size(self) -> Tuple[int, int]:
        return self.watermark_width, self.watermark_height

    @property
    def watermark_size_label(self) -> str:
        return (
            f"{self.watermark_width}"
            f"x"
            f"{self.watermark_height}"
        )

    def candidate_key(self) -> str:
        """
        Stable key for duplicate and resume checks.
        """
        return (
            f"{self.method}|"
            f"{self.alpha}|"
            f"{self.watermark_size_label}|"
            f"{self.channel}|"
            f"{self.block_size}|"
            f"{self.decomposition_level}|"
            f"{self.subband or 'NONE'}"
        )


# ============================================================
# ATTACK CONFIGURATION
# ============================================================

@dataclass
class AttackConfiguration(SerializableModel):
    """
    Configuration of an image attack.
    """

    attack_type: str = "none"
    parameter: Optional[float | int | str] = None
    sequence_order: int = 0

    attack_id: str = field(
        default_factory=lambda: generate_identifier("ATK")
    )

    def __post_init__(self) -> None:
        if not self.attack_type:
            self.attack_type = "none"

        self.attack_type = self.attack_type.strip().lower()
        self.sequence_order = max(
            0,
            safe_int(self.sequence_order),
        )

    def attack_key(self) -> str:
        """
        Stable representation used by benchmark resume logic.
        """
        return (
            f"{self.sequence_order}|"
            f"{self.attack_type}|"
            f"{self.parameter}"
        )

    @property
    def is_no_attack(self) -> bool:
        return self.attack_type in {
            "none",
            "no_attack",
            "no attack",
        }
    # ============================================================
# PERFORMANCE METRICS
# ============================================================

@dataclass
class ImperceptibilityMetrics(SerializableModel):
    """
    Visual quality metrics.
    """

    mse: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0

    def __post_init__(self):

        self.mse = max(0.0, safe_float(self.mse))
        self.psnr = max(0.0, safe_float(self.psnr))

        self.ssim = clamp(
            safe_float(self.ssim),
            0.0,
            1.0,
        )


@dataclass
class RobustnessMetrics(SerializableModel):
    """
    Extraction quality metrics.
    """

    ber: float = 0.0
    correlation: float = 1.0

    extracted_successfully: bool = True

    def __post_init__(self):

        self.ber = clamp(
            safe_float(self.ber),
            0.0,
            1.0,
        )

        self.correlation = clamp(
            safe_float(self.correlation),
            -1.0,
            1.0,
        )


@dataclass
class RuntimeMetrics(SerializableModel):
    """
    Runtime measurements.
    """

    embedding_time: float = 0.0
    attack_time: float = 0.0
    extraction_time: float = 0.0

    @property
    def total_runtime(self):

        return (
            self.embedding_time
            + self.attack_time
            + self.extraction_time
        )

    def __post_init__(self):

        self.embedding_time = max(
            0.0,
            safe_float(self.embedding_time)
        )

        self.attack_time = max(
            0.0,
            safe_float(self.attack_time)
        )

        self.extraction_time = max(
            0.0,
            safe_float(self.extraction_time)
        )


# ============================================================
# PREDICTION
# ============================================================

@dataclass
class PredictionResult(SerializableModel):
    """
    Recommendation prediction.
    """

    expected_psnr: float = 0.0
    expected_ssim: float = 0.0
    expected_ber: float = 0.0
    expected_correlation: float = 0.0

    expected_runtime: float = 0.0

    expected_capacity: float = 0.0

    confidence: float = 0.0

    prediction_source: str = "Strategy Prior"

    def __post_init__(self):

        self.expected_psnr = max(
            0.0,
            safe_float(self.expected_psnr)
        )

        self.expected_ssim = clamp(
            safe_float(self.expected_ssim),
            0.0,
            1.0,
        )

        self.expected_ber = clamp(
            safe_float(self.expected_ber),
            0.0,
            1.0,
        )

        self.expected_correlation = clamp(
            safe_float(self.expected_correlation),
            -1.0,
            1.0,
        )

        self.expected_runtime = max(
            0.0,
            safe_float(self.expected_runtime)
        )

        self.expected_capacity = max(
            0.0,
            safe_float(self.expected_capacity)
        )

        self.confidence = clamp(
            safe_float(self.confidence),
            0.0,
            100.0,
        )


# ============================================================
# RECOMMENDATION CANDIDATE
# ============================================================

@dataclass
class RecommendationCandidate(SerializableModel):
    """
    One candidate evaluated by AWSRE.
    """

    configuration: EmbeddingConfiguration

    prediction: PredictionResult

    recommendation_score: float = 0.0

    pareto_optimal: bool = False

    explanation: List[str] = field(
        default_factory=list
    )

    def __post_init__(self):

        self.recommendation_score = clamp(
            safe_float(self.recommendation_score),
            0.0,
            1.0,
        )


# ============================================================
# EXPERIMENT RECORD
# ============================================================

@dataclass
class ExperimentRecord(SerializableModel):
    """
    One complete benchmark experiment.
    """

    experiment_id: str = field(
        default_factory=lambda: generate_identifier("EXP")
    )

    benchmark_version: str = "1.0"

    benchmark_tag: str = "General"

    timestamp_utc: str = field(
        default_factory=utc_now_iso
    )

    host: HostImageRecord = field(
        default_factory=HostImageRecord
    )

    watermark: WatermarkRecord = field(
        default_factory=WatermarkRecord
    )

    embedding: EmbeddingConfiguration = field(
        default_factory=lambda:
        EmbeddingConfiguration(
            method="DCT",
            alpha=10,
            watermark_width=32,
            watermark_height=32,
        )
    )

    attack: AttackConfiguration = field(
        default_factory=AttackConfiguration
    )

    imperceptibility: ImperceptibilityMetrics = field(
        default_factory=ImperceptibilityMetrics
    )

    robustness: RobustnessMetrics = field(
        default_factory=RobustnessMetrics
    )

    runtime: RuntimeMetrics = field(
        default_factory=RuntimeMetrics
    )

    prediction: Optional[
        PredictionResult
    ] = None

    status: str = "SUCCESS"

    error_message: str = ""

    notes: str = ""

    @property
    def experiment_key(self):
        """
        Unique benchmark key.
        """

        return (
            self.host.image_id
            + "_"
            + self.watermark.watermark_id
            + "_"
            + self.embedding.candidate_key()
            + "_"
            + self.attack.attack_key()
        )

    @property
    def runtime_seconds(self):

        return self.runtime.total_runtime

    @property
    def psnr(self):

        return self.imperceptibility.psnr

    @property
    def ssim(self):

        return self.imperceptibility.ssim

    @property
    def mse(self):

        return self.imperceptibility.mse

    @property
    def ber(self):

        return self.robustness.ber

    @property
    def correlation(self):

        return self.robustness.correlation
    # ============================================================
# RECOMMENDATION RESULT
# ============================================================

@dataclass
class RecommendationResult(SerializableModel):
    """
    Final AWSRE recommendation returned to the UI.
    """

    host: HostImageRecord
    watermark: WatermarkRecord
    objective: str

    candidates: List[RecommendationCandidate] = field(default_factory=list)

    created_at: str = field(default_factory=utc_now_iso)

    def top1(self):

        if len(self.candidates) == 0:
            return None

        return self.candidates[0]

    def top3(self):

        return self.candidates[:3]

    def sort(self):

        self.candidates.sort(
            key=lambda x: x.recommendation_score,
            reverse=True
        )

    @property
    def recommendation_count(self):

        return len(self.candidates)


# ============================================================
# BENCHMARK SUMMARY
# ============================================================

@dataclass
class BenchmarkSummary(SerializableModel):

    total_experiments: int = 0

    successful_experiments: int = 0

    failed_experiments: int = 0

    average_psnr: float = 0.0

    average_ssim: float = 0.0

    average_ber: float = 0.0

    average_correlation: float = 0.0

    average_runtime: float = 0.0

    created_at: str = field(default_factory=utc_now_iso)


# ============================================================
# PREDICTION ERROR
# ============================================================

@dataclass
class PredictionError(SerializableModel):

    psnr_error: float = 0.0

    ssim_error: float = 0.0

    ber_error: float = 0.0

    correlation_error: float = 0.0

    runtime_error: float = 0.0

    def mae(self):

        return np.mean([
            abs(self.psnr_error),
            abs(self.ssim_error),
            abs(self.ber_error),
            abs(self.correlation_error),
            abs(self.runtime_error),
        ])


# ============================================================
# EXPERIMENT DATABASE
# ============================================================

class ExperimentDatabase:

    """
    In-memory experiment database.

    Later this can easily become SQLite
    without changing the rest of AWSRE.
    """

    def __init__(self):

        self.records: List[ExperimentRecord] = []

    def add(
        self,
        record: ExperimentRecord,
    ):

        self.records.append(record)

    def __len__(self):

        return len(self.records)

    def clear(self):

        self.records.clear()

    def experiment_ids(self):

        return [
            x.experiment_id
            for x in self.records
        ]

    def experiment_keys(self):

        return [
            x.experiment_key
            for x in self.records
        ]

    def already_exists(
        self,
        experiment_key,
    ):

        return experiment_key in self.experiment_keys()

    def successful(self):

        return [
            x
            for x in self.records
            if x.status == "SUCCESS"
        ]

    def failed(self):

        return [
            x
            for x in self.records
            if x.status == "FAILED"
        ]

    def summary(self):

        success = self.successful()

        if len(success) == 0:

            return BenchmarkSummary()

        return BenchmarkSummary(

            total_experiments=len(self.records),

            successful_experiments=len(success),

            failed_experiments=len(
                self.failed()
            ),

            average_psnr=np.mean(
                [x.psnr for x in success]
            ),

            average_ssim=np.mean(
                [x.ssim for x in success]
            ),

            average_ber=np.mean(
                [x.ber for x in success]
            ),

            average_correlation=np.mean(
                [x.correlation for x in success]
            ),

            average_runtime=np.mean(
                [x.runtime_seconds for x in success]
            ),

        )

    def to_dataframe(self):

        rows = []

        for exp in self.records:

            rows.append({

                "experiment_id": exp.experiment_id,

                "timestamp": exp.timestamp_utc,

                "host": exp.host.file_name,

                "watermark": exp.watermark.file_name,

                "method": exp.embedding.method,

                "alpha": exp.embedding.alpha,

                "watermark_size":
                    exp.embedding.watermark_size_label,

                "attack":
                    exp.attack.attack_type,

                "attack_parameter":
                    exp.attack.parameter,

                "psnr":
                    exp.psnr,

                "ssim":
                    exp.ssim,

                "mse":
                    exp.mse,

                "ber":
                    exp.ber,

                "correlation":
                    exp.correlation,

                "runtime":
                    exp.runtime_seconds,

                "status":
                    exp.status,

            })

        return pd.DataFrame(rows)

    def save_csv(
        self,
        csv_path,
    ):

        df = self.to_dataframe()

        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )

    def load_csv(
        self,
        csv_path,
    ):

        df = pd.read_csv(csv_path)

        return df


# ============================================================
# FACTORY
# ============================================================

def create_empty_database():

    return ExperimentDatabase()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    db = ExperimentDatabase()

    print("=" * 60)

    print("AWSRE Recommendation Models")

    print("=" * 60)

    print()

    print("Supported methods:")

    for method in SUPPORTED_METHODS:

        print("-", method)

    print()

    print("Supported objectives:")

    for obj in SUPPORTED_OBJECTIVES:

        print("-", obj)

    print()

    print("Database created successfully.")

    print()

    print("Records:", len(db))
