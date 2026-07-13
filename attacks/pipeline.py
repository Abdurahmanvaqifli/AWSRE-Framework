"""
AWSRE Attack Pipeline

Central attack simulation module used by:

- AWSRE-Bench
- Verification module
- Streamlit platform
- Desktop application
- Research experiments

Supported attacks:
- none
- JPEG compression
- Gaussian noise
- salt-and-pepper noise
- Gaussian blur
- rotation
- cropping
- brightness adjustment
- contrast adjustment
- gamma correction
- sharpening
- combined attacks
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import time

import cv2
import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_ATTACKS = (
    "none",
    "jpeg",
    "gaussian_noise",
    "salt_pepper",
    "gaussian_blur",
    "rotation",
    "crop",
    "brightness",
    "contrast",
    "gamma",
    "sharpen",
    "combined",
)


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class AttackStep:
    """
    One attack inside an attack pipeline.
    """

    attack_type: str
    parameter: Any = None
    options: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttackResult:
    """
    Result returned by the attack engine.
    """

    image: np.ndarray
    attack_type: str
    parameter: Any
    runtime_seconds: float

    steps: List[Dict[str, Any]] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
        include_image: bool = False,
    ) -> Dict[str, Any]:
        data = {
            "attack_type": self.attack_type,
            "parameter": self.parameter,
            "runtime_seconds": self.runtime_seconds,
            "steps": self.steps,
            "metadata": self.metadata,
        }

        if include_image:
            data["image"] = self.image

        return data


# ============================================================
# VALIDATION
# ============================================================

def validate_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Validate and normalize an image to uint8.
    """
    if image is None:
        raise ValueError(
            "Attack input image cannot be None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "Attack input must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Attack input image cannot be empty."
        )

    if image.ndim not in (
        2,
        3,
    ):
        raise ValueError(
            "Attack input must be grayscale or color."
        )

    if image.ndim == 3 and image.shape[2] not in (
        3,
        4,
    ):
        raise ValueError(
            "Color image must contain 3 or 4 channels."
        )

    if not np.all(
        np.isfinite(image)
    ):
        raise ValueError(
            "Attack input contains NaN or infinity."
        )

    if image.dtype == np.uint8:
        return image.copy()

    return np.uint8(
        np.clip(
            np.rint(image),
            0,
            255,
        )
    )


def normalize_attack_name(
    attack_type: str,
) -> str:
    """
    Normalize an attack name.
    """
    if not isinstance(
        attack_type,
        str,
    ):
        raise TypeError(
            "attack_type must be a string."
        )

    normalized = (
        attack_type
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "no_attack": "none",
        "noattack": "none",
        "jpeg_compression": "jpeg",
        "jpg": "jpeg",
        "gaussian": "gaussian_noise",
        "noise": "gaussian_noise",
        "salt_and_pepper": "salt_pepper",
        "saltpepper": "salt_pepper",
        "blur": "gaussian_blur",
        "rotate": "rotation",
        "cropping": "crop",
        "brightness_adjustment": "brightness",
        "contrast_adjustment": "contrast",
        "gamma_correction": "gamma",
        "sharp": "sharpen",
    }

    normalized = aliases.get(
        normalized,
        normalized,
    )

    if normalized not in SUPPORTED_ATTACKS:
        raise ValueError(
            f"Unsupported attack: {normalized}. "
            f"Supported attacks: {', '.join(SUPPORTED_ATTACKS)}."
        )

    return normalized


# ============================================================
# INDIVIDUAL ATTACKS
# ============================================================

def apply_no_attack(
    image: np.ndarray,
) -> np.ndarray:
    """
    Return an unchanged image copy.
    """
    return validate_image(
        image
    )


def apply_jpeg_compression(
    image: np.ndarray,
    quality: int = 70,
) -> np.ndarray:
    """
    Simulate JPEG compression.

    quality must be between 1 and 100.
    """
    image_uint8 = validate_image(
        image
    )

    quality = int(
        quality
    )

    if not 1 <= quality <= 100:
        raise ValueError(
            "JPEG quality must be between 1 and 100."
        )

    success, encoded = cv2.imencode(
        ".jpg",
        image_uint8,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
        ],
    )

    if not success:
        raise RuntimeError(
            "JPEG encoding failed."
        )

    flag = (
        cv2.IMREAD_GRAYSCALE
        if image_uint8.ndim == 2
        else cv2.IMREAD_COLOR
    )

    decoded = cv2.imdecode(
        encoded,
        flag,
    )

    if decoded is None:
        raise RuntimeError(
            "JPEG decoding failed."
        )

    return decoded


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float = 0.01,
    *,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Add Gaussian noise.

    sigma <= 1 is interpreted relative to the 0–255 range.
    For example, 0.01 means standard deviation 2.55.
    """
    image_uint8 = validate_image(
        image
    )

    sigma = float(
        sigma
    )

    if sigma < 0:
        raise ValueError(
            "Gaussian-noise sigma cannot be negative."
        )

    sigma_pixels = (
        sigma * 255.0
        if sigma <= 1.0
        else sigma
    )

    rng = np.random.default_rng(
        seed
    )

    noise = rng.normal(
        loc=0.0,
        scale=sigma_pixels,
        size=image_uint8.shape,
    )

    attacked = (
        image_uint8.astype(np.float32)
        + noise.astype(np.float32)
    )

    return np.uint8(
        np.clip(
            np.rint(attacked),
            0,
            255,
        )
    )


def apply_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.05,
    *,
    salt_ratio: float = 0.5,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Apply salt-and-pepper noise.

    amount represents the fraction of affected pixels.
    """
    image_uint8 = validate_image(
        image
    )

    amount = float(
        amount
    )

    salt_ratio = float(
        salt_ratio
    )

    if not 0.0 <= amount <= 1.0:
        raise ValueError(
            "Salt-and-pepper amount must be between 0 and 1."
        )

    if not 0.0 <= salt_ratio <= 1.0:
        raise ValueError(
            "salt_ratio must be between 0 and 1."
        )

    attacked = image_uint8.copy()

    height, width = attacked.shape[:2]

    affected_pixels = int(
        round(
            height
            * width
            * amount
        )
    )

    if affected_pixels == 0:
        return attacked

    rng = np.random.default_rng(
        seed
    )

    flat_indexes = rng.choice(
        height * width,
        size=min(
            affected_pixels,
            height * width,
        ),
        replace=False,
    )

    salt_count = int(
        round(
            len(flat_indexes)
            * salt_ratio
        )
    )

    salt_indexes = flat_indexes[
        :salt_count
    ]

    pepper_indexes = flat_indexes[
        salt_count:
    ]

    salt_rows, salt_columns = np.unravel_index(
        salt_indexes,
        (height, width),
    )

    pepper_rows, pepper_columns = np.unravel_index(
        pepper_indexes,
        (height, width),
    )

    if attacked.ndim == 2:
        attacked[
            salt_rows,
            salt_columns,
        ] = 255

        attacked[
            pepper_rows,
            pepper_columns,
        ] = 0

    else:
        attacked[
            salt_rows,
            salt_columns,
            :,
        ] = 255

        attacked[
            pepper_rows,
            pepper_columns,
            :,
        ] = 0

    return attacked


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 3,
    *,
    sigma_x: float = 0.0,
) -> np.ndarray:
    """
    Apply Gaussian blur.
    """
    image_uint8 = validate_image(
        image
    )

    kernel_size = int(
        kernel_size
    )

    if kernel_size <= 0:
        raise ValueError(
            "Blur kernel size must be positive."
        )

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(
        image_uint8,
        (
            kernel_size,
            kernel_size,
        ),
        sigmaX=float(
            sigma_x
        ),
    )


def apply_rotation(
    image: np.ndarray,
    angle: float = 5.0,
    *,
    border_mode: int = cv2.BORDER_REFLECT,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Rotate while preserving the original image dimensions.
    """
    image_uint8 = validate_image(
        image
    )

    angle = float(
        angle
    )

    height, width = image_uint8.shape[:2]

    center = (
        width / 2.0,
        height / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    return cv2.warpAffine(
        image_uint8,
        matrix,
        (
            width,
            height,
        ),
        flags=interpolation,
        borderMode=border_mode,
    )


def apply_crop(
    image: np.ndarray,
    retain_ratio: float = 0.90,
    *,
    resize_back: bool = True,
) -> np.ndarray:
    """
    Apply centered cropping.

    retain_ratio=0.90 retains 90% of width and height.
    """
    image_uint8 = validate_image(
        image
    )

    retain_ratio = float(
        retain_ratio
    )

    if not 0.0 < retain_ratio <= 1.0:
        raise ValueError(
            "Crop retain_ratio must be in (0, 1]."
        )

    height, width = image_uint8.shape[:2]

    crop_width = max(
        1,
        int(
            round(
                width * retain_ratio
            )
        ),
    )

    crop_height = max(
        1,
        int(
            round(
                height * retain_ratio
            )
        ),
    )

    x_start = (
        width - crop_width
    ) // 2

    y_start = (
        height - crop_height
    ) // 2

    cropped = image_uint8[
        y_start:y_start + crop_height,
        x_start:x_start + crop_width,
    ]

    if not resize_back:
        return cropped.copy()

    return cv2.resize(
        cropped,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )


def apply_brightness(
    image: np.ndarray,
    value: float = 20.0,
) -> np.ndarray:
    """
    Add a brightness offset.
    """
    image_uint8 = validate_image(
        image
    )

    attacked = (
        image_uint8.astype(np.float32)
        + float(value)
    )

    return np.uint8(
        np.clip(
            np.rint(attacked),
            0,
            255,
        )
    )


def apply_contrast(
    image: np.ndarray,
    value: float = 20.0,
) -> np.ndarray:
    """
    Adjust contrast.

    The default interpretation is percentage:
    value=20 means a factor of 1.20.
    Negative values reduce contrast.
    """
    image_uint8 = validate_image(
        image
    )

    value = float(
        value
    )

    factor = (
        1.0 + value / 100.0
        if abs(value) > 2.0
        else value
    )

    if factor < 0:
        raise ValueError(
            "Contrast factor cannot be negative."
        )

    attacked = (
        (
            image_uint8.astype(np.float32)
            - 127.5
        )
        * factor
        + 127.5
    )

    return np.uint8(
        np.clip(
            np.rint(attacked),
            0,
            255,
        )
    )


def apply_gamma(
    image: np.ndarray,
    gamma: float = 1.2,
) -> np.ndarray:
    """
    Apply gamma correction.
    """
    image_uint8 = validate_image(
        image
    )

    gamma = float(
        gamma
    )

    if gamma <= 0:
        raise ValueError(
            "Gamma must be greater than zero."
        )

    normalized = (
        image_uint8.astype(np.float32)
        / 255.0
    )

    corrected = np.power(
        normalized,
        gamma,
    )

    return np.uint8(
        np.clip(
            np.rint(
                corrected * 255.0
            ),
            0,
            255,
        )
    )


def apply_sharpen(
    image: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Apply unsharp masking.
    """
    image_uint8 = validate_image(
        image
    )

    strength = float(
        strength
    )

    if strength < 0:
        raise ValueError(
            "Sharpen strength cannot be negative."
        )

    blurred = cv2.GaussianBlur(
        image_uint8,
        (3, 3),
        0,
    )

    sharpened = cv2.addWeighted(
        image_uint8,
        1.0 + strength,
        blurred,
        -strength,
        0,
    )

    return np.uint8(
        np.clip(
            sharpened,
            0,
            255,
        )
    )


# ============================================================
# ATTACK DISPATCH
# ============================================================

def apply_attack_array(
    image: np.ndarray,
    attack_type: str,
    parameter: Any = None,
    **options: Any,
) -> np.ndarray:
    """
    Apply one attack and return only the processed image.
    """
    normalized = normalize_attack_name(
        attack_type
    )

    if normalized == "none":
        return apply_no_attack(
            image
        )

    if normalized == "jpeg":
        quality = (
            70
            if parameter is None
            else parameter
        )

        return apply_jpeg_compression(
            image,
            quality=int(
                quality
            ),
        )

    if normalized == "gaussian_noise":
        sigma = (
            0.01
            if parameter is None
            else parameter
        )

        return apply_gaussian_noise(
            image,
            sigma=float(
                sigma
            ),
            seed=options.get(
                "seed",
                42,
            ),
        )

    if normalized == "salt_pepper":
        amount = (
            0.05
            if parameter is None
            else parameter
        )

        return apply_salt_pepper_noise(
            image,
            amount=float(
                amount
            ),
            salt_ratio=float(
                options.get(
                    "salt_ratio",
                    0.5,
                )
            ),
            seed=options.get(
                "seed",
                42,
            ),
        )

    if normalized == "gaussian_blur":
        kernel_size = (
            3
            if parameter is None
            else parameter
        )

        return apply_gaussian_blur(
            image,
            kernel_size=int(
                kernel_size
            ),
            sigma_x=float(
                options.get(
                    "sigma_x",
                    0.0,
                )
            ),
        )

    if normalized == "rotation":
        angle = (
            5.0
            if parameter is None
            else parameter
        )

        return apply_rotation(
            image,
            angle=float(
                angle
            ),
        )

    if normalized == "crop":
        retain_ratio = (
            0.90
            if parameter is None
            else parameter
        )

        return apply_crop(
            image,
            retain_ratio=float(
                retain_ratio
            ),
            resize_back=bool(
                options.get(
                    "resize_back",
                    True,
                )
            ),
        )

    if normalized == "brightness":
        value = (
            20.0
            if parameter is None
            else parameter
        )

        return apply_brightness(
            image,
            value=float(
                value
            ),
        )

    if normalized == "contrast":
        value = (
            20.0
            if parameter is None
            else parameter
        )

        return apply_contrast(
            image,
            value=float(
                value
            ),
        )

    if normalized == "gamma":
        gamma = (
            1.2
            if parameter is None
            else parameter
        )

        return apply_gamma(
            image,
            gamma=float(
                gamma
            ),
        )

    if normalized == "sharpen":
        strength = (
            1.0
            if parameter is None
            else parameter
        )

        return apply_sharpen(
            image,
            strength=float(
                strength
            ),
        )

    raise ValueError(
        "Combined attacks must be supplied through "
        "apply_combined_attacks()."
    )


def apply_attack(
    image: np.ndarray,
    attack_type: str,
    parameter: Any = None,
    **options: Any,
) -> AttackResult:
    """
    Apply one named attack and return a complete result.
    """
    normalized = normalize_attack_name(
        attack_type
    )

    if normalized == "combined":
        raise ValueError(
            "Use apply_combined_attacks() for combined attacks."
        )

    started_at = time.perf_counter()

    attacked = apply_attack_array(
        image,
        normalized,
        parameter,
        **options,
    )

    runtime = (
        time.perf_counter()
        - started_at
    )

    return AttackResult(
        image=attacked,
        attack_type=normalized,
        parameter=parameter,
        runtime_seconds=runtime,
        steps=[
            {
                "attack_type": normalized,
                "parameter": parameter,
                "options": options,
            }
        ],
        metadata={
            "original_shape": list(
                image.shape
            ),
            "output_shape": list(
                attacked.shape
            ),
            "shape_preserved": (
                image.shape
                == attacked.shape
            ),
        },
    )


# ============================================================
# COMBINED ATTACKS
# ============================================================

def apply_combined_attacks(
    image: np.ndarray,
    steps: Sequence[
        AttackStep | Dict[str, Any]
    ],
) -> AttackResult:
    """
    Apply multiple attacks sequentially.

    Example
    -------
    steps = [
        AttackStep("jpeg", 70),
        AttackStep("gaussian_blur", 3),
        AttackStep("gaussian_noise", 0.01),
    ]
    """
    if not steps:
        raise ValueError(
            "Combined attack must contain at least one step."
        )

    current = validate_image(
        image
    )

    normalized_steps: List[
        Dict[str, Any]
    ] = []

    started_at = time.perf_counter()

    for sequence_order, step in enumerate(
        steps
    ):
        if isinstance(
            step,
            AttackStep,
        ):
            attack_type = step.attack_type
            parameter = step.parameter
            options = dict(
                step.options
            )

        elif isinstance(
            step,
            dict,
        ):
            attack_type = step.get(
                "attack_type",
                step.get(
                    "type",
                ),
            )

            parameter = step.get(
                "parameter"
            )

            options = dict(
                step.get(
                    "options",
                    {},
                )
            )

        else:
            raise TypeError(
                "Combined attack steps must be AttackStep "
                "objects or dictionaries."
            )

        normalized_type = normalize_attack_name(
            attack_type
        )

        if normalized_type in (
            "combined",
            "none",
        ):
            if normalized_type == "none":
                normalized_steps.append({
                    "sequence_order": sequence_order,
                    "attack_type": "none",
                    "parameter": parameter,
                    "options": options,
                })

                continue

            raise ValueError(
                "Nested combined attacks are not supported."
            )

        current = apply_attack_array(
            current,
            normalized_type,
            parameter,
            **options,
        )

        normalized_steps.append({
            "sequence_order": sequence_order,
            "attack_type": normalized_type,
            "parameter": parameter,
            "options": options,
        })

    runtime = (
        time.perf_counter()
        - started_at
    )

    return AttackResult(
        image=current,
        attack_type="combined",
        parameter=None,
        runtime_seconds=runtime,
        steps=normalized_steps,
        metadata={
            "step_count": len(
                normalized_steps
            ),
            "original_shape": list(
                image.shape
            ),
            "output_shape": list(
                current.shape
            ),
            "shape_preserved": (
                image.shape
                == current.shape
            ),
        },
    )


# ============================================================
# ATTACK PIPELINE CLASS
# ============================================================

class AttackPipeline:
    """
    Reusable object-oriented attack interface.
    """

    def __init__(
        self,
        *,
        default_seed: Optional[int] = 42,
    ) -> None:
        self.default_seed = default_seed

    def apply(
        self,
        image: np.ndarray,
        attack_type: str,
        parameter: Any = None,
        **options: Any,
    ) -> AttackResult:
        if (
            "seed" not in options
            and self.default_seed is not None
        ):
            options["seed"] = self.default_seed

        return apply_attack(
            image,
            attack_type,
            parameter,
            **options,
        )

    def apply_combined(
        self,
        image: np.ndarray,
        steps: Sequence[
            AttackStep | Dict[str, Any]
        ],
    ) -> AttackResult:
        prepared_steps = []

        for step in steps:
            if isinstance(
                step,
                AttackStep,
            ):
                options = dict(
                    step.options
                )

                if (
                    "seed" not in options
                    and self.default_seed
                    is not None
                ):
                    options["seed"] = (
                        self.default_seed
                    )

                prepared_steps.append(
                    AttackStep(
                        attack_type=step.attack_type,
                        parameter=step.parameter,
                        options=options,
                    )
                )

            else:
                copied = dict(
                    step
                )

                options = dict(
                    copied.get(
                        "options",
                        {},
                    )
                )

                if (
                    "seed" not in options
                    and self.default_seed
                    is not None
                ):
                    options["seed"] = (
                        self.default_seed
                    )

                copied["options"] = options
                prepared_steps.append(
                    copied
                )

        return apply_combined_attacks(
            image,
            prepared_steps,
        )

    def supported_attacks(
        self,
    ) -> List[str]:
        return list(
            SUPPORTED_ATTACKS
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(
        seed=42
    )

    test_image = rng.integers(
        0,
        256,
        size=(512, 512),
        dtype=np.uint8,
    )

    pipeline = AttackPipeline(
        default_seed=42
    )

    results = []

    attack_cases = [
        ("none", None),
        ("jpeg", 70),
        ("gaussian_noise", 0.01),
        ("salt_pepper", 0.05),
        ("gaussian_blur", 3),
        ("rotation", 5),
        ("crop", 0.90),
        ("brightness", 20),
        ("contrast", 20),
        ("gamma", 1.2),
        ("sharpen", 1.0),
    ]

    for attack_type, parameter in attack_cases:
        result = pipeline.apply(
            test_image,
            attack_type,
            parameter,
        )

        assert result.image.shape == (
            test_image.shape
        )

        assert result.image.dtype == np.uint8

        results.append(
            result
        )

    combined = pipeline.apply_combined(
        test_image,
        [
            AttackStep(
                "jpeg",
                70,
            ),
            AttackStep(
                "gaussian_blur",
                3,
            ),
            AttackStep(
                "gaussian_noise",
                0.01,
            ),
        ],
    )

    assert combined.image.shape == (
        test_image.shape
    )

    print("=" * 72)
    print("AWSRE ATTACK PIPELINE SELF TEST")
    print("=" * 72)

    print(
        "Individual attacks tested:",
        len(results),
    )

    print(
        "Combined attack steps:",
        len(combined.steps),
    )

    print(
        "Supported attacks:",
        pipeline.supported_attacks(),
    )

    print(
        "\n✅ Attack pipeline self test passed."
    )
