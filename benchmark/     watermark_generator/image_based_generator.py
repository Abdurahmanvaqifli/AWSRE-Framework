"""
AWSRE Image-Based Watermark Generator Base

Shared preprocessing logic for image-based watermark types:

- Logo
- Signature
- Future seals, stamps and institutional identifiers

Responsibilities:
- accept file paths or NumPy arrays;
- load grayscale or color images;
- preserve aspect ratio;
- center images on a canvas;
- threshold or Otsu binarization;
- optional inversion;
- optional morphological cleanup;
- return binary values 0 and 1.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from benchmark.watermark_generator.base_generator import (
    BaseWatermarkGenerator,
    GeneratedWatermark,
    GenerationTimer,
)


class ImageBasedWatermarkGenerator(BaseWatermarkGenerator):
    """
    Abstract base for image-based watermark generators.
    """

    @staticmethod
    def load_source_image(
        source: str | Path | np.ndarray,
    ) -> tuple[np.ndarray, Optional[str]]:
        """
        Load source image from path or NumPy array.

        Returns
        -------
        image:
            Loaded source image.
        source_path:
            String path when file-based, otherwise None.
        """
        if isinstance(source, np.ndarray):
            if source.size == 0:
                raise ValueError(
                    "Source image array cannot be empty."
                )

            return source.copy(), None

        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(
                f"Source watermark image does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Source watermark path is not a file: {path}"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise ValueError(
                f"OpenCV could not load source watermark: {path}"
            )

        if image.size == 0:
            raise ValueError(
                f"Loaded source watermark is empty: {path}"
            )

        return image, str(path)

    @staticmethod
    def convert_to_grayscale(
        image: np.ndarray,
        *,
        alpha_background: int = 255,
    ) -> np.ndarray:
        """
        Convert grayscale, BGR or BGRA image to grayscale.

        Transparent pixels are composited over a configurable background.
        """
        if image.ndim == 2:
            return image.astype(np.uint8)

        if image.ndim != 3:
            raise ValueError(
                "Source watermark must be grayscale or multi-channel image."
            )

        channels = image.shape[2]

        if channels == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        if channels == 4:
            bgr = image[:, :, :3].astype(np.float32)
            alpha = (
                image[:, :, 3].astype(np.float32)
                / 255.0
            )

            background = np.full_like(
                bgr,
                float(alpha_background),
                dtype=np.float32,
            )

            alpha_3 = alpha[:, :, None]

            composited = (
                bgr * alpha_3
                + background * (1.0 - alpha_3)
            )

            return cv2.cvtColor(
                np.uint8(
                    np.clip(
                        composited,
                        0,
                        255,
                    )
                ),
                cv2.COLOR_BGR2GRAY,
            )

        raise ValueError(
            f"Unsupported channel count: {channels}."
        )

    @staticmethod
    def resize_preserving_aspect_ratio(
        image: np.ndarray,
        *,
        target_size: Tuple[int, int],
        padding: int = 0,
        background_value: int = 255,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Resize and center source image without distortion.
        """
        target_width = int(target_size[0])
        target_height = int(target_size[1])

        padding = max(
            0,
            int(padding),
        )

        available_width = target_width - 2 * padding
        available_height = target_height - 2 * padding

        if available_width <= 0 or available_height <= 0:
            raise ValueError(
                "Padding leaves no usable watermark area."
            )

        source_height, source_width = image.shape[:2]

        scale = min(
            available_width / source_width,
            available_height / source_height,
        )

        resized_width = max(
            1,
            int(round(source_width * scale)),
        )

        resized_height = max(
            1,
            int(round(source_height * scale)),
        )

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_CUBIC
        )

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )

        canvas = np.full(
            (target_height, target_width),
            int(background_value),
            dtype=np.uint8,
        )

        x_start = (
            target_width - resized_width
        ) // 2

        y_start = (
            target_height - resized_height
        ) // 2

        canvas[
            y_start:y_start + resized_height,
            x_start:x_start + resized_width,
        ] = resized

        metadata = {
            "source_width": int(source_width),
            "source_height": int(source_height),
            "resized_width": int(resized_width),
            "resized_height": int(resized_height),
            "scale": float(scale),
            "x_offset": int(x_start),
            "y_offset": int(y_start),
            "padding": int(padding),
        }

        return canvas, metadata

    @staticmethod
    def binarize_image(
        image: np.ndarray,
        *,
        threshold: Optional[int] = None,
        use_otsu: bool = True,
        foreground_dark: bool = True,
    ) -> tuple[np.ndarray, float]:
        """
        Convert grayscale image into binary 0/1 representation.

        When foreground_dark=True:
        - dark logo/signature strokes become 1;
        - light background becomes 0.
        """
        image_uint8 = np.uint8(
            np.clip(
                image,
                0,
                255,
            )
        )

        if use_otsu:
            threshold_value, binary_255 = cv2.threshold(
                image_uint8,
                0,
                255,
                (
                    cv2.THRESH_BINARY_INV
                    if foreground_dark
                    else cv2.THRESH_BINARY
                )
                + cv2.THRESH_OTSU,
            )

        else:
            if threshold is None:
                threshold = 127

            threshold_value, binary_255 = cv2.threshold(
                image_uint8,
                int(threshold),
                255,
                (
                    cv2.THRESH_BINARY_INV
                    if foreground_dark
                    else cv2.THRESH_BINARY
                ),
            )

        binary = (
            binary_255 > 0
        ).astype(np.uint8)

        return binary, float(threshold_value)

    @staticmethod
    def apply_morphology(
        binary: np.ndarray,
        *,
        operation: str = "none",
        kernel_size: int = 3,
        iterations: int = 1,
    ) -> np.ndarray:
        """
        Apply optional binary morphological cleanup.
        """
        normalized_operation = (
            str(operation)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        aliases = {
            "": "none",
            "no": "none",
            "off": "none",
            "open": "opening",
            "close": "closing",
        }

        normalized_operation = aliases.get(
            normalized_operation,
            normalized_operation,
        )

        supported = {
            "none",
            "opening",
            "closing",
            "erode",
            "dilate",
        }

        if normalized_operation not in supported:
            raise ValueError(
                f"Unsupported morphology operation: "
                f"{normalized_operation}."
            )

        if normalized_operation == "none":
            return binary.astype(np.uint8)

        kernel_size = max(
            1,
            int(kernel_size),
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        iterations = max(
            1,
            int(iterations),
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        binary_255 = (
            binary * 255
        ).astype(np.uint8)

        if normalized_operation == "opening":
            processed = cv2.morphologyEx(
                binary_255,
                cv2.MORPH_OPEN,
                kernel,
                iterations=iterations,
            )

        elif normalized_operation == "closing":
            processed = cv2.morphologyEx(
                binary_255,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=iterations,
            )

        elif normalized_operation == "erode":
            processed = cv2.erode(
                binary_255,
                kernel,
                iterations=iterations,
            )

        else:
            processed = cv2.dilate(
                binary_255,
                kernel,
                iterations=iterations,
            )

        return (
            processed > 0
        ).astype(np.uint8)

    def generate_from_image(
        self,
        *,
        size: Tuple[int, int],
        source: str | Path | np.ndarray,
        seed: Optional[int] = None,
        padding: int = 2,
        threshold: Optional[int] = None,
        use_otsu: bool = True,
        foreground_dark: bool = True,
        invert: bool = False,
        morphology: str = "none",
        morphology_kernel_size: int = 3,
        morphology_iterations: int = 1,
        alpha_background: int = 255,
        extra_metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> GeneratedWatermark:
        """
        Shared full image-based generation pipeline.
        """
        width, height = self.validate_size(
            size
        )

        resolved_seed = self.resolve_seed(
            seed
        )

        with GenerationTimer() as timer:
            source_image, source_path = (
                self.load_source_image(
                    source
                )
            )

            grayscale = self.convert_to_grayscale(
                source_image,
                alpha_background=alpha_background,
            )

            fitted, resize_metadata = (
                self.resize_preserving_aspect_ratio(
                    grayscale,
                    target_size=(width, height),
                    padding=padding,
                    background_value=(
                        255
                        if foreground_dark
                        else 0
                    ),
                )
            )

            binary, threshold_value = (
                self.binarize_image(
                    fitted,
                    threshold=threshold,
                    use_otsu=use_otsu,
                    foreground_dark=foreground_dark,
                )
            )

            binary = self.apply_morphology(
                binary,
                operation=morphology,
                kernel_size=morphology_kernel_size,
                iterations=morphology_iterations,
            )

            if invert:
                binary = (
                    1 - binary
                ).astype(np.uint8)

        if binary.shape != (
            height,
            width,
        ):
            raise RuntimeError(
                "Image-based generator returned incorrect dimensions."
            )

        foreground_pixels = int(
            np.sum(binary)
        )

        if foreground_pixels == 0:
            raise RuntimeError(
                "Generated watermark contains no foreground pixels."
            )

        metadata: Dict[str, Any] = {
            "source_path": source_path,
            "source_array_shape": list(
                source_image.shape
            ),
            "threshold_mode": (
                "otsu"
                if use_otsu
                else "manual"
            ),
            "threshold_value": float(
                threshold_value
            ),
            "requested_threshold": threshold,
            "foreground_dark": bool(
                foreground_dark
            ),
            "invert": bool(invert),
            "morphology": morphology,
            "morphology_kernel_size": int(
                morphology_kernel_size
            ),
            "morphology_iterations": int(
                morphology_iterations
            ),
            "alpha_background": int(
                alpha_background
            ),
            "foreground_pixels": foreground_pixels,
            "actual_density": float(
                np.mean(binary)
            ),
            **resize_metadata,
        }

        if extra_metadata:
            metadata.update(
                extra_metadata
            )

        return self.build_result(
            image=binary,
            size=(width, height),
            seed=resolved_seed,
            generation_time_seconds=timer.elapsed,
            metadata=metadata,
        )

    @abstractmethod
    def generate(
        self,
        size: Tuple[int, int],
        **kwargs: Any,
    ) -> GeneratedWatermark:
        """
        Implemented by Logo and Signature generators.
        """
