"""
AWSRE Watermarking Registry

This module provides a central registry and factory interface
for all watermarking algorithms supported by AWSRE.

The registry allows the rest of the framework to use a common API:

    create_watermarker(method="DCT", alpha=20)
    embed(method="DCT", host=host, watermark_data=watermark, alpha=20)
    extract(
        method="DCT",
        original=host,
        watermarked=watermarked,
        watermark_shape=(32, 32),
        alpha=20,
    )

No Streamlit code is used in this module.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Type

from watermarking.base import BaseWatermarker


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class WatermarkerRegistryError(Exception):
    """
    Base exception for watermarking registry errors.
    """


class WatermarkerNotFoundError(WatermarkerRegistryError):
    """
    Raised when a requested method is not registered.
    """


class WatermarkerAlreadyRegisteredError(WatermarkerRegistryError):
    """
    Raised when another class is already registered
    under the same method name.
    """


# ============================================================
# INTERNAL REGISTRY
# ============================================================

_WATERMARKER_REGISTRY: Dict[str, Type[BaseWatermarker]] = {}


BUILTIN_WATERMARKER_MODULES = (
    "watermarking.dct",
    "watermarking.dwt",
    "watermarking.dct_svd",
    "watermarking.dwt_svd",
    "watermarking.block_svd",
)


# ============================================================
# METHOD NAME NORMALIZATION
# ============================================================

def normalize_method_name(method: str) -> str:
    """
    Normalize a watermarking method name.

    Examples
    --------
    "dct"       -> "DCT"
    "dct_svd"   -> "DCT-SVD"
    "dwt svd"   -> "DWT-SVD"
    "blocks-vd" -> normalized input form
    """
    if not isinstance(method, str):
        raise TypeError(
            "Watermarking method name must be a string."
        )

    normalized = method.strip().upper()

    if not normalized:
        raise ValueError(
            "Watermarking method name cannot be empty."
        )

    normalized = normalized.replace("_", "-")
    normalized = normalized.replace(" ", "-")

    while "--" in normalized:
        normalized = normalized.replace("--", "-")

    aliases = {
        "DCTSVD": "DCT-SVD",
        "DWTSVD": "DWT-SVD",
        "BLOCKSVD": "BLOCK-SVD",
        "BLOCK-SVD": "BLOCK-SVD",
    }

    return aliases.get(normalized, normalized)


# ============================================================
# REGISTRATION
# ============================================================

def register_watermarker(
    watermarker_class: Type[BaseWatermarker],
    *,
    overwrite: bool = False,
) -> Type[BaseWatermarker]:
    """
    Register a watermarking class.

    The supplied class must:

    - inherit from BaseWatermarker;
    - define a valid METHOD attribute;
    - use a unique method name.

    Re-registering the exact same class is treated as a
    successful no-op. This makes module imports idempotent.

    Parameters
    ----------
    watermarker_class:
        Class to register.
    overwrite:
        Replace another class already registered under the
        same method name.

    Returns
    -------
    Type[BaseWatermarker]
        Registered class.
    """
    if not isinstance(watermarker_class, type):
        raise TypeError(
            "Registered watermarker must be a class."
        )

    if not issubclass(
        watermarker_class,
        BaseWatermarker,
    ):
        raise TypeError(
            "Registered class must inherit from BaseWatermarker."
        )

    raw_method = getattr(
        watermarker_class,
        "METHOD",
        "",
    )

    method = normalize_method_name(
        raw_method
    )

    if method == "BASE":
        raise ValueError(
            "A concrete watermarker must define a METHOD "
            "different from 'BASE'."
        )

    existing_class = _WATERMARKER_REGISTRY.get(
        method
    )

    if existing_class is not None:
        # Same class imported more than once:
        # registration is already correct.
        if existing_class is watermarker_class:
            return watermarker_class

        # In some Python execution modes, the same logical class
        # may be loaded as a new class object. Treat it as equivalent
        # when module and class names are identical.
        same_logical_class = (
            existing_class.__name__
            == watermarker_class.__name__
            and existing_class.__module__
            == watermarker_class.__module__
        )

        if same_logical_class:
            _WATERMARKER_REGISTRY[
                method
            ] = watermarker_class

            return watermarker_class

        if not overwrite:
            raise WatermarkerAlreadyRegisteredError(
                f"Method '{method}' is already registered by "
                f"{existing_class.__module__}."
                f"{existing_class.__name__}."
            )

    _WATERMARKER_REGISTRY[
        method
    ] = watermarker_class

    return watermarker_class


def watermarker(
    watermarker_class: Type[BaseWatermarker],
) -> Type[BaseWatermarker]:
    """
    Class decorator for registering a watermarker.

    Example
    -------
    @watermarker
    class DCTWatermarker(BaseWatermarker):
        METHOD = "DCT"
    """
    return register_watermarker(
        watermarker_class
    )


def unregister_watermarker(
    method: str,
) -> bool:
    """
    Remove a method from the registry.

    Returns
    -------
    bool
        True when the method existed and was removed.
    """
    normalized = normalize_method_name(
        method
    )

    if normalized not in _WATERMARKER_REGISTRY:
        return False

    del _WATERMARKER_REGISTRY[
        normalized
    ]

    return True


def clear_registry() -> None:
    """
    Remove every registered method.

    Mainly intended for isolated tests.
    """
    _WATERMARKER_REGISTRY.clear()


# ============================================================
# LOOKUP
# ============================================================

def is_registered(
    method: str,
) -> bool:
    """
    Return whether a method is registered.
    """
    normalized = normalize_method_name(
        method
    )

    return normalized in _WATERMARKER_REGISTRY


def get_watermarker_class(
    method: str,
) -> Type[BaseWatermarker]:
    """
    Return the registered class for a method.

    Raises
    ------
    WatermarkerNotFoundError
        When no class is registered under the requested name.
    """
    normalized = normalize_method_name(
        method
    )

    watermarker_class = _WATERMARKER_REGISTRY.get(
        normalized
    )

    if watermarker_class is None:
        available_methods = list_registered_methods()

        available_text = (
            ", ".join(available_methods)
            if available_methods
            else "none"
        )

        raise WatermarkerNotFoundError(
            f"Unknown watermarking method: '{normalized}'. "
            f"Registered methods: {available_text}."
        )

    return watermarker_class


# ============================================================
# FACTORY
# ============================================================

def create_watermarker(
    method: str,
    alpha: float | None = None,
    **kwargs: Any,
) -> BaseWatermarker:
    """
    Create a registered watermarker instance.

    Built-in methods are loaded automatically when necessary.

    Parameters
    ----------
    method:
        Method name, for example DCT or DWT-SVD.
    alpha:
        Embedding strength. When omitted, the algorithm's
        DEFAULT_ALPHA value is used.
    kwargs:
        Additional algorithm-specific constructor options.

    Returns
    -------
    BaseWatermarker
        Initialized algorithm instance.
    """
    normalized = normalize_method_name(
        method
    )

    if normalized not in _WATERMARKER_REGISTRY:
        load_builtin_watermarkers()

    watermarker_class = get_watermarker_class(
        normalized
    )

    if alpha is None:
        alpha = getattr(
            watermarker_class,
            "DEFAULT_ALPHA",
            10,
        )

    instance = watermarker_class(
        alpha=alpha,
        **kwargs,
    )

    if not isinstance(
        instance,
        BaseWatermarker,
    ):
        raise TypeError(
            "Factory created an invalid watermarker instance."
        )

    return instance


# ============================================================
# REGISTRY INFORMATION
# ============================================================

def list_registered_methods() -> List[str]:
    """
    Return registered method names alphabetically.
    """
    return sorted(
        _WATERMARKER_REGISTRY.keys()
    )


def registry_size() -> int:
    """
    Return the number of registered methods.
    """
    return len(
        _WATERMARKER_REGISTRY
    )


def get_registry_info() -> List[Dict[str, Any]]:
    """
    Return descriptive metadata for registered algorithms.
    """
    rows: List[Dict[str, Any]] = []

    for method in list_registered_methods():
        algorithm_class = _WATERMARKER_REGISTRY[
            method
        ]

        rows.append({
            "method": method,
            "class_name": algorithm_class.__name__,
            "module": algorithm_class.__module__,
            "description": getattr(
                algorithm_class,
                "DESCRIPTION",
                "",
            ),
            "default_alpha": getattr(
                algorithm_class,
                "DEFAULT_ALPHA",
                None,
            ),
            "supports_grayscale": getattr(
                algorithm_class,
                "SUPPORTS_GRAYSCALE",
                False,
            ),
            "supports_rgb": getattr(
                algorithm_class,
                "SUPPORTS_RGB",
                False,
            ),
        })

    return rows


# ============================================================
# HIGH-LEVEL FRAMEWORK API
# ============================================================

def embed(
    method: str,
    host,
    watermark_data,
    alpha: float | None = None,
    **kwargs: Any,
):
    """
    Embed a watermark through the central registry.
    """
    algorithm = create_watermarker(
        method=method,
        alpha=alpha,
        **kwargs,
    )

    return algorithm.embed(
        host,
        watermark_data,
    )


def extract(
    method: str,
    original,
    watermarked,
    watermark_shape,
    alpha: float | None = None,
    **kwargs: Any,
):
    """
    Extract a watermark through the central registry.
    """
    algorithm = create_watermarker(
        method=method,
        alpha=alpha,
        **kwargs,
    )

    return algorithm.extract(
        original,
        watermarked,
        watermark_shape,
    )


# ============================================================
# BUILT-IN ALGORITHM LOADING
# ============================================================

def load_builtin_watermarkers(
    *,
    strict: bool = False,
) -> List[str]:
    """
    Import all implemented AWSRE watermarking modules.

    During early development, empty or not-yet-implemented method
    modules may exist. Import failures caused by those modules can
    be skipped when strict=False.

    Parameters
    ----------
    strict:
        Re-raise import errors when True.

    Returns
    -------
    List[str]
        Registered method names after loading.
    """
    for module_name in BUILTIN_WATERMARKER_MODULES:
        try:
            importlib.import_module(
                module_name
            )

        except (
            ImportError,
            AttributeError,
        ):
            if strict:
                raise

            continue

    return list_registered_methods()


def reload_builtin_watermarkers() -> List[str]:
    """
    Reload already imported built-in method modules.

    Intended mainly for notebook development and debugging.
    """
    for module_name in BUILTIN_WATERMARKER_MODULES:
        try:
            module = importlib.import_module(
                module_name
            )

            importlib.reload(
                module
            )

        except (
            ImportError,
            AttributeError,
        ):
            continue

    return list_registered_methods()


# ============================================================
# DIAGNOSTICS
# ============================================================

def registry_diagnostics() -> Dict[str, Any]:
    """
    Return a diagnostic snapshot of the registry.
    """
    return {
        "registered_count": registry_size(),
        "registered_methods": list_registered_methods(),
        "algorithms": get_registry_info(),
        "builtin_modules": list(
            BUILTIN_WATERMARKER_MODULES
        ),
    }
