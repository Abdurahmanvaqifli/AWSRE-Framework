"""
AWSRE Watermarking Registry

This module provides a central registry and factory for all
watermarking algorithms supported by AWSRE.

The rest of the platform does not need to import DCT, DWT,
DCT-SVD, DWT-SVD or Block-SVD directly.

Example
-------
from watermarking.registry import create_watermarker

watermarker = create_watermarker(
    method="DCT",
    alpha=10
)

embedding_result = watermarker.embed(
    host,
    watermark
)
"""

from __future__ import annotations

from typing import Dict, List, Type

from watermarking.base import BaseWatermarker


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class WatermarkerRegistryError(Exception):
    """
    Base exception for registry-related errors.
    """


class WatermarkerNotFoundError(WatermarkerRegistryError):
    """
    Raised when an unknown watermarking method is requested.
    """


class WatermarkerAlreadyRegisteredError(WatermarkerRegistryError):
    """
    Raised when a method name is registered more than once.
    """


# ============================================================
# INTERNAL REGISTRY
# ============================================================

_WATERMARKER_REGISTRY: Dict[str, Type[BaseWatermarker]] = {}


def normalize_method_name(method: str) -> str:
    """
    Normalize watermarking method names.

    Examples
    --------
    dct-svd -> DCT-SVD
    block_svd -> BLOCK-SVD
    Dwt Svd -> DWT-SVD
    """
    if not isinstance(method, str):
        raise TypeError("Method name must be a string.")

    normalized = method.strip().upper()

    normalized = normalized.replace("_", "-")
    normalized = normalized.replace(" ", "-")

    while "--" in normalized:
        normalized = normalized.replace("--", "-")

    aliases = {
        "BLOCKSVD": "BLOCK-SVD",
        "DCTSVD": "DCT-SVD",
        "DWTSVD": "DWT-SVD",
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

    The class must inherit from BaseWatermarker and define
    a non-empty METHOD attribute.

    Parameters
    ----------
    watermarker_class:
        Watermarker class to register.
    overwrite:
        Replace an existing registration when True.

    Returns
    -------
    Type[BaseWatermarker]
        The registered class.
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
            "Watermarker must inherit from BaseWatermarker."
        )

    method = normalize_method_name(
        getattr(watermarker_class, "METHOD", "")
    )

    if not method or method == "BASE":
        raise ValueError(
            "Watermarker class must define a valid METHOD attribute."
        )

  if method in _WATERMARKER_REGISTRY:
    existing_class = _WATERMARKER_REGISTRY[method]

    if existing_class is watermarker_class:
        return watermarker_class

    if not overwrite:
        raise WatermarkerAlreadyRegisteredError(
            f"Watermarker already registered: {method}"
        )

    _WATERMARKER_REGISTRY[method] = watermarker_class

    return watermarker_class


def watermarker(
    watermarker_class: Type[BaseWatermarker],
) -> Type[BaseWatermarker]:
    """
    Decorator for registering watermarking classes.

    Example
    -------
    @watermarker
    class DCTWatermarker(BaseWatermarker):
        METHOD = "DCT"
    """
    return register_watermarker(
        watermarker_class
    )


def unregister_watermarker(method: str) -> bool:
    """
    Remove a watermarking method from the registry.

    Returns True when the method existed.
    """
    normalized = normalize_method_name(method)

    if normalized not in _WATERMARKER_REGISTRY:
        return False

    del _WATERMARKER_REGISTRY[normalized]

    return True


def clear_registry() -> None:
    """
    Remove all registered watermarking methods.

    Mainly intended for testing.
    """
    _WATERMARKER_REGISTRY.clear()


# ============================================================
# LOOKUP AND FACTORY
# ============================================================

def is_registered(method: str) -> bool:
    """
    Check whether a method is registered.
    """
    normalized = normalize_method_name(method)

    return normalized in _WATERMARKER_REGISTRY


def get_watermarker_class(
    method: str,
) -> Type[BaseWatermarker]:
    """
    Return the class registered for a method.
    """
    normalized = normalize_method_name(method)

    if normalized not in _WATERMARKER_REGISTRY:
        available = ", ".join(
            sorted(_WATERMARKER_REGISTRY.keys())
        )

        if not available:
            available = "none"

        raise WatermarkerNotFoundError(
            f"Unknown watermarking method: {normalized}. "
            f"Available methods: {available}"
        )

    return _WATERMARKER_REGISTRY[normalized]


def create_watermarker(
    method: str,
    alpha: float | None = None,
    **kwargs,
) -> BaseWatermarker:
    """
    Create a watermarker instance.

    Parameters
    ----------
    method:
        Registered method name.
    alpha:
        Embedding strength. If omitted, the method's default
        alpha value is used.
    kwargs:
        Additional constructor parameters for future methods.

    Returns
    -------
    BaseWatermarker
        Initialized watermarker instance.
    """
    watermarker_class = get_watermarker_class(
        method
    )

    if alpha is None:
        alpha = watermarker_class.DEFAULT_ALPHA

    return watermarker_class(
        alpha=alpha,
        **kwargs,
    )


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
    return len(_WATERMARKER_REGISTRY)


def get_registry_info() -> List[dict]:
    """
    Return metadata for all registered methods.
    """
    info_rows = []

    for method in list_registered_methods():
        watermarker_class = _WATERMARKER_REGISTRY[
            method
        ]

        info_rows.append({
            "method": method,
            "class_name": watermarker_class.__name__,
            "description": getattr(
                watermarker_class,
                "DESCRIPTION",
                "",
            ),
            "default_alpha": getattr(
                watermarker_class,
                "DEFAULT_ALPHA",
                None,
            ),
            "supports_grayscale": getattr(
                watermarker_class,
                "SUPPORTS_GRAYSCALE",
                False,
            ),
            "supports_rgb": getattr(
                watermarker_class,
                "SUPPORTS_RGB",
                False,
            ),
        })

    return info_rows


# ============================================================
# HIGH-LEVEL API
# ============================================================

def embed(
    method: str,
    host,
    watermark_data,
    alpha: float | None = None,
    **kwargs,
):
    """
    Create the requested watermarker and embed a watermark.
    """
    instance = create_watermarker(
        method=method,
        alpha=alpha,
        **kwargs,
    )

    return instance.embed(
        host,
        watermark_data,
    )


def extract(
    method: str,
    original,
    watermarked,
    watermark_shape,
    alpha: float | None = None,
    **kwargs,
):
    """
    Create the requested watermarker and extract a watermark.
    """
    instance = create_watermarker(
        method=method,
        alpha=alpha,
        **kwargs,
    )

    return instance.extract(
        original,
        watermarked,
        watermark_shape,
    )


# ============================================================
# METHOD AUTO-LOADING
# ============================================================

def load_builtin_watermarkers() -> None:
    """
    Import built-in AWSRE algorithms.

    Importing each module causes its class to register through
    the @watermarker decorator.

    Modules that have not yet been implemented are skipped
    safely during early framework development.
    """
    module_names = [
        "watermarking.dct",
        "watermarking.dwt",
        "watermarking.dct_svd",
        "watermarking.dwt_svd",
        "watermarking.block_svd",
    ]

    for module_name in module_names:
        try:
            __import__(module_name)

        except ImportError as exc:
            missing_module = getattr(
                exc,
                "name",
                "",
            )

            # Skip only when the target algorithm module
            # itself is still missing.
            if missing_module == module_name:
                continue

            raise


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 68)
    print("AWSRE Watermarking Registry")
    print("=" * 68)

    load_builtin_watermarkers()

    print(
        "\nRegistered methods:",
        registry_size(),
    )

    methods = list_registered_methods()

    if not methods:
        print(
            "\nNo algorithms are registered yet. "
            "This is expected before dct.py is implemented."
        )
    else:
        for item in get_registry_info():
            print(
                f"- {item['method']} "
                f"({item['class_name']})"
            )
