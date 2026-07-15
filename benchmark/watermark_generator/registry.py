"""
AWSRE Watermark Generator Registry

Central registry and factory for benchmark watermark generators.

Supported generators:

- Binary Pattern
- Text
- QR Code
- Logo
- Signature

The benchmark runner uses this module instead of importing
individual generator classes directly.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Type

from benchmark.watermark_generator.base_generator import (
    BaseWatermarkGenerator,
    WatermarkType,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class GeneratorRegistryError(Exception):
    """
    Base exception for generator registry errors.
    """


class GeneratorNotFoundError(GeneratorRegistryError):
    """
    Raised when a requested generator is not registered.
    """


class GeneratorAlreadyRegisteredError(
    GeneratorRegistryError
):
    """
    Raised when a different class is registered under
    an existing watermark type.
    """


# ============================================================
# INTERNAL REGISTRY
# ============================================================

_GENERATOR_REGISTRY: Dict[
    str,
    Type[BaseWatermarkGenerator],
] = {}


BUILTIN_GENERATOR_MODULES = (
    "benchmark.watermark_generator.binary_pattern",
    "benchmark.watermark_generator.text_generator",
    "benchmark.watermark_generator.qr_generator",
    "benchmark.watermark_generator.logo_generator",
    "benchmark.watermark_generator.signature_generator",
)


# ============================================================
# TYPE NORMALIZATION
# ============================================================

def normalize_watermark_type(
    watermark_type: str | WatermarkType,
) -> str:
    """
    Normalize a watermark type.

    Examples
    --------
    binary -> Binary Pattern
    qr -> QR Code
    text -> Text
    """
    if isinstance(
        watermark_type,
        WatermarkType,
    ):
        return watermark_type.value

    if not isinstance(
        watermark_type,
        str,
    ):
        raise TypeError(
            "watermark_type must be a string or WatermarkType."
        )

    normalized = (
        watermark_type
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    while "  " in normalized:
        normalized = normalized.replace(
            "  ",
            " ",
        )

    aliases = {
        "binary": WatermarkType.BINARY_PATTERN.value,
        "binary pattern": WatermarkType.BINARY_PATTERN.value,
        "pattern": WatermarkType.BINARY_PATTERN.value,

        "text": WatermarkType.TEXT.value,
        "text watermark": WatermarkType.TEXT.value,

        "qr": WatermarkType.QR_CODE.value,
        "qr code": WatermarkType.QR_CODE.value,
        "qrcode": WatermarkType.QR_CODE.value,

        "logo": WatermarkType.LOGO.value,
        "logo watermark": WatermarkType.LOGO.value,

        "signature": WatermarkType.SIGNATURE.value,
        "signature watermark": WatermarkType.SIGNATURE.value,
    }

    if normalized not in aliases:
        supported = ", ".join(
            item.value
            for item in WatermarkType
        )

        raise ValueError(
            f"Unsupported watermark type: {watermark_type}. "
            f"Supported types: {supported}."
        )

    return aliases[normalized]


# ============================================================
# REGISTRATION
# ============================================================

def register_generator(
    generator_class: Type[
        BaseWatermarkGenerator
    ],
    *,
    overwrite: bool = False,
) -> Type[BaseWatermarkGenerator]:
    """
    Register a watermark generator class.
    """
    if not isinstance(
        generator_class,
        type,
    ):
        raise TypeError(
            "Registered generator must be a class."
        )

    if not issubclass(
        generator_class,
        BaseWatermarkGenerator,
    ):
        raise TypeError(
            "Generator must inherit from "
            "BaseWatermarkGenerator."
        )

    watermark_type = normalize_watermark_type(
        getattr(
            generator_class,
            "WATERMARK_TYPE",
            "",
        )
    )

    existing_class = _GENERATOR_REGISTRY.get(
        watermark_type
    )

    if existing_class is not None:
        if existing_class is generator_class:
            return generator_class

        same_logical_class = (
            existing_class.__name__
            == generator_class.__name__
            and existing_class.__module__
            == generator_class.__module__
        )

        if same_logical_class:
            _GENERATOR_REGISTRY[
                watermark_type
            ] = generator_class

            return generator_class

        if not overwrite:
            raise GeneratorAlreadyRegisteredError(
                f"A generator is already registered for "
                f"'{watermark_type}': "
                f"{existing_class.__name__}."
            )

    _GENERATOR_REGISTRY[
        watermark_type
    ] = generator_class

    return generator_class


def watermark_generator(
    generator_class: Type[
        BaseWatermarkGenerator
    ],
) -> Type[BaseWatermarkGenerator]:
    """
    Registration decorator.

    Example
    -------
    @watermark_generator
    class ExampleGenerator(BaseWatermarkGenerator):
        ...
    """
    return register_generator(
        generator_class
    )


def unregister_generator(
    watermark_type: str | WatermarkType,
) -> bool:
    """
    Remove a generator registration.
    """
    normalized = normalize_watermark_type(
        watermark_type
    )

    if normalized not in _GENERATOR_REGISTRY:
        return False

    del _GENERATOR_REGISTRY[
        normalized
    ]

    return True


def clear_generator_registry() -> None:
    """
    Remove all registered generators.
    """
    _GENERATOR_REGISTRY.clear()


# ============================================================
# BUILT-IN LOADING
# ============================================================

def load_builtin_generators(
    *,
    strict: bool = False,
) -> List[str]:
    """
    Import and register all built-in generators.
    """
    for module_name in BUILTIN_GENERATOR_MODULES:
        try:
            module = importlib.import_module(
                module_name
            )

            generator_classes = [
                value
                for value in vars(module).values()
                if (
                    isinstance(value, type)
                    and issubclass(
                        value,
                        BaseWatermarkGenerator,
                    )
                    and value
                    is not BaseWatermarkGenerator
                    and getattr(
                        value,
                        "WATERMARK_TYPE",
                        None,
                    )
                    is not None
                    and not getattr(
                        value,
                        "__abstractmethods__",
                        set(),
                    )
                )
            ]

            for generator_class in generator_classes:
                register_generator(
                    generator_class
                )

        except Exception:
            if strict:
                raise

    return list_registered_generators()


# ============================================================
# LOOKUP AND FACTORY
# ============================================================

def is_generator_registered(
    watermark_type: str | WatermarkType,
) -> bool:
    """
    Check whether a generator is registered.
    """
    normalized = normalize_watermark_type(
        watermark_type
    )

    return normalized in _GENERATOR_REGISTRY


def get_generator_class(
    watermark_type: str | WatermarkType,
) -> Type[BaseWatermarkGenerator]:
    """
    Return the registered generator class.
    """
    normalized = normalize_watermark_type(
        watermark_type
    )

    if normalized not in _GENERATOR_REGISTRY:
        load_builtin_generators()

    generator_class = _GENERATOR_REGISTRY.get(
        normalized
    )

    if generator_class is None:
        available = ", ".join(
            list_registered_generators()
        )

        if not available:
            available = "none"

        raise GeneratorNotFoundError(
            f"No generator registered for "
            f"'{normalized}'. Available: {available}."
        )

    return generator_class


def create_generator(
    watermark_type: str | WatermarkType,
    *,
    default_seed: int | None = 42,
    **kwargs: Any,
) -> BaseWatermarkGenerator:
    """
    Create a watermark generator instance.
    """
    generator_class = get_generator_class(
        watermark_type
    )

    return generator_class(
        default_seed=default_seed,
        **kwargs,
    )


# ============================================================
# HIGH-LEVEL GENERATION API
# ============================================================

def generate_watermark(
    watermark_type: str | WatermarkType,
    size: tuple[int, int],
    *,
    default_seed: int | None = 42,
    **generation_kwargs: Any,
):
    """
    Create a generator and generate one watermark.
    """
    generator = create_generator(
        watermark_type,
        default_seed=default_seed,
    )

    return generator.generate(
        size=size,
        **generation_kwargs,
    )


# ============================================================
# INFORMATION
# ============================================================

def list_registered_generators() -> List[str]:
    """
    Return registered watermark types alphabetically.
    """
    return sorted(
        _GENERATOR_REGISTRY.keys()
    )


def generator_registry_size() -> int:
    """
    Return number of registered generators.
    """
    return len(
        _GENERATOR_REGISTRY
    )


def get_generator_registry_info() -> List[
    Dict[str, Any]
]:
    """
    Return metadata for all registered generators.
    """
    rows = []

    for watermark_type in (
        list_registered_generators()
    ):
        generator_class = _GENERATOR_REGISTRY[
            watermark_type
        ]

        rows.append({
            "watermark_type": watermark_type,
            "class_name": generator_class.__name__,
            "module": generator_class.__module__,
            "generator_name": getattr(
                generator_class,
                "GENERATOR_NAME",
                "",
            ),
            "description": getattr(
                generator_class,
                "DESCRIPTION",
                "",
            ),
        })

    return rows


def generator_registry_diagnostics() -> Dict[
    str,
    Any,
]:
    """
    Return complete registry diagnostics.
    """
    return {
        "registered_count": (
            generator_registry_size()
        ),
        "registered_generators": (
            list_registered_generators()
        ),
        "generator_info": (
            get_generator_registry_info()
        ),
        "builtin_modules": list(
            BUILTIN_GENERATOR_MODULES
        ),
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    methods = load_builtin_generators(
        strict=True
    )

    print("=" * 72)
    print("AWSRE WATERMARK GENERATOR REGISTRY")
    print("=" * 72)

    print(
        "\nRegistered generators:",
        generator_registry_size(),
    )

    for item in get_generator_registry_info():
        print(
            f"- {item['watermark_type']}: "
            f"{item['class_name']}"
        )

    expected = {
        WatermarkType.BINARY_PATTERN.value,
        WatermarkType.TEXT.value,
        WatermarkType.QR_CODE.value,
        WatermarkType.LOGO.value,
        WatermarkType.SIGNATURE.value,
    }

    if set(methods) != expected:
        raise RuntimeError(
            "Generator registry self test failed."
        )

    print(
        "\n✅ Generator registry self test passed."
    )
