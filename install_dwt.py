"""
Install the AWSRE DWT module into an existing repository.

Run from the repository root:

    python install_dwt.py
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "watermarking" / "registry.py"
PACKAGE_INIT = ROOT / "watermarking" / "__init__.py"
DWT_MODULE = ROOT / "watermarking" / "dwt.py"
DWT_TEST = ROOT / "tests" / "test_dwt.py"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file is missing: {path}"
        )


def patch_registry() -> bool:
    text = REGISTRY.read_text(encoding="utf-8")

    if "watermarking.dwt" in text:
        print("✅ Registry already includes watermarking.dwt.")
        return False

    patterns = [
        (
            r'(["\']watermarking\.dct["\']\s*,?)',
            r'\1\n        "watermarking.dwt",',
        ),
        (
            r'(importlib\.import_module\(\s*["\']watermarking\.dct["\']\s*\))',
            r'\1\n    importlib.import_module("watermarking.dwt")',
        ),
    ]

    for pattern, replacement in patterns:
        updated, count = re.subn(
            pattern,
            replacement,
            text,
            count=1,
        )

        if count:
            compile(updated, str(REGISTRY), "exec")
            REGISTRY.write_text(
                updated,
                encoding="utf-8",
            )
            print("✅ Registry patched for DWT.")
            return True

    print(
        "ℹ️ Registry built-in list was not recognized. "
        "DWT is still registered whenever watermarking.dwt "
        "is imported."
    )
    return False


def patch_package_init() -> bool:
    text = PACKAGE_INIT.read_text(encoding="utf-8")

    if "DWTWatermarker" in text:
        print("✅ watermarking.__init__ already exports DWTWatermarker.")
        return False

    addition = (
        "\n\n# DWT algorithm export\n"
        "from watermarking.dwt import DWTWatermarker\n"
    )

    updated = text.rstrip() + addition
    compile(updated, str(PACKAGE_INIT), "exec")

    PACKAGE_INIT.write_text(
        updated,
        encoding="utf-8",
    )

    print("✅ watermarking.__init__ patched for DWT.")
    return True


def main() -> None:
    for path in (
        REGISTRY,
        PACKAGE_INIT,
        DWT_MODULE,
        DWT_TEST,
    ):
        require(path)

    compile(
        DWT_MODULE.read_text(encoding="utf-8"),
        str(DWT_MODULE),
        "exec",
    )

    compile(
        DWT_TEST.read_text(encoding="utf-8"),
        str(DWT_TEST),
        "exec",
    )

    patch_registry()
    patch_package_init()

    print("\n✅ AWSRE DWT installation completed.")
    print("Run:")
    print("  python -m watermarking.dwt")
    print("  python -m tests.test_dwt")
    print("  python -m tests.smoke_test")


if __name__ == "__main__":
    main()
