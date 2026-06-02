"""Regression tests for package-only import boundaries."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AZ_MODULE_PATH = PROJECT_ROOT / "deepretro" / "utils" / "az.py"


def test_az_module_uses_package_local_imports() -> None:
    """The package AiZynthFinder wrapper should not import from ``src``."""
    source = AZ_MODULE_PATH.read_text(encoding="utf-8")

    assert "from src.variables import BASIC_MOLECULES" not in source
    assert "from src.cache import cache_results" not in source
    assert "from deepretro.utils.variables import BASIC_MOLECULES" in source
    assert "from deepretro.utils.cache import cache_results" in source
