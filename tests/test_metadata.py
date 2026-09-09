"""Packaging metadata consistency."""

from __future__ import annotations

import tomllib
from pathlib import Path

import h5t


def test_pyproject_version_matches_package_version() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    assert data["project"]["version"] == h5t.__version__
