"""Shared Group/Dataset schemas and HDF5 fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

import h5t


class MeasurementAttrs(BaseModel):
    """Dataset attributes for a measurement."""

    unit: Literal["m"]
    scale: float = 1.0


class NestedAttrs(BaseModel):
    """Attributes for the nested group fixture."""

    model_config = ConfigDict(extra="forbid")

    answer: int


class Nested(h5t.Group[NestedAttrs]):
    """Nested group fixture schema."""

    model_config = ConfigDict(extra="forbid")


class ResultAttrs(BaseModel):
    """Group attributes for the representative ``Result`` schema."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore", populate_by_name=True)

    version: int
    renamed: str = Field(alias="stored-name")
    config: Annotated[dict[str, Any], BeforeValidator(json.loads)]
    array_attr: np.ndarray
    optional_note: str | None = None
    defaulted: int = "42"  # type: ignore[assignment]


class Result(h5t.Group[ResultAttrs]):
    """Representative schema containing every supported field kind."""

    model_config = ConfigDict(extra="ignore")

    values: np.ndarray
    measurement: h5t.Dataset[MeasurementAttrs]
    raw_measurement: h5t.Dataset
    nested: Nested


def write_result(path: Path, *, root: str = "/") -> None:
    """Write a file conforming to ``Result`` at ``root``."""
    with h5py.File(path, "w") as file:
        group = file.require_group(root)
        group.attrs["version"] = np.int64(2)
        group.attrs["stored-name"] = "current"
        group.attrs["config"] = '{"enabled": true}'
        group.attrs["array_attr"] = np.arange(3)
        group.attrs["undeclared"] = "raw"
        values = group.create_dataset("values", data=np.arange(4))
        values.attrs["ignored"] = "plain arrays do not inspect this"
        for name in ("measurement", "raw_measurement"):
            dataset = group.create_dataset(name, data=np.arange(5))
            dataset.attrs["unit"] = "m"
            dataset.attrs["extra"] = name
        group.create_group("nested").attrs["answer"] = 42
        group.create_dataset("ignored-child", data=[1])


@pytest.fixture
def result_file(tmp_path: Path) -> Path:
    """A conforming result file."""
    path = tmp_path / "result.h5"
    write_result(path)
    return path


def open_fd_count() -> int:
    """Return the process's open descriptor count on POSIX."""
    return len(os.listdir("/dev/fd"))
