"""Shared detached-model schemas and HDF5 fixtures."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import h5py
import numpy as np
import pytest

import h5t


class Measurement(h5t.Dataset, extras="ignore"):
    """Dataset carrying typed attrs."""

    unit: Literal["m"]
    scale: float = 1.0


@dataclasses.dataclass
class PlainMeasurement:
    """Foreign record over the same dataset as ``Measurement``, eager payload.

    ``data`` is reserved on ``h5t.Dataset`` but legal here -- a plain class has
    nothing in ``dir()`` for h5t's reserved-name check to forbid.
    """

    unit: Literal["m"]
    data: np.ndarray
    scale: float = 1.0


@dataclasses.dataclass
class LazyMeasurement:
    """Foreign record over the same dataset as ``Measurement``, lazily read payload."""

    unit: Literal["m"]
    data: h5t.LazyArray
    scale: float = 1.0


class Nested(h5t.Group, extras="forbid"):
    """Nested group fixture schema."""

    answer: int


class Result(h5t.Group, extras="ignore"):
    """Representative schema containing every supported field kind."""

    version: int
    renamed: Annotated[str, h5t.Name("stored-name")]
    config: Annotated[dict[str, Any], h5t.Attr(converter=json.loads)]
    array_attr: Annotated[np.ndarray, h5t.Attr()]
    values: np.ndarray
    measurement: Measurement
    eager_measurement: Annotated[Measurement, h5t.Eager()]
    nested: Nested
    optional_note: str | None
    defaulted: int = "42"  # type: ignore[assignment]


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
        for name in ("measurement", "eager_measurement"):
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
