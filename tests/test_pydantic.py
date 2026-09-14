"""Pydantic coercion, constraints, converters, and errors."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import h5py
import pytest
from pydantic import Field

import h5t


def test_coercion_literals_nested_containers_and_constraints(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Schema:
        count: int
        mode: Literal["fast", "safe"]
        constrained: Annotated[int, Field(gt=0)]
        payload: Annotated[dict[str, list[int]], h5t.Attr(converter=json.loads)]

    path = tmp_path / "pydantic.h5"
    with h5py.File(path, "w") as file:
        file.attrs["count"] = "3"
        file.attrs["mode"] = "fast"
        file.attrs["constrained"] = 4
        file.attrs["payload"] = '{"numbers": [1, "2"]}'
    value = h5t.load(Schema, path)
    assert value.count == 3
    assert value.payload == {"numbers": [1, 2]}


def test_converter_runs_before_validation_and_chains_failure(tmp_path: Path) -> None:
    calls: list[Any] = []

    def convert(value: Any) -> str:
        calls.append(value)
        raise RuntimeError("broken decoder")

    @h5t.group()
    @dataclasses.dataclass
    class Schema:
        payload: Annotated[dict[str, int], h5t.Attr(converter=convert)]

    path = tmp_path / "conversion.h5"
    with h5py.File(path, "w") as file:
        file.attrs["payload"] = "{}"
    with pytest.raises(h5t.ConversionError, match="broken decoder") as caught:
        h5t.load(Schema, path)
    assert calls == ["{}"]
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.path == "/@payload"


def test_pydantic_failure_is_concise_and_path_aware(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Schema:
        payload: Annotated[dict[str, list[int]], h5t.Attr(converter=json.loads)]

    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as file:
        file.attrs["payload"] = '{"values": ["not-an-int"]}'
    with pytest.raises(h5t.ValidationError) as caught:
        h5t.load(Schema, path)
    assert caught.value.path == "/@payload"
    assert "values.0" in caught.value.message
    assert "validation error" not in caught.value.message.lower()
