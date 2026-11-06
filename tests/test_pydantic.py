"""Pydantic coercion, constraints, validators, and errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import h5py
import pytest
from pydantic import BaseModel, BeforeValidator, Field

import h5t


def test_coercion_literals_nested_containers_and_constraints(tmp_path: Path) -> None:
    class SchemaAttrs(BaseModel):
        count: int
        mode: Literal["fast", "safe"]
        constrained: Annotated[int, Field(gt=0)]
        payload: Annotated[dict[str, list[int]], BeforeValidator(json.loads)]

    class Schema(h5t.Group[SchemaAttrs]):
        pass

    path = tmp_path / "pydantic.h5"
    with h5py.File(path, "w") as file:
        file.attrs["count"] = "3"
        file.attrs["mode"] = "fast"
        file.attrs["constrained"] = 4
        file.attrs["payload"] = '{"numbers": [1, "2"]}'
    value = Schema.model_validate_hdf5(path)
    assert value.attrs.count == 3
    assert value.attrs.payload == {"numbers": [1, 2]}


def test_before_validator_runs_before_validation_and_surfaces_path(tmp_path: Path) -> None:
    calls: list[Any] = []

    def convert(value: Any) -> str:
        calls.append(value)
        raise ValueError("broken decoder")

    class SchemaAttrs(BaseModel):
        payload: Annotated[dict[str, int], BeforeValidator(convert)]

    class Schema(h5t.Group[SchemaAttrs]):
        pass

    path = tmp_path / "conversion.h5"
    with h5py.File(path, "w") as file:
        file.attrs["payload"] = "{}"
    with pytest.raises(h5t.ValidationError, match="broken decoder") as caught:
        Schema.model_validate_hdf5(path)
    assert calls == ["{}"]
    assert caught.value.path == "/@payload"


def test_pydantic_failure_is_concise_and_path_aware(tmp_path: Path) -> None:
    class SchemaAttrs(BaseModel):
        payload: Annotated[dict[str, list[int]], BeforeValidator(json.loads)]

    class Schema(h5t.Group[SchemaAttrs]):
        pass

    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as file:
        file.attrs["payload"] = '{"values": ["not-an-int"]}'
    with pytest.raises(h5t.ValidationError) as caught:
        Schema.model_validate_hdf5(path)
    assert caught.value.path == "/@payload"
    assert "values.0" in caught.value.message
    assert "validation error" not in caught.value.message.lower()
