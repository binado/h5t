"""Pydantic coercion, constraints, validators, and errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import h5py
import pytest
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError

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


@pytest.fixture
def attrs_file(tmp_path: Path) -> Path:
    path = tmp_path / "attrs.h5"
    with h5py.File(path, "w") as file:
        file.attrs["name"] = "  Ada  "
        file.attrs["count"] = 3
    return path


def test_wrap_field_validator_runs_on_load(attrs_file: Path) -> None:
    class WrapAttrs(BaseModel):
        name: str
        count: int

        @field_validator("name", mode="wrap")
        @classmethod
        def strip(cls, value: Any, handler: Any) -> str:
            return handler(value).strip()

    class Schema(h5t.Group[WrapAttrs]):
        pass

    assert Schema.model_validate_hdf5(attrs_file).attrs.name == "Ada"


def test_before_field_validator_runs_on_load(attrs_file: Path) -> None:
    class BeforeAttrs(BaseModel):
        name: str
        count: int

        @field_validator("count", mode="before")
        @classmethod
        def double(cls, value: Any) -> Any:
            return int(value) * 2

    class Schema(h5t.Group[BeforeAttrs]):
        pass

    assert Schema.model_validate_hdf5(attrs_file).attrs.count == 6


def test_star_field_validator_applies_to_every_attribute(attrs_file: Path) -> None:
    seen: list[str] = []

    class StarAttrs(BaseModel):
        name: str
        count: int

        @field_validator("*")
        @classmethod
        def record(cls, value: Any) -> Any:
            seen.append(type(value).__name__)
            return value

    class Schema(h5t.Group[StarAttrs]):
        pass

    Schema.model_validate_hdf5(attrs_file)
    assert sorted(seen) == ["int", "str"]


def test_after_field_validator_receives_validation_info(attrs_file: Path) -> None:
    class InfoAttrs(BaseModel):
        name: str
        count: int

        @field_validator("count")
        @classmethod
        def check(cls, value: int, info: ValidationInfo) -> int:
            assert info.field_name == "count"
            assert info.data["name"] == "  Ada  "
            return value

    class Schema(h5t.Group[InfoAttrs]):
        pass

    assert Schema.model_validate_hdf5(attrs_file).attrs.count == 3


def test_frozen_attrs_model_loads_and_stays_frozen(attrs_file: Path) -> None:
    class FrozenAttrs(BaseModel):
        model_config = ConfigDict(frozen=True)

        name: str
        count: int

    class Schema(h5t.Group[FrozenAttrs]):
        pass

    attrs = Schema.model_validate_hdf5(attrs_file).attrs
    with pytest.raises(PydanticValidationError, match="frozen"):
        attrs.count = 4  # type: ignore[misc]


def test_loaded_attrs_equal_in_memory_model(attrs_file: Path) -> None:
    class EqAttrs(BaseModel):
        name: str
        count: int

    class Schema(h5t.Group[EqAttrs]):
        pass

    loaded = Schema.model_validate_hdf5(attrs_file)
    assert loaded == Schema(attrs=EqAttrs(name="  Ada  ", count=3))


def test_validator_bug_propagates_instead_of_becoming_validation_error(attrs_file: Path) -> None:
    class BuggyAttrs(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def explode(cls, value: str) -> str:
            raise KeyError("bug")

    class Schema(h5t.Group[BuggyAttrs]):
        pass

    with pytest.raises(KeyError):
        Schema.model_validate_hdf5(attrs_file)


def test_before_model_validator_on_group_runs_on_load(attrs_file: Path) -> None:
    class Schema(h5t.Group):
        @model_validator(mode="before")
        @classmethod
        def tag(cls, data: Any) -> Any:
            data["attrs"] = {**data["attrs"], "tagged": True}
            return data

    assert Schema.model_validate_hdf5(attrs_file).attrs["tagged"] is True
