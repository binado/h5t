"""Dataset[T] and Group[T] attribute defaults, validators, and specialization edge cases."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest
from jaxtyping import Float
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

import h5t

from .conftest import MeasurementAttrs


class _WithDefaultsAttrs(BaseModel):
    unit: str
    note: str | None = None
    revision: int = Field(default_factory=lambda: 9)


class _CustomDataset(h5t.Dataset):
    pass


def test_defaults_and_optional_attrs_on_dataset(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: h5t.Dataset[_WithDefaultsAttrs]

    result = Owner.model_validate_hdf5(result_file)
    assert result.measurement.attrs.unit == "m"
    assert result.measurement.attrs.note is None
    assert result.measurement.attrs.revision == 9


def test_custom_dataset_subclass_with_specialization(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: _CustomDataset[MeasurementAttrs]

    measurement = Owner.model_validate_hdf5(result_file).measurement
    assert isinstance(measurement, _CustomDataset)
    assert measurement.attrs.unit == "m"


def test_extras_forbid_on_dataset_attrs_rejects_undeclared_attribute(result_file: Path) -> None:
    class StrictAttrs(BaseModel):
        model_config = ConfigDict(extra="forbid")

        unit: str
        scale: float = 1.0

    class Owner(h5t.Group):
        measurement: h5t.Dataset[StrictAttrs]

    with pytest.raises(h5t.ValidationError) as caught:
        Owner.model_validate_hdf5(result_file)
    assert caught.value.path == "/measurement@extra"


def test_model_validator_invariant_surfaces_as_validation_error_with_path(
    result_file: Path,
) -> None:
    class StrictAttrs(BaseModel):
        unit: str

        @model_validator(mode="after")
        def _check_unit(self) -> StrictAttrs:
            if self.unit != "s":
                raise ValueError(f"expected seconds, got {self.unit!r}")
            return self

    class Owner(h5t.Group):
        measurement: h5t.Dataset[StrictAttrs]

    with pytest.raises(h5t.ValidationError, match="expected seconds") as caught:
        Owner.model_validate_hdf5(result_file)
    assert caught.value.path == "/measurement"


def test_invalid_dataset_type_argument_is_schema_error() -> None:
    with pytest.raises(h5t.SchemaError, match="BaseModel subclass"):
        _ = h5t.Dataset[int]  # type: ignore[type-var]


def test_invalid_group_attrs_type_argument_is_schema_error() -> None:
    class OtherGroup(h5t.Group):
        pass

    with pytest.raises(h5t.SchemaError, match="cannot be a Group"):

        class BadGroup(h5t.Group[OtherGroup]):
            pass


def test_in_memory_group_construction_validates_dataset_attrs() -> None:
    class Owner(h5t.Group):
        measurement: h5t.Dataset[MeasurementAttrs]

    owner = Owner(measurement=h5t.Dataset(np.arange(3), {"unit": "m"}))
    assert isinstance(owner.measurement.attrs, MeasurementAttrs)
    assert owner.measurement.attrs.unit == "m"


class _GroupAttrs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(alias="stored-name")
    count: int


class _AttrGroup(h5t.Group[_GroupAttrs]):
    pass


@pytest.fixture
def attr_group_file(tmp_path: Path) -> Path:
    path = tmp_path / "group_attrs.h5"
    with h5py.File(path, "w") as file:
        file.attrs["stored-name"] = "ada"
        file.attrs["count"] = 2
    return path


def test_group_attrs_alias_is_the_hdf5_attribute_name(attr_group_file: Path) -> None:
    loaded = _AttrGroup.model_validate_hdf5(attr_group_file)
    assert loaded.attrs.name == "ada"


def test_group_attrs_extra_forbid_rejects_undeclared_attribute(attr_group_file: Path) -> None:
    with h5py.File(attr_group_file, "a") as file:
        file.attrs["surprise"] = 1
    with pytest.raises(h5t.ValidationError) as caught:
        _AttrGroup.model_validate_hdf5(attr_group_file)
    assert caught.value.path == "/@surprise"


def test_group_attrs_missing_required_field_reports_attribute_path(attr_group_file: Path) -> None:
    with h5py.File(attr_group_file, "a") as file:
        del file.attrs["count"]
    with pytest.raises(h5t.ValidationError, match="required attribute") as caught:
        _AttrGroup.model_validate_hdf5(attr_group_file)
    assert caught.value.path == "/@count"


class _CountedAttrs(BaseModel):
    n: Annotated[int, AfterValidator(lambda v: v + 1)]


class _CountedGroup(h5t.Group[_CountedAttrs]):
    pass


def test_dump_attrs_with_non_idempotent_validator_does_not_revalidate(tmp_path: Path) -> None:
    target = tmp_path / "counted.h5"
    group = _CountedGroup(attrs=_CountedAttrs(n=1))
    assert group.attrs.n == 2

    group.model_dump_hdf5(target)

    with h5py.File(target) as f:
        assert f.attrs["n"] == 2


class _ArrayAttrs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: Float[np.ndarray, "n"]
    bias: Float[np.ndarray, "n"] | None = None


class _ArrayAttrsGroup(h5t.Group[_ArrayAttrs]):
    pass


def test_load_array_attr_with_wrong_shape_reports_attribute_path(tmp_path: Path) -> None:
    path = tmp_path / "arrays.h5"
    with h5py.File(path, "w") as file:
        file.attrs["weights"] = np.zeros((2, 2))
    with pytest.raises(h5t.ValidationError) as caught:
        _ArrayAttrsGroup.model_validate_hdf5(path)
    assert caught.value.path == "/@weights"


def test_load_missing_optional_array_attr_is_none(tmp_path: Path) -> None:
    path = tmp_path / "arrays.h5"
    with h5py.File(path, "w") as file:
        file.attrs["weights"] = np.zeros(3)
    loaded = _ArrayAttrsGroup.model_validate_hdf5(path)
    assert loaded.attrs.bias is None


def test_attrs_model_with_extra_allow_is_schema_error() -> None:
    class OpenAttrs(BaseModel):
        model_config = ConfigDict(extra="allow")

        declared: int

    with pytest.raises(h5t.SchemaError, match="ignore.*forbid"):
        h5t.Group[OpenAttrs]
