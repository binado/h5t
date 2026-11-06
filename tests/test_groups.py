"""The ``h5t.Group[T]`` and ``h5t.Dataset[T]`` generic API."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

import h5t
from h5t._spec import MemberKind

from .conftest import MeasurementAttrs, Nested, Result


class _ForwardBAttrs(BaseModel):
    value: int


class _ForwardA(h5t.Group):
    b: _ForwardB


class _ForwardB(h5t.Group[_ForwardBAttrs]):
    pass


class _NodeAttrs(BaseModel):
    value: int


class _RecursiveNode(h5t.Group[_NodeAttrs]):
    child: _RecursiveNode | None = None


def test_group_validates_eagerly_at_definition() -> None:
    with pytest.raises(h5t.SchemaError, match=r"Bad\.missing:"):

        class Bad(h5t.Group):
            missing: int


def test_attrs_cache_reuses_live_specs_without_retaining_models() -> None:
    class RecordingAttrs(BaseModel):
        unit: str

    class First(h5t.Group):
        measurement: h5t.Dataset[RecordingAttrs]

    class Second(h5t.Group):
        measurement: h5t.Dataset[RecordingAttrs]

    def measurement_field(group_cls: type[h5t.Group[Any]]) -> Any:
        spec = group_cls.__dict__["__h5t_group__"]
        return {field.py_name: field for field in spec.fields}["measurement"]

    assert measurement_field(First).dataset_attrs is measurement_field(Second).dataset_attrs

    def make_transient() -> weakref.ReferenceType[type]:
        class TransientAttrs(BaseModel):
            unit: str

        class Transient(h5t.Group):
            measurement: h5t.Dataset[TransientAttrs]

        return weakref.ref(TransientAttrs)

    transient = make_transient()
    gc.collect()
    assert transient() is None


def test_group_defaults_are_validated_only_with_validate_default(result_file: Path) -> None:
    class ChildAttrs(BaseModel):
        answer: int

    class Child(h5t.Group[ChildAttrs]):
        pass

    class WithInvalidDefault(h5t.Group):
        child: Child = Field(default=123, validate_default=True)  # type: ignore[assignment]

    class WithInvalidFactory(h5t.Group):
        child: Child = Field(default_factory=lambda: 123, validate_default=True)  # type: ignore[arg-type]

    for schema in (WithInvalidDefault, WithInvalidFactory):
        with pytest.raises(h5t.ValidationError) as caught:
            schema.model_validate_hdf5(result_file)
        assert caught.value.path == "/child"


def test_dataset_generic_loads_typed_attrs_and_slices(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: h5t.Dataset[MeasurementAttrs]

    result = Owner.model_validate_hdf5(result_file)
    assert isinstance(result.measurement, h5t.Dataset)
    assert result.measurement.attrs.unit == "m"
    assert result.measurement.attrs.scale == 1.0
    assert np.array_equal(result.measurement[()], np.arange(5))


def test_untyped_dataset_and_group_expose_raw_attrs_mapping(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: h5t.Dataset

    result = Owner.model_validate_hdf5(result_file)
    assert result.attrs["version"] == 2
    assert result.attrs["undeclared"] == "raw"
    assert result.measurement.attrs["unit"] == "m"
    assert result.measurement.attrs["extra"] == "measurement"


def test_forward_reference_to_a_group_defined_later_in_the_module_resolves() -> None:
    assert isinstance(_ForwardA.__dict__["__h5t_group__"], h5t._compile._PendingGroup)
    spec = h5t._compile._compiled_group(_ForwardA)
    fields = {field.py_name: field for field in spec.fields}
    assert fields["b"].member_type is _ForwardB


def test_recursive_group_compiles_and_loads_a_nested_file(tmp_path: Path) -> None:
    path = tmp_path / "recursive.h5"
    with h5py.File(path, "w") as file:
        file.attrs["value"] = 1
        child = file.create_group("child")
        child.attrs["value"] = 2

    result = _RecursiveNode.model_validate_hdf5(path)
    assert result.attrs.value == 1
    assert result.child is not None
    assert result.child.attrs.value == 2
    assert result.child.child is None


def test_group_nested_inside_another_group(result_file: Path) -> None:
    class InnerAttrs(BaseModel):
        answer: int

    class Inner(h5t.Group[InnerAttrs]):
        pass

    class Outer(h5t.Group):
        nested: Inner

    result = Outer.model_validate_hdf5(result_file)
    assert isinstance(result.nested, Inner)
    assert result.nested.attrs.answer == 42


def test_a_group_from_another_module_is_usable_as_a_field(result_file: Path) -> None:
    class Owner(h5t.Group):
        nested: Nested

    result = Owner.model_validate_hdf5(result_file)
    assert isinstance(result.nested, Nested)
    assert result.nested.attrs.answer == 42


def test_still_unresolved_nested_group_is_a_schema_error(tmp_path: Path) -> None:
    class Inner(h5t.Group):
        later: _NeverDefined  # noqa: F821

    assert isinstance(Inner.__dict__["__h5t_group__"], h5t._compile._PendingGroup)

    class Outer(h5t.Group):
        nested: Inner

    path = tmp_path / "nested.h5"
    with h5py.File(path, "w") as file:
        file.create_group("nested").attrs["later"] = 1

    for load in (
        lambda: Outer.model_validate_hdf5(path),
        lambda: Inner.model_validate_hdf5(path),
    ):
        with pytest.raises(h5t.SchemaError, match="_NeverDefined"):
            load()


def test_extras_forbid_on_a_group_rejects_undeclared_members(result_file: Path) -> None:
    class StrictAttrs(BaseModel):
        version: int

    class Strict(h5t.Group[StrictAttrs]):
        model_config = ConfigDict(extra="forbid")

    with pytest.raises(h5t.ValidationError):
        Strict.model_validate_hdf5(result_file)


def test_the_shared_result_schema_loads_every_field_kind(result_file: Path) -> None:
    actual = Result.model_validate_hdf5(result_file)

    assert actual.attrs.version == 2
    assert actual.attrs.renamed == "current"
    assert actual.attrs.config == {"enabled": True}
    assert np.array_equal(actual.attrs.array_attr, np.arange(3))
    assert np.array_equal(actual.values, np.arange(4))
    assert actual.measurement.attrs.unit == "m"
    assert actual.measurement.attrs.scale == 1.0
    assert np.array_equal(actual.measurement[()], np.arange(5))
    assert actual.raw_measurement.attrs["unit"] == "m"
    assert np.array_equal(actual.raw_measurement[()], np.arange(5))
    assert isinstance(actual.nested, Nested)
    assert actual.nested.attrs.answer == 42
    assert actual.attrs.optional_note is None
    assert actual.attrs.defaulted == 42


def test_dataset_is_a_member_in_its_own_right(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: h5t.Dataset
        absent: h5t.Dataset | None = None

    spec = Owner.__dict__["__h5t_group__"]
    field_kinds = {field.py_name: field.kind for field in spec.fields}
    assert field_kinds["measurement"] is MemberKind.DATASET
    assert field_kinds["absent"] is MemberKind.DATASET

    loaded = Owner.model_validate_hdf5(result_file)
    assert loaded.absent is None
    assert loaded.measurement.path == "/measurement"
    assert loaded.measurement.shape == (5,)
    assert loaded.measurement.dtype == np.dtype("int64")

    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10
    assert np.array_equal(loaded.measurement.data, np.arange(5) + 10)


def test_dataset_member_preserves_a_custom_subclass(result_file: Path) -> None:
    class Tracked(h5t.Dataset):
        pass

    class Owner(h5t.Group):
        measurement: Tracked
        absent: Tracked | None = None

    loaded = Owner.model_validate_hdf5(result_file)
    assert type(loaded.measurement) is Tracked
    assert loaded.absent is None
    assert np.array_equal(loaded.measurement.read(), np.arange(5))
