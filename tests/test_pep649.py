"""Lazy annotations under PEP 649/749.

Deliberately no ``from __future__ import annotations``: this module exists to exercise
the annotation forms every other test module opts out of. Record types are declared
inside the test functions because a module-level bare forward reference would raise at
collection time on the versions ``pytestmark`` skips.
"""

import gc
import inspect
import sys
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pytest
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict

import h5t
from h5t._compile import _PendingGroup
from h5t._spec import GroupSpec, MemberKind

from .conftest import write_result

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="PEP 649 lazy annotations require Python 3.14"
)


def marker(group_cls: type) -> Any:
    return group_cls.__dict__["__h5t_group__"]


def test_this_module_keeps_lazy_annotations() -> None:
    class Probe:
        value: int

    assert inspect.get_annotations(Probe)["value"] is int


def test_a_lazy_annotation_may_name_a_class_defined_later() -> None:
    class Deferred(h5t.Group):
        later: _DefinedLater  # noqa: F821

    assert isinstance(marker(Deferred), _PendingGroup)

    class _DefinedLater(h5t.Group):
        value: np.ndarray

    spec = h5t._compile._ensure_group_compiled(Deferred)
    fields = {field.py_name: field for field in spec.fields}
    assert fields["later"].kind is MemberKind.GROUP
    assert fields["later"].member_type is _DefinedLater


def test_a_lazy_forward_reference_still_unresolved_reports_a_schema_error() -> None:
    class Deferred(h5t.Group):
        undefined: _NeverDefined  # noqa: F821

    with pytest.raises(h5t.SchemaError, match="_NeverDefined"):
        Deferred.model_validate_hdf5("never-opened.h5")


def test_a_broken_lazy_annotation_helper_is_not_a_forward_reference() -> None:
    def alias():
        return _typo_inside_the_helper  # noqa: F821

    with pytest.raises(h5t.SchemaError, match="_typo_inside_the_helper"):

        class Broken(h5t.Group):
            value: alias()


def test_a_lazy_annotation_helper_that_raises_becomes_a_schema_error() -> None:
    def alias():
        raise ValueError("kaboom")

    with pytest.raises(h5t.SchemaError, match="kaboom"):

        class Broken(h5t.Group):
            value: alias()


def test_a_deferred_group_needs_localns_for_a_quoted_function_local_annotation() -> None:
    class Dependency(h5t.Group):
        value: np.ndarray

    def declare(**kwargs: Any) -> type[h5t.Group[Any]]:
        class Deferred(h5t.Group, **kwargs):
            dependency: "Dependency"

        return Deferred

    plain = declare()
    with_localns = declare(localns={"Dependency": Dependency})

    with pytest.raises(h5t.SchemaError, match="Dependency"):
        plain.model_validate_hdf5("never-opened.h5")
    fields = h5t._compile._compiled_group(with_localns).fields
    assert fields[0].member_type is Dependency


def test_a_group_whose_base_carries_the_unresolved_annotation_defers_too() -> None:
    class Base:
        later: _DefinedLater  # noqa: F821

    class Sub(Base, h5t.Group):
        also: np.ndarray

    assert isinstance(marker(Sub), _PendingGroup)

    class _DefinedLater(h5t.Group):
        value: np.ndarray

    spec = h5t._compile._ensure_group_compiled(Sub)
    assert [field.py_name for field in spec.fields] == ["later", "also"]


def test_the_group_defers_under_pep_649() -> None:
    def declare() -> type[h5t.Group[Any]]:
        class Local(h5t.Group):
            later: Later  # noqa: F821

        class Later(h5t.Group):
            value: np.ndarray

        return Local

    record = declare()
    gc.collect()
    assert marker(record).localns is None
    spec = h5t._compile._ensure_group_compiled(record)
    fields = {field.py_name: field for field in spec.fields}
    assert fields["later"].member_type.__name__ == "Later"


def test_a_lazy_schema_loads_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "result.h5"
    write_result(path)

    class LazyAttrs(BaseModel):
        version: int
        optional_note: str | None = None

    class Lazy(h5t.Group[LazyAttrs]):
        values: np.ndarray
        measurement: h5t.Dataset[LazyRecordingAttrs]  # noqa: F821
        nested: LazyNested  # noqa: F821

    class LazyRecordingAttrs(BaseModel):
        unit: Literal["m"]
        scale: float = 1.0

    class LazyNestedAttrs(BaseModel):
        model_config = ConfigDict(extra="forbid")

        answer: int

    class LazyNested(h5t.Group[LazyNestedAttrs]):
        model_config = ConfigDict(extra="forbid")

    assert isinstance(marker(Lazy), _PendingGroup)
    record = Lazy.model_validate_hdf5(path)
    assert isinstance(marker(Lazy), GroupSpec)
    assert record.attrs.version == 2
    assert np.array_equal(record.values, np.arange(4))
    assert record.measurement.attrs.unit == "m"
    assert record.measurement.attrs.scale == 1.0
    assert np.array_equal(record.measurement.read(), np.arange(5))
    assert record.nested.attrs.answer == 42
    assert record.attrs.optional_note is None


def test_bare_jaxtyping_annotation_inside_a_function(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Float[np.ndarray, "n"]
        y: Float[np.ndarray, "n"]

    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        f["x"] = np.zeros(3)
        f["y"] = np.zeros(4)
    with pytest.raises(h5t.ValidationError, match=r"/y: the size of dimension n is 4"):
        R.model_validate_hdf5(path)


def test_dataset_forward_reference_to_attrs_defined_later(tmp_path: Path) -> None:
    class Owner(h5t.Group):
        measurement: h5t.Dataset[LaterAttrs]  # noqa: F821

    assert isinstance(marker(Owner), _PendingGroup)

    class LaterAttrs(BaseModel):
        unit: Literal["m"]

    path = tmp_path / "later.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("measurement", data=np.arange(3)).attrs["unit"] = "m"
    loaded = Owner.model_validate_hdf5(path)
    assert isinstance(loaded.measurement.attrs, LaterAttrs)
