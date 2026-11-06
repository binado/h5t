"""Lazy annotations under PEP 649/749.

Deliberately no ``from __future__ import annotations``: this module exists to exercise
the annotation forms every other test module opts out of. Schema types are declared
inside the test functions because a module-level bare forward reference would raise at
collection time on the versions ``pytestmark`` skips.
"""

import inspect
import sys
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pytest
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, PydanticUndefinedAnnotation

import h5t

from .conftest import write_result

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="PEP 649 lazy annotations require Python 3.14"
)


def test_this_module_keeps_lazy_annotations() -> None:
    class Probe:
        value: int

    assert inspect.get_annotations(Probe)["value"] is int


def test_a_lazy_annotation_may_name_a_class_defined_later(tmp_path: Path) -> None:
    class Deferred(h5t.Group):
        later: _DefinedLater  # noqa: F821

    class _DefinedLater(h5t.Group):
        value: np.ndarray

    Deferred.model_rebuild(_types_namespace={"_DefinedLater": _DefinedLater})
    value = Deferred(later=_DefinedLater(value=np.zeros(2)))
    path = tmp_path / "forward.h5"
    value.model_dump_hdf5(path)
    assert Deferred.model_validate_hdf5(path).later.value.shape == (2,)


def test_unresolved_lazy_forward_reference_raises_pydantic_error() -> None:
    class Deferred(h5t.Group):
        undefined: _NeverDefined  # noqa: F821

    with pytest.raises(PydanticUndefinedAnnotation, match="_NeverDefined"):
        Deferred.model_validate_hdf5("never-opened.h5")


def test_a_broken_lazy_annotation_helper_is_reported_on_first_load() -> None:
    def alias():
        return _typo_inside_the_helper  # noqa: F821

    class Broken(h5t.Group):
        value: alias()

    # Pydantic cannot tell a callee's NameError from a forward reference, so the group
    # defers and the missing name is reported on first use.
    with pytest.raises(PydanticUndefinedAnnotation, match="_typo_inside_the_helper"):
        Broken.model_validate_hdf5("never-opened.h5")


def test_lazy_annotation_helper_error_propagates() -> None:
    def alias():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):

        class Broken(h5t.Group):
            value: alias()


def test_a_group_whose_base_carries_the_unresolved_annotation_defers_too(tmp_path: Path) -> None:
    class Base:
        later: _DefinedLater  # noqa: F821

    class Sub(Base, h5t.Group):
        also: np.ndarray

    class _DefinedLater(h5t.Group):
        value: np.ndarray

    Sub.model_rebuild(_types_namespace={"_DefinedLater": _DefinedLater})
    path = tmp_path / "inherited.h5"
    Sub(later=_DefinedLater(value=np.zeros(2)), also=np.zeros(1)).model_dump_hdf5(path)
    assert Sub.model_validate_hdf5(path).later.value.shape == (2,)


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

    loaded = Lazy.model_validate_hdf5(path)
    assert loaded.attrs.version == 2
    assert np.array_equal(loaded.values, np.arange(4))
    assert loaded.measurement.attrs.unit == "m"
    assert loaded.measurement.attrs.scale == 1.0
    assert np.array_equal(loaded.measurement.read(), np.arange(5))
    assert loaded.nested.attrs.answer == 42
    assert loaded.attrs.optional_note is None


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

    class LaterAttrs(BaseModel):
        unit: Literal["m"]

    path = tmp_path / "later.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("measurement", data=np.arange(3)).attrs["unit"] = "m"
    loaded = Owner.model_validate_hdf5(path)
    assert isinstance(loaded.measurement.attrs, LaterAttrs)
