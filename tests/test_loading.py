"""Recursive loading, snapshots, defaults, and fail-fast errors."""

from __future__ import annotations

from pathlib import Path
from typing import Generic

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypeVar

import h5t

from .conftest import MeasurementAttrs, Nested, Result, open_fd_count, write_result


def test_model_validate_hdf5_loads_typed_attrs_and_children(result_file: Path) -> None:
    result = Result.model_validate_hdf5(result_file)
    assert isinstance(result, Result)
    assert result.attrs.version == 2
    assert result.attrs.renamed == "current"
    assert result.attrs.config == {"enabled": True}
    assert np.array_equal(result.attrs.array_attr, np.arange(3))
    assert np.array_equal(result.values, np.arange(4))
    assert isinstance(result.measurement, h5t.Dataset)
    assert isinstance(result.measurement.attrs, MeasurementAttrs)
    assert isinstance(result.nested, Nested)
    assert result.nested.attrs.answer == 42
    assert result.attrs.optional_note is None
    assert result.attrs.defaulted == 42
    assert "version=2" in repr(result)


def test_custom_root_and_absolute_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "nested.h5"
    write_result(path, root="/results/run")
    monkeypatch.chdir(tmp_path)
    result = Result.model_validate_hdf5("nested.h5", root="/results/run")
    assert result.measurement.path == "/results/run/measurement"
    assert result.measurement.filename == str(path.resolve())


@pytest.mark.parametrize("root", ["", "relative", "results/run"])
def test_root_must_be_absolute(result_file: Path, root: str) -> None:
    with pytest.raises(ValueError, match="absolute"):
        Result.model_validate_hdf5(result_file, root=root)


def test_path_must_be_a_filesystem_path(result_file: Path) -> None:
    with h5py.File(result_file, "r") as file:
        with pytest.raises(TypeError, match="filesystem path"):
            Result.model_validate_hdf5(file)  # type: ignore[arg-type]


def test_missing_root_group_and_wrong_root_kind_fail_with_paths(tmp_path: Path) -> None:
    class Empty(h5t.Group):
        pass

    path = tmp_path / "roots.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("leaf", data=1)
    with pytest.raises(h5t.ValidationError, match="root group does not exist") as caught:
        Empty.model_validate_hdf5(path, root="/absent")
    assert caught.value.path == "/absent"

    with pytest.raises(h5t.ValidationError, match="expected a group") as caught:
        Empty.model_validate_hdf5(path, root="/leaf")
    assert caught.value.path == "/leaf"


def test_missing_wrong_kind_and_forbidden_extra_fail_with_paths(tmp_path: Path) -> None:
    class Strict(h5t.Group):
        model_config = ConfigDict(extra="forbid")

        value: np.ndarray

    path = tmp_path / "errors.h5"
    with h5py.File(path, "w") as file:
        file.attrs["extra"] = 1
    with pytest.raises(h5t.ValidationError) as caught:
        Strict.model_validate_hdf5(path)
    assert caught.value.path == "/@extra"

    with h5py.File(path, "w") as file:
        file.create_group("value")
    with pytest.raises(
        h5t.ValidationError, match="Input should be an instance of ndarray"
    ) as caught:
        Strict.model_validate_hdf5(path)
    assert caught.value.path == "/value"

    with h5py.File(path, "w"):
        pass
    with pytest.raises(h5t.ValidationError, match="Field required") as caught:
        Strict.model_validate_hdf5(path)
    assert caught.value.path == "/value"


def test_group_and_dataset_kind_mismatches_fail_with_paths(tmp_path: Path) -> None:
    class ChildAttrs(BaseModel):
        value: int

    class Child(h5t.Group[ChildAttrs]):
        pass

    class TypedAttrs(BaseModel):
        unit: str

    class Owner(h5t.Group):
        child: Child
        typed: h5t.Dataset[TypedAttrs]

    path = tmp_path / "kind-mismatch.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("child", data=1)  # a group field finds a dataset
    with pytest.raises(h5t.ValidationError, match="Input should be a valid dictionary") as caught:
        Owner.model_validate_hdf5(path)
    assert caught.value.path == "/child"

    with h5py.File(path, "w") as file:
        group = file.create_group("child")
        group.attrs["value"] = 1
        file.create_group("typed")  # a dataset field finds a group
    with pytest.raises(h5t.ValidationError, match="expected a dataset") as caught:
        Owner.model_validate_hdf5(path)
    assert caught.value.path == "/typed"


def test_nested_record_controls_its_own_extras(tmp_path: Path) -> None:
    path = tmp_path / "nested-extra.h5"
    write_result(path)
    with h5py.File(path, "a") as file:
        file["nested"].attrs["surprise"] = 1
    with pytest.raises(h5t.ValidationError) as caught:
        Result.model_validate_hdf5(path)
    assert caught.value.path == "/nested@surprise"


class _InnerAttrs(BaseModel):
    n: int


class _Inner(h5t.Group[_InnerAttrs]):
    values: np.ndarray


class _Outer(h5t.Group):
    inner: _Inner
    runs: dict[str, _Inner]


def _write_outer(path: Path, *, inner_values: bool = True, run_values: bool = True) -> Path:
    with h5py.File(path, "w") as file:
        for group in (file.create_group("inner"), file.create_group("runs/r1")):
            group.attrs["n"] = 1
        if inner_values:
            file["inner/values"] = np.arange(3)
        if run_values:
            file["runs/r1/values"] = np.arange(2)
    return path


def test_load_nested_group_and_collection_item_read_plain_arrays(tmp_path: Path) -> None:
    loaded = _Outer.model_validate_hdf5(_write_outer(tmp_path / "o.h5"))
    np.testing.assert_array_equal(loaded.inner.values, [0, 1, 2])
    np.testing.assert_array_equal(loaded.runs["r1"].values, [0, 1])


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"inner_values": False}, "/inner/values"),
        ({"run_values": False}, "/runs/r1/values"),
    ],
)
def test_load_missing_nested_member_reports_its_hdf5_path(
    tmp_path: Path, kwargs: dict[str, bool], path: str
) -> None:
    file = _write_outer(tmp_path / "o.h5", **kwargs)
    with pytest.raises(h5t.ValidationError, match="Field required") as caught:
        _Outer.model_validate_hdf5(file)
    assert caught.value.path == path


def test_load_failure_mid_pass_leaks_no_file_descriptors(tmp_path: Path) -> None:
    file = _write_outer(tmp_path / "o.h5", run_values=False)
    before = open_fd_count()
    for _ in range(5):
        with pytest.raises(h5t.ValidationError):
            _Outer.model_validate_hdf5(file)
    assert open_fd_count() == before


_GenericAttrs = TypeVar("_GenericAttrs", bound=BaseModel)


class _GenericBox(h5t.Group[_GenericAttrs], Generic[_GenericAttrs]):
    values: np.ndarray


def test_load_parametrized_generic_group_child_reads_plain_arrays(tmp_path: Path) -> None:
    class Holder(h5t.Group):
        box: _GenericBox[_InnerAttrs]

    path = tmp_path / "h.h5"
    with h5py.File(path, "w") as file:
        file.create_group("box").attrs["n"] = 4
        file["box/values"] = np.ones(2)
    loaded = Holder.model_validate_hdf5(path)
    assert loaded.box.values.sum() == 2
    assert loaded.box.attrs.n == 4
