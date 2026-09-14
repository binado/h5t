"""Recursive loading, snapshots, defaults, and fail-fast errors."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import h5py
import numpy as np
import pytest

import h5t

from .conftest import Measurement, Result, write_result


def test_materializes_detached_typed_record(result_file: Path) -> None:
    result = Result.from_file(result_file)
    assert isinstance(result, Result)
    assert result.version == 2
    assert result.renamed == "current"
    assert result.config == {"enabled": True}
    assert np.array_equal(result.array_attr, np.arange(3))
    assert np.array_equal(result.values, np.arange(4))
    assert isinstance(result.measurement, Measurement)
    assert result.nested.answer == 42
    assert result.optional_note is None
    assert result.defaulted == 42
    assert isinstance(result.attrs, MappingProxyType)
    assert result.attrs["stored-name"] == "current"
    assert result.attrs["undeclared"] == "raw"
    assert result.attrs["optional_note"] is None
    assert result.attrs["defaulted"] == 42
    with pytest.raises(TypeError):
        result.attrs["new"] = 1  # type: ignore[index]
    assert "Result(version=2" in repr(result)


def test_custom_root_and_absolute_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "nested.h5"
    write_result(path, root="/results/run")
    monkeypatch.chdir(tmp_path)
    result = Result.from_file("nested.h5", root="/results/run")
    assert result.measurement.path == "/results/run/measurement"
    assert result.measurement._h5t_array.filename == str(path.resolve())


@pytest.mark.parametrize("root", ["", "relative", "results/run"])
def test_root_must_be_absolute(result_file: Path, root: str) -> None:
    with pytest.raises(ValueError, match="absolute"):
        Result.from_file(result_file, root=root)


def test_path_must_be_a_filesystem_path(result_file: Path) -> None:
    with h5py.File(result_file, "r") as file:
        with pytest.raises(TypeError, match="filesystem path"):
            Result.from_file(file)  # type: ignore[arg-type]


def test_missing_wrong_kind_and_forbidden_extra_fail_with_paths(tmp_path: Path) -> None:
    class Strict(h5t.Group, extras="forbid"):
        value: np.ndarray

    path = tmp_path / "errors.h5"
    with h5py.File(path, "w") as file:
        file.attrs["extra"] = 1
    with pytest.raises(h5t.ValidationError) as caught:
        Strict.from_file(path)
    assert caught.value.path == "/@extra"

    with h5py.File(path, "w") as file:
        file.create_group("value")
    with pytest.raises(h5t.ValidationError, match="expected a dataset") as caught:
        Strict.from_file(path)
    assert caught.value.path == "/value"

    with h5py.File(path, "w"):
        pass
    with pytest.raises(h5t.ValidationError, match="required array") as caught:
        Strict.from_file(path)
    assert caught.value.path == "/value"


def test_group_and_dataset_kind_mismatches_fail_with_paths(tmp_path: Path) -> None:
    class Child(h5t.Group):
        value: int

    class Typed(h5t.Dataset):
        unit: str

    class Owner(h5t.Group):
        child: Child
        typed: Typed

    path = tmp_path / "kind-mismatch.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("child", data=1)  # a group field finds a dataset
    with pytest.raises(h5t.ValidationError, match="expected a group") as caught:
        Owner.from_file(path)
    assert caught.value.path == "/child"

    with h5py.File(path, "w") as file:
        group = file.create_group("child")
        group.attrs["value"] = 1
        file.create_group("typed")  # a dataset field finds a group
    with pytest.raises(h5t.ValidationError, match="expected a dataset") as caught:
        Owner.from_file(path)
    assert caught.value.path == "/typed"


def test_nested_class_controls_its_own_extras(tmp_path: Path) -> None:
    path = tmp_path / "nested-extra.h5"
    write_result(path)
    with h5py.File(path, "a") as file:
        file["nested"].attrs["surprise"] = 1
    with pytest.raises(h5t.ValidationError) as caught:
        Result.from_file(path)
    assert caught.value.path == "/nested@surprise"


def test_direct_construction_and_generated_equality_are_unsupported(result_file: Path) -> None:
    with pytest.raises(TypeError):
        Result()
    with pytest.raises(TypeError):
        Measurement()
    first = Result.from_file(result_file)
    second = Result.from_file(result_file)
    assert first != second
