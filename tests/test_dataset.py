"""Detached dataset metadata, lazy slicing/caching, live opens, and lifetime hygiene."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

import h5t

from .conftest import MeasurementAttrs, Result, open_fd_count, write_result


def test_metadata_and_all_attributes_are_snapshots(result_file: Path) -> None:
    class SnapshotAttrs(BaseModel):
        unit: str
        scale: float = 1.0

    class Owner(h5t.Group):
        measurement: h5t.Dataset[SnapshotAttrs]
        raw_measurement: h5t.Dataset

    loaded = Owner.model_validate_hdf5(result_file)
    measurement = loaded.measurement
    assert measurement.path == "/measurement"
    assert measurement.shape == (5,)
    assert measurement.dtype == np.dtype("int64")
    assert measurement.ndim == 1
    assert measurement.size == 5
    assert measurement.nbytes == 40
    assert measurement.attrs.unit == "m"
    assert measurement.attrs.scale == 1.0
    assert loaded.raw_measurement.attrs["unit"] == "m"
    assert loaded.raw_measurement.attrs["extra"] == "raw_measurement"

    with h5py.File(result_file, "a") as file:
        del file["measurement"]
        changed = file.create_dataset("measurement", data=np.arange(10))
        changed.attrs["unit"] = "s"
    assert measurement.shape == (5,)
    assert measurement.attrs.unit == "m"


def test_forbidden_dataset_extra_attribute_fails_with_path(tmp_path: Path) -> None:
    class StrictAttrs(BaseModel):
        model_config = ConfigDict(extra="forbid")

        unit: str

    class Owner(h5t.Group):
        payload: h5t.Dataset[StrictAttrs]

    path = tmp_path / "strict-dataset.h5"
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("payload", data=np.arange(3))
        dataset.attrs["unit"] = "m"
        dataset.attrs["surprise"] = 1
    with pytest.raises(h5t.ValidationError) as caught:
        Owner.model_validate_hdf5(path)
    assert caught.value.path == "/payload@surprise"


def test_lazy_data_and_indexing_observe_file_and_cache(result_file: Path) -> None:
    measurement = Result.model_validate_hdf5(result_file).measurement
    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10
    # Partial slice reads from disk without caching the full array
    assert np.array_equal(measurement[:2], [10, 11])
    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 20
    # Full read via [()] or .data caches the array
    first = measurement[()]
    assert np.array_equal(first, np.arange(5) + 20)
    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 30
    assert measurement.data is first
    assert measurement.read() is first
    assert measurement[:] is first
    assert np.array_equal(measurement[:2], [20, 21])
    assert np.array_equal(np.asarray(measurement), first)


def test_scalar_dataset_data_is_zero_dimensional_array(tmp_path: Path) -> None:
    class Scalar(h5t.Group):
        plain: np.ndarray
        detached: h5t.Dataset

    path = tmp_path / "scalar.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("plain", data=7)
        file.create_dataset("detached", data=9)
    loaded = Scalar.model_validate_hdf5(path)
    assert isinstance(loaded.plain, np.ndarray) and loaded.plain.shape == ()
    assert isinstance(loaded.detached[()], np.ndarray) and loaded.detached[()].shape == ()


def test_open_supports_slices_repeated_opens_and_closes_on_exceptions(result_file: Path) -> None:
    measurement = Result.model_validate_hdf5(result_file).measurement
    with measurement.open() as live:
        assert np.array_equal(live[:2], [0, 1])
    assert not live.id.valid
    with measurement.open() as second:
        assert second.shape == (5,)
    with pytest.raises(RuntimeError):
        with measurement.open() as exceptional:
            raise RuntimeError("consumer failed")
    assert not exceptional.id.valid


def test_in_memory_dataset_slicing_and_equality() -> None:
    ds1 = h5t.Dataset(np.arange(5), MeasurementAttrs(unit="m"))
    ds2 = h5t.Dataset[MeasurementAttrs](np.arange(5), {"unit": "m"})
    assert ds1 == ds2
    assert np.array_equal(ds1[1:4], [1, 2, 3])
    with pytest.raises(h5t.ValidationError, match="not backed by an HDF5 file"):
        with ds1.open():
            pass


def test_deleted_source_preserves_snapshot_but_live_access_fails(result_file: Path) -> None:
    measurement = Result.model_validate_hdf5(result_file).measurement
    os.unlink(result_file)
    assert measurement.shape == (5,)
    assert measurement.attrs.unit == "m"
    with pytest.raises(h5t.ValidationError, match="cannot open the source file") as caught:
        measurement.read()
    assert caught.value.path == "/measurement"


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="requires POSIX descriptors")
def test_no_descriptors_leak_on_success_or_failure(tmp_path: Path) -> None:
    class Owner(h5t.Group):
        measurement: h5t.Dataset[MeasurementAttrs]
        values: h5t.Dataset

    good = tmp_path / "good.h5"
    write_result(good)
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w"):
        pass

    def exercise() -> None:
        Result.model_validate_hdf5(good)
        Owner.model_validate_hdf5(good).measurement.read()
        Owner.model_validate_hdf5(good).values[:2]
        with pytest.raises(h5t.ValidationError):
            Result.model_validate_hdf5(bad)
        with pytest.raises(h5t.ValidationError):
            Owner.model_validate_hdf5(bad)

    exercise()
    baseline = open_fd_count()
    for _ in range(20):
        exercise()
    assert open_fd_count() == baseline


_KEY_CASES = {
    "ellipsis": ...,
    "empty-tuple": (),
    "full-slice": slice(None),
    "negative-int": -1,
    "int-list": [0, 2, 4],
    "int-array": np.array([0, 2, 4]),
    "bool-mask": np.array([True, False, True, False, True]),
    "slice-tuple": (slice(1, 4),),
    "stepped-slice": slice(None, None, 2),
}


@pytest.fixture(params=["in-memory", "file-backed"])
def indexed_dataset(request: pytest.FixtureRequest, result_file: Path) -> h5t.Dataset:
    if request.param == "in-memory":
        return h5t.Dataset(np.arange(5), MeasurementAttrs(unit="m"))
    return Result.model_validate_hdf5(result_file).measurement


@pytest.mark.parametrize("key", list(_KEY_CASES.values()), ids=list(_KEY_CASES))
def test_dataset_getitem_numpy_keys_match_ndarray_indexing(
    indexed_dataset: h5t.Dataset, key: object
) -> None:
    np.testing.assert_array_equal(indexed_dataset[key], np.arange(5)[key])


@pytest.mark.parametrize(
    ("mutate", "match", "path"),
    [
        pytest.param("shape", "changed since it was loaded", None, id="shape-change"),
        pytest.param("dtype", "changed since it was loaded", None, id="dtype-change"),
        pytest.param("deleted", "no longer exists", "/measurement", id="deleted"),
        pytest.param("replaced-by-group", "found a group", "/measurement", id="replaced-by-group"),
    ],
)
def test_open_detects_source_changes_since_load(
    result_file: Path, mutate: str, match: str, path: str | None
) -> None:
    measurement = Result.model_validate_hdf5(result_file).measurement
    with h5py.File(result_file, "a") as file:
        del file["measurement"]
        if mutate == "shape":
            file.create_dataset("measurement", data=np.arange(7))
        elif mutate == "dtype":
            file.create_dataset("measurement", data=np.arange(5.0))
        elif mutate == "replaced-by-group":
            file.create_group("measurement")
    with pytest.raises(h5t.ValidationError, match=match) as caught:
        measurement.read()
    if path is not None:
        assert caught.value.path == path


@pytest.mark.parametrize("access", [lambda ds: ds.data, lambda ds: ds[()], lambda ds: ds[:]])
def test_full_reads_are_read_only(indexed_dataset: h5t.Dataset, access) -> None:
    array = access(indexed_dataset)
    assert not array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        array[0] = 99


def test_in_memory_dataset_does_not_freeze_the_callers_array() -> None:
    source = np.arange(3)
    dataset = h5t.Dataset(source)
    assert not dataset.data.flags.writeable
    assert source.flags.writeable


def test_equality_compares_metadata_before_reading_data(result_file: Path) -> None:
    measurement = Result.model_validate_hdf5(result_file).measurement
    os.unlink(result_file)  # reading the data would now fail
    assert measurement != h5t.Dataset(np.arange(6), MeasurementAttrs(unit="m"))
    assert measurement != h5t.Dataset(np.arange(5.0), MeasurementAttrs(unit="m"))
    assert measurement != h5t.Dataset(np.arange(5), MeasurementAttrs(unit="m", scale=2.0))


class _OtherAttrs(BaseModel):
    note: str = "n"


@pytest.fixture
def typed_adapter() -> TypeAdapter[h5t.Dataset[MeasurementAttrs]]:
    return TypeAdapter(h5t.Dataset[MeasurementAttrs])


@pytest.mark.parametrize(
    ("value", "match"),
    [
        pytest.param(np.arange(3), "instance of Dataset", id="non-dataset"),
        pytest.param(
            h5t.Dataset(np.arange(3), _OtherAttrs()), "MeasurementAttrs", id="wrong-attrs-model"
        ),
        pytest.param(
            h5t.Dataset(np.arange(3), {"unit": "parsec"}),
            "unit",
            id="invalid-attrs-mapping",
        ),
    ],
)
def test_dataset_schema_rejects_bad_values(
    typed_adapter: TypeAdapter[h5t.Dataset[MeasurementAttrs]], value: Any, match: str
) -> None:
    with pytest.raises(PydanticValidationError, match=match):
        typed_adapter.validate_python(value)


def test_dataset_schema_clone_of_lazy_dataset_stays_lazy(
    result_file: Path, typed_adapter: TypeAdapter[h5t.Dataset[MeasurementAttrs]]
) -> None:
    raw = Result.model_validate_hdf5(result_file).raw_measurement
    clone = typed_adapter.validate_python(raw)
    assert isinstance(clone, h5t.Dataset[MeasurementAttrs])
    assert (clone.filename, clone.path) == (raw.filename, raw.path)
    assert clone.attrs.unit == "m"
    with h5py.File(result_file, "a") as file:
        file["raw_measurement"][...] = np.arange(5) + 100
    np.testing.assert_array_equal(clone[:2], [100, 101])  # still reads from the file


def test_dataset_schema_clone_keeps_cached_data(
    result_file: Path, typed_adapter: TypeAdapter[h5t.Dataset[MeasurementAttrs]]
) -> None:
    raw = Result.model_validate_hdf5(result_file).raw_measurement
    cached = raw.data
    with h5py.File(result_file, "a") as file:
        file["raw_measurement"][...] = np.arange(5) + 100
    clone = typed_adapter.validate_python(raw)
    assert clone.data is cached


def test_dataset_subscript_with_mapping_returns_the_untyped_base() -> None:
    assert h5t.Dataset[Mapping[str, Any]] is h5t.Dataset


@pytest.mark.parametrize("bad", [int, 3, list[int]])
def test_dataset_subscript_with_invalid_argument_is_schema_error(bad: Any) -> None:
    with pytest.raises(h5t.SchemaError, match="type argument"):
        h5t.Dataset[bad]


@pytest.mark.parametrize("name", ["filename", "path", "shape", "dtype", "attrs"])
def test_dataset_metadata_is_read_only(name: str) -> None:
    dataset = h5t.Dataset(np.arange(3))
    with pytest.raises(AttributeError):
        setattr(dataset, name, None)


@pytest.mark.parametrize("annotation", [Mapping, dict, Mapping[str, Any], dict[str, Any]])
def test_dataset_accepts_untyped_mapping_attrs_argument(annotation: Any) -> None:
    assert h5t.Dataset[annotation] is h5t.Dataset


@pytest.mark.parametrize("annotation", [Mapping[str, int], dict[str, str]])
def test_dataset_rejects_typed_mapping_attrs_argument(annotation: Any) -> None:
    with pytest.raises(h5t.SchemaError, match="BaseModel"):
        h5t.Dataset[annotation]


def test_type_adapter_builds_lazy_dataset_from_open_h5py_node(tmp_path: Path) -> None:
    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as f:
        f["m"] = np.arange(6.0)
        f["m"].attrs["unit"] = "m"
        f["m"].attrs["scale"] = np.float64(2.0)
    adapter = TypeAdapter(h5t.Dataset[MeasurementAttrs])
    with h5py.File(path, "r") as f:
        ds = adapter.validate_python(f["m"])
    assert ds.path == "/m"
    assert ds.filename == str(path)
    assert ds.shape == (6,)
    assert isinstance(ds.attrs, MeasurementAttrs)
    assert ds[:2].tolist() == [0.0, 1.0]  # read lazily, after the file closed


def test_type_adapter_untyped_dataset_exposes_read_only_attrs(tmp_path: Path) -> None:
    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as f:
        f["m"] = np.zeros(2)
        f["m"].attrs["k"] = 1
    with h5py.File(path, "r") as f:
        ds = TypeAdapter(h5t.Dataset).validate_python(f["m"])
    assert dict(ds.attrs) == {"k": 1}
    with pytest.raises(TypeError):
        ds.attrs["k"] = 2  # type: ignore[index]
