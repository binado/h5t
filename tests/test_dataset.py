"""Detached dataset metadata, lazy slicing/caching, live opens, and lifetime hygiene."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

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


def test_scalar_dataset_payload_is_zero_dimensional_array(tmp_path: Path) -> None:
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
    with pytest.raises(OSError):
        measurement.read()


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
