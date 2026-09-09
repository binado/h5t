"""Detached dataset metadata, lazy caching, live opens, and lifetime hygiene."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import h5t

from .conftest import Result, open_fd_count, write_result


def test_metadata_and_all_attributes_are_snapshots(result_file: Path) -> None:
    result = Result.from_file(result_file)
    dataset = result.measurement
    assert dataset.path == "/measurement"
    assert dataset.shape == (5,)
    assert dataset.dtype == np.dtype("int64")
    assert dataset.ndim == 1
    assert dataset.unit == "m"
    assert dataset.scale == 1.0
    assert dataset.attrs["unit"] == "m"
    assert dataset.attrs["scale"] == 1.0
    assert dataset.attrs["extra"] == "measurement"
    with pytest.raises(TypeError):
        dataset.attrs["unit"] = "s"  # type: ignore[index]

    with h5py.File(result_file, "a") as file:
        del file["measurement"]
        changed = file.create_dataset("measurement", data=np.arange(10))
        changed.attrs["unit"] = "s"
    assert dataset.shape == (5,)
    assert dataset.unit == "m"
    assert dataset.attrs["unit"] == "m"


def test_forbidden_dataset_extra_attribute_fails_with_path(tmp_path: Path) -> None:
    class Strict(h5t.Dataset, extras="forbid"):
        unit: str

    class Owner(h5t.Group):
        payload: Strict

    path = tmp_path / "strict-dataset.h5"
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("payload", data=np.arange(3))
        dataset.attrs["unit"] = "m"
        dataset.attrs["surprise"] = 1
    with pytest.raises(h5t.ValidationError) as caught:
        Owner.from_file(path)
    assert caught.value.path == "/payload@surprise"


def test_lazy_data_observes_current_file_once_and_preserves_identity(result_file: Path) -> None:
    dataset = Result.from_file(result_file).measurement
    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10
    first = dataset.data
    assert np.array_equal(first, np.arange(5) + 10)
    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 20
    assert dataset.data is first
    assert dataset.read() is first


def test_eager_data_is_captured_during_loading(result_file: Path) -> None:
    dataset = Result.from_file(result_file).eager_measurement
    with h5py.File(result_file, "a") as file:
        file["eager_measurement"][...] = np.arange(5) + 100
    assert np.array_equal(dataset.data, np.arange(5))


def test_scalar_dataset_payload_is_zero_dimensional_array(tmp_path: Path) -> None:
    class Scalar(h5t.Group):
        plain: np.ndarray
        detached: h5t.Dataset

    path = tmp_path / "scalar.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("plain", data=7)
        file.create_dataset("detached", data=9)
    loaded = Scalar.from_file(path)
    assert isinstance(loaded.plain, np.ndarray) and loaded.plain.shape == ()
    assert isinstance(loaded.detached.data, np.ndarray) and loaded.detached.data.shape == ()


def test_open_supports_slices_repeated_opens_and_closes_on_exceptions(result_file: Path) -> None:
    dataset = Result.from_file(result_file).measurement
    with dataset.open() as live:
        assert np.array_equal(live[:2], [0, 1])
    assert not live.id.valid
    with dataset.open() as second:
        assert second.shape == (5,)
    with pytest.raises(RuntimeError):
        with dataset.open() as exceptional:
            raise RuntimeError("consumer failed")
    assert not exceptional.id.valid


def test_deleted_source_preserves_snapshot_but_live_access_fails(result_file: Path) -> None:
    dataset = Result.from_file(result_file).measurement
    os.unlink(result_file)
    assert dataset.shape == (5,)
    assert dataset.attrs["unit"] == "m"
    with pytest.raises(OSError):
        dataset.read()


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="requires POSIX descriptors")
def test_no_descriptors_leak_on_success_or_failure(tmp_path: Path) -> None:
    good = tmp_path / "good.h5"
    write_result(good)
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w"):
        pass
    Result.from_file(good)
    with pytest.raises(h5t.ValidationError):
        Result.from_file(bad)
    baseline = open_fd_count()
    for _ in range(20):
        Result.from_file(good)
        with pytest.raises(h5t.ValidationError):
            Result.from_file(bad)
    assert open_fd_count() == baseline
