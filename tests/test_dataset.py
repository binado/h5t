"""Detached dataset metadata, lazy caching, live opens, and lifetime hygiene."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest

import h5t

from .conftest import EagerMeasurement, LazyMeasurement, Result, open_fd_count, write_result


def test_metadata_and_all_attributes_are_snapshots(result_file: Path) -> None:
    @h5t.dataset(data="data", attrs="attrs")
    @dataclasses.dataclass
    class Snapshot:
        unit: str
        data: h5t.LazyArray
        attrs: dict[str, object]
        scale: float = 1.0

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Snapshot

    measurement = h5t.load(Owner, result_file).measurement
    assert measurement.data.path == "/measurement"
    assert measurement.data.shape == (5,)
    assert measurement.data.dtype == np.dtype("int64")
    assert measurement.data.ndim == 1
    assert measurement.unit == "m"
    assert measurement.scale == 1.0
    assert measurement.attrs["unit"] == "m"
    assert measurement.attrs["scale"] == 1.0
    assert measurement.attrs["extra"] == "measurement"

    with h5py.File(result_file, "a") as file:
        del file["measurement"]
        changed = file.create_dataset("measurement", data=np.arange(10))
        changed.attrs["unit"] = "s"
    assert measurement.data.shape == (5,)
    assert measurement.unit == "m"
    assert measurement.attrs["unit"] == "m"


def test_forbidden_dataset_extra_attribute_fails_with_path(tmp_path: Path) -> None:
    @h5t.dataset(data="data", extras="forbid")
    @dataclasses.dataclass
    class Strict:
        unit: str
        data: h5t.LazyArray

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        payload: Strict

    path = tmp_path / "strict-dataset.h5"
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("payload", data=np.arange(3))
        dataset.attrs["unit"] = "m"
        dataset.attrs["surprise"] = 1
    with pytest.raises(h5t.ValidationError) as caught:
        h5t.load(Owner, path)
    assert caught.value.path == "/payload@surprise"


def test_lazy_data_observes_current_file_once_and_preserves_identity(result_file: Path) -> None:
    payload = h5t.load(Result, result_file).measurement.data
    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10
    first = payload.data
    assert np.array_equal(first, np.arange(5) + 10)
    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 20
    assert payload.data is first
    assert payload.read() is first


def test_eager_data_is_captured_during_loading(result_file: Path) -> None:
    payload = h5t.load(Result, result_file).eager_measurement.data
    with h5py.File(result_file, "a") as file:
        file["eager_measurement"][...] = np.arange(5) + 100
    assert np.array_equal(payload.data, np.arange(5))


def test_scalar_dataset_payload_is_zero_dimensional_array(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Scalar:
        plain: np.ndarray
        detached: h5t.LazyArray

    path = tmp_path / "scalar.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("plain", data=7)
        file.create_dataset("detached", data=9)
    loaded = h5t.load(Scalar, path)
    assert isinstance(loaded.plain, np.ndarray) and loaded.plain.shape == ()
    assert isinstance(loaded.detached.data, np.ndarray) and loaded.detached.data.shape == ()


def test_open_supports_slices_repeated_opens_and_closes_on_exceptions(result_file: Path) -> None:
    payload = h5t.load(Result, result_file).measurement.data
    with payload.open() as live:
        assert np.array_equal(live[:2], [0, 1])
    assert not live.id.valid
    with payload.open() as second:
        assert second.shape == (5,)
    with pytest.raises(RuntimeError):
        with payload.open() as exceptional:
            raise RuntimeError("consumer failed")
    assert not exceptional.id.valid


def test_deleted_source_preserves_snapshot_but_live_access_fails(result_file: Path) -> None:
    measurement = h5t.load(Result, result_file).measurement
    os.unlink(result_file)
    assert measurement.data.shape == (5,)
    assert measurement.unit == "m"
    with pytest.raises(OSError):
        measurement.data.read()


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="requires POSIX descriptors")
def test_no_descriptors_leak_on_success_or_failure(tmp_path: Path) -> None:
    @h5t.group(extras="ignore")
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[LazyMeasurement, h5t.Payload("data")]
        eager_measurement: EagerMeasurement
        values: h5t.LazyArray

    good = tmp_path / "good.h5"
    write_result(good)
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w"):
        pass

    def exercise() -> None:
        h5t.load(Result, good)
        h5t.load(Owner, good).measurement.data.read()
        h5t.load(Owner, good).values.read()
        with pytest.raises(h5t.ValidationError):
            h5t.load(Result, bad)
        with pytest.raises(h5t.ValidationError):
            h5t.load(Owner, bad)

    exercise()
    baseline = open_fd_count()
    for _ in range(20):
        exercise()
    assert open_fd_count() == baseline
