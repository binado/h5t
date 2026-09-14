"""Round-trip coverage for writing records to new HDF5 files."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import pytest

import h5t

from .conftest import EagerMeasurement, Nested


@h5t.group()
@dataclasses.dataclass
class WritableResult:
    """Small writable composition of the canonical record schemas."""

    version: int
    values: np.ndarray
    measurement: EagerMeasurement
    nested: Nested
    note: str | None


def test_dump_round_trips_canonical_schemas(tmp_path: Path) -> None:
    path = tmp_path / "round-trip.h5"
    expected = WritableResult(
        version=2,
        values=np.arange(4),
        measurement=EagerMeasurement(unit="m", data=np.arange(5), scale=2.5),
        nested=Nested(answer=42),
        note=None,
    )

    h5t.dump(expected, path, root="/result")
    actual = h5t.load(WritableResult, path, root="/result")

    assert actual.version == expected.version
    np.testing.assert_array_equal(actual.values, expected.values)
    assert actual.measurement.unit == expected.measurement.unit
    assert actual.measurement.scale == expected.measurement.scale
    np.testing.assert_array_equal(actual.measurement.data, expected.measurement.data)
    assert actual.nested == expected.nested
    assert actual.note is None
    with h5py.File(path, "r") as file:
        assert "note" not in file["/result"].attrs


def test_dump_requires_a_new_destination(tmp_path: Path) -> None:
    path = tmp_path / "existing.h5"
    path.write_bytes(b"do not replace")
    record = WritableResult(1, np.arange(1), EagerMeasurement("m", np.arange(1)), Nested(1), None)

    with pytest.raises(FileExistsError):
        h5t.dump(record, path)

    assert path.read_bytes() == b"do not replace"


def test_dump_reports_write_time_type_errors_with_member_path(tmp_path: Path) -> None:
    record = WritableResult(
        version="wrong",  # type: ignore[arg-type]
        values=np.arange(1),
        measurement=EagerMeasurement("m", np.arange(1)),
        nested=Nested(1),
        note=None,
    )

    with pytest.raises(h5t.ValidationError, match=r"^/result@version: "):
        h5t.dump(record, tmp_path / "invalid.h5", root="/result")
