"""Loading a child dataset into a plain record type via ``h5t.Payload``."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest

import h5t

from .conftest import LazyMeasurement, Measurement, PlainMeasurement


@dataclasses.dataclass
class _WithDefaults:
    unit: str
    data: np.ndarray
    note: str | None
    revision: int = dataclasses.field(default_factory=lambda: 9)


def test_eager_payload_is_materialized_and_attrs_validate(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: Annotated[PlainMeasurement, h5t.Payload("data")]

    result = Owner.from_file(result_file)
    measurement = result.measurement
    assert isinstance(measurement, PlainMeasurement)
    assert measurement.unit == "m"
    assert measurement.scale == 1.0
    assert isinstance(measurement.data, np.ndarray)
    assert np.array_equal(measurement.data, np.arange(5))


def test_payload_field_named_data_compiles(result_file: Path) -> None:
    # `data` is reserved on h5t.Dataset (dir()-based) but a plain class has nothing
    # in dir() for that check to forbid.
    class Owner(h5t.Group):
        measurement: Annotated[PlainMeasurement, h5t.Payload("data")]

    fields = {field.py_name: field for field in Owner.__h5spec__.fields}
    assert fields["measurement"].foreign is not None
    assert fields["measurement"].foreign.data == "data"
    Owner.from_file(result_file)  # does not raise


def test_defaults_and_optional_attrs_match_dataset_semantics(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: Annotated[_WithDefaults, h5t.Payload("data")]

    result = Owner.from_file(result_file)
    assert result.measurement.note is None
    assert result.measurement.revision == 9  # from default_factory, absent from file


def test_attrs_binding_snapshots_validated_and_raw_values(result_file: Path) -> None:
    @dataclasses.dataclass
    class WithAttrs:
        unit: str
        data: np.ndarray
        attrs: Mapping[str, Any]

    class Owner(h5t.Group):
        measurement: Annotated[WithAttrs, h5t.Payload("data", attrs="attrs")]

    result = Owner.from_file(result_file)
    snapshot = result.measurement.attrs
    assert snapshot["unit"] == "m"  # declared: validated value
    assert snapshot["extra"] == "measurement"  # undeclared: reachable raw


def test_eager_on_lazy_payload_prefills_without_a_second_read(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: Annotated[LazyMeasurement, h5t.Payload("data"), h5t.Eager()]

    result = Owner.from_file(result_file)
    measurement = result.measurement
    assert isinstance(measurement.data, h5t.LazyArray)

    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10

    # Prefilled during from_file: the post-load mutation above must not be seen.
    assert np.array_equal(measurement.data.data, np.arange(5))


def test_lazy_payload_snapshot_reads_once_and_caches(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: Annotated[LazyMeasurement, h5t.Payload("data")]

    result = Owner.from_file(result_file)
    measurement = result.measurement
    assert isinstance(measurement, LazyMeasurement)
    assert isinstance(measurement.data, h5t.LazyArray)
    assert measurement.data.path == "/measurement"
    assert measurement.data.shape == (5,)
    assert measurement.data.dtype == np.dtype("int64")
    assert measurement.data.ndim == 1

    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10
    first = measurement.data.data
    assert np.array_equal(first, np.arange(5) + 10)
    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 20
    assert measurement.data.data is first
    assert measurement.data.read() is first


def test_lazy_payload_open_slices_and_closes(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: Annotated[LazyMeasurement, h5t.Payload("data")]

    measurement = Owner.from_file(result_file).measurement
    with measurement.data.open() as live:
        assert np.array_equal(live[:2], [0, 1])
    assert not live.id.valid


def test_extras_forbid_on_marker_rejects_undeclared_attribute(result_file: Path) -> None:
    class Owner(h5t.Group):
        measurement: Annotated[PlainMeasurement, h5t.Payload("data", extras="forbid")]

    with pytest.raises(h5t.ValidationError) as caught:
        Owner.from_file(result_file)
    assert caught.value.path == "/measurement@extra"


def test_post_init_invariant_surfaces_as_validation_error_with_path(result_file: Path) -> None:
    @dataclasses.dataclass
    class Strict:
        unit: str
        data: np.ndarray

        def __post_init__(self) -> None:
            if self.unit != "s":
                raise ValueError(f"expected seconds, got {self.unit!r}")

    class Owner(h5t.Group):
        measurement: Annotated[Strict, h5t.Payload("data")]

    with pytest.raises(h5t.ValidationError, match="expected seconds") as caught:
        Owner.from_file(result_file)
    assert caught.value.path == "/measurement"


def test_missing_data_attr_field_is_schema_error() -> None:
    @dataclasses.dataclass
    class NoPayload:
        unit: str

    with pytest.raises(h5t.SchemaError, match="not a field"):

        class Owner(h5t.Group):
            measurement: Annotated[NoPayload, h5t.Payload("data")]


def test_payload_field_wrong_type_is_schema_error() -> None:
    @dataclasses.dataclass
    class BadPayload:
        unit: str
        data: str

    with pytest.raises(h5t.SchemaError, match="np.ndarray or LazyArray"):

        class Owner(h5t.Group):
            measurement: Annotated[BadPayload, h5t.Payload("data")]


def test_non_attribute_field_on_record_is_schema_error() -> None:
    @dataclasses.dataclass
    class BadRecord:
        unit: str
        data: np.ndarray
        extra_array: np.ndarray

    with pytest.raises(h5t.SchemaError, match="only attributes"):

        class Owner(h5t.Group):
            measurement: Annotated[BadRecord, h5t.Payload("data")]


def test_payload_combined_with_attr_or_eager_is_schema_error() -> None:
    with pytest.raises(h5t.SchemaError, match="Attr and Payload"):

        class WithAttr(h5t.Group):
            measurement: Annotated[PlainMeasurement, h5t.Payload("data"), h5t.Attr()]

    with pytest.raises(h5t.SchemaError, match="Eager and Payload"):

        class WithEager(h5t.Group):
            measurement: Annotated[PlainMeasurement, h5t.Payload("data"), h5t.Eager()]


def test_payload_on_dataset_subclass_is_schema_error() -> None:
    with pytest.raises(h5t.SchemaError, match="Payload cannot annotate"):

        class Owner(h5t.Group):
            measurement: Annotated[Measurement, h5t.Payload("data")]


def test_foreign_record_does_not_trip_member_adapter_construction(result_file: Path) -> None:
    # Regression: TypeAdapter(PlainMeasurement, config=...) raises PydanticUserError
    # (pydantic rejects `config=` alongside a dataclass type). The member adapter for
    # a foreign field must be built from Any instead, or this would raise SchemaError.
    class Owner(h5t.Group):
        measurement: Annotated[PlainMeasurement, h5t.Payload("data")]

    Owner.from_file(result_file)  # does not raise


def test_foreign_and_dataset_styles_coexist(result_file: Path) -> None:
    class Owner(h5t.Group, extras="ignore"):
        measurement: Measurement
        eager_measurement: Annotated[PlainMeasurement, h5t.Payload("data")]

    result = Owner.from_file(result_file)
    assert isinstance(result.measurement, Measurement)
    assert isinstance(result.eager_measurement, PlainMeasurement)
    assert np.array_equal(result.eager_measurement.data, np.arange(5))
