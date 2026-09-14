"""Loading a child dataset into a plain record type via ``h5t.Payload``."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

import h5t

from .conftest import EagerMeasurement, LazyMeasurement


@dataclasses.dataclass
class _WithDefaults:
    unit: str
    data: np.ndarray
    note: str | None
    revision: int = dataclasses.field(default_factory=lambda: 9)


class _CustomLazyArray(h5t.LazyArray):
    pass


def test_eager_payload_is_materialized_and_attrs_validate(result_file: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[EagerMeasurement, h5t.Payload("data")]

    result = h5t.load(Owner, result_file)
    measurement = result.measurement
    assert isinstance(measurement, EagerMeasurement)
    assert measurement.unit == "m"
    assert measurement.scale == 1.0
    assert isinstance(measurement.data, np.ndarray)
    assert np.array_equal(measurement.data, np.arange(5))


def test_payload_field_named_data_compiles(result_file: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[EagerMeasurement, h5t.Payload("data")]

    spec = Owner.__dict__["__h5t_record__"].spec
    fields = {field.py_name: field for field in spec.fields}
    assert fields["measurement"].foreign is not None
    assert fields["measurement"].foreign.data == "data"
    h5t.load(Owner, result_file)  # does not raise


def test_defaults_and_optional_attrs_match_dataset_semantics(result_file: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[_WithDefaults, h5t.Payload("data")]

    result = h5t.load(Owner, result_file)
    assert result.measurement.note is None
    assert result.measurement.revision == 9  # from default_factory, absent from file


def test_inherited_plain_class_default_is_loaded_and_snapshotted(result_file: Path) -> None:
    class Base:
        revision: int = 9

    class Recording(Base):
        unit: str
        data: np.ndarray
        attrs: Mapping[str, Any]

        def __init__(
            self, *, revision: int, unit: str, data: np.ndarray, attrs: Mapping[str, Any]
        ) -> None:
            self.revision = revision
            self.unit = unit
            self.data = data
            self.attrs = attrs

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[Recording, h5t.Payload("data", attrs="attrs")]

    result = h5t.load(Owner, result_file).measurement
    assert result.revision == 9
    assert result.attrs["revision"] == 9


def test_constructor_signature_default_is_loaded_for_pydantic_model(result_file: Path) -> None:
    class Recording(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        unit: str
        data: np.ndarray
        attrs: Mapping[str, Any]
        scale: float = 1.0

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[Recording, h5t.Payload("data", attrs="attrs")]

    result = h5t.load(Owner, result_file).measurement
    assert result.scale == 1.0
    assert result.attrs["scale"] == 1.0


def test_attrs_binding_snapshots_validated_and_raw_values(result_file: Path) -> None:
    @dataclasses.dataclass
    class WithAttrs:
        unit: str
        data: np.ndarray
        attrs: Mapping[str, Any]

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[WithAttrs, h5t.Payload("data", attrs="attrs")]

    result = h5t.load(Owner, result_file)
    snapshot = result.measurement.attrs
    assert snapshot["unit"] == "m"  # declared: validated value
    assert snapshot["extra"] == "measurement"  # undeclared: reachable raw


def test_eager_on_lazy_payload_prefills_without_a_second_read(result_file: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[LazyMeasurement, h5t.Payload("data"), h5t.Eager()]

    result = h5t.load(Owner, result_file)
    measurement = result.measurement
    assert isinstance(measurement.data, h5t.LazyArray)

    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10

    # Prefilled during h5t.load: the post-load mutation above must not be seen.
    assert np.array_equal(measurement.data.data, np.arange(5))


def test_payload_preserves_custom_lazyarray_subclass(result_file: Path) -> None:
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: _CustomLazyArray

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[Recording, h5t.Payload("data")]

    measurement = h5t.load(Owner, result_file).measurement
    assert type(measurement.data) is _CustomLazyArray

    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10
    assert np.array_equal(measurement.data.data, np.arange(5) + 10)


def test_lazy_payload_snapshot_reads_once_and_caches(result_file: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[LazyMeasurement, h5t.Payload("data")]

    result = h5t.load(Owner, result_file)
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
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[LazyMeasurement, h5t.Payload("data")]

    measurement = h5t.load(Owner, result_file).measurement
    with measurement.data.open() as live:
        assert np.array_equal(live[:2], [0, 1])
    assert not live.id.valid


def test_extras_forbid_on_marker_rejects_undeclared_attribute(result_file: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[EagerMeasurement, h5t.Payload("data", extras="forbid")]

    with pytest.raises(h5t.ValidationError) as caught:
        h5t.load(Owner, result_file)
    assert caught.value.path == "/measurement@extra"


def test_post_init_invariant_surfaces_as_validation_error_with_path(result_file: Path) -> None:
    @dataclasses.dataclass
    class Strict:
        unit: str
        data: np.ndarray

        def __post_init__(self) -> None:
            if self.unit != "s":
                raise ValueError(f"expected seconds, got {self.unit!r}")

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[Strict, h5t.Payload("data")]

    with pytest.raises(h5t.ValidationError, match="expected seconds") as caught:
        h5t.load(Owner, result_file)
    assert caught.value.path == "/measurement"


def test_type_error_from_post_init_is_a_validation_error(result_file: Path) -> None:
    @dataclasses.dataclass
    class Strict:
        unit: str
        data: np.ndarray

        def __post_init__(self) -> None:
            raise TypeError("invalid measurement invariant")

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[Strict, h5t.Payload("data")]

    with pytest.raises(h5t.ValidationError, match="invalid measurement invariant") as caught:
        h5t.load(Owner, result_file)
    assert caught.value.path == "/measurement"


def test_constructor_signature_mismatch_is_a_schema_error(result_file: Path) -> None:
    class Mismatched:
        unit: str
        data: np.ndarray

        def __init__(self, *, unit: str) -> None:
            self.unit = unit

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[Mismatched, h5t.Payload("data")]

    with pytest.raises(h5t.SchemaError, match="cannot construct from loaded fields"):
        h5t.load(Owner, result_file)


def test_missing_data_attr_field_is_schema_error() -> None:
    @dataclasses.dataclass
    class NoPayload:
        unit: str

    with pytest.raises(h5t.SchemaError, match="not a field"):

        @h5t.group()
        @dataclasses.dataclass
        class Owner:
            measurement: Annotated[NoPayload, h5t.Payload("data")]


def test_payload_field_wrong_type_is_schema_error() -> None:
    @dataclasses.dataclass
    class BadPayload:
        unit: str
        data: str

    with pytest.raises(h5t.SchemaError, match="np.ndarray or LazyArray"):

        @h5t.group()
        @dataclasses.dataclass
        class Owner:
            measurement: Annotated[BadPayload, h5t.Payload("data")]


def test_non_attribute_field_on_record_is_schema_error() -> None:
    @dataclasses.dataclass
    class BadRecord:
        unit: str
        data: np.ndarray
        extra_array: np.ndarray

    with pytest.raises(h5t.SchemaError, match="only attributes"):

        @h5t.group()
        @dataclasses.dataclass
        class Owner:
            measurement: Annotated[BadRecord, h5t.Payload("data")]


def test_payload_combined_with_attr_or_eager_is_schema_error() -> None:
    with pytest.raises(h5t.SchemaError, match="Attr and Payload"):

        @h5t.group()
        @dataclasses.dataclass
        class WithAttr:
            measurement: Annotated[EagerMeasurement, h5t.Payload("data"), h5t.Attr()]

    with pytest.raises(h5t.SchemaError, match="Eager and Payload"):

        @h5t.group()
        @dataclasses.dataclass
        class WithEager:
            measurement: Annotated[EagerMeasurement, h5t.Payload("data"), h5t.Eager()]


def test_foreign_record_does_not_trip_member_adapter_construction(result_file: Path) -> None:
    # Regression: TypeAdapter(EagerMeasurement, config=...) raises PydanticUserError
    # (pydantic rejects `config=` alongside a dataclass type). The member adapter for
    # a foreign field must be built from Any instead, or this would raise SchemaError.
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[EagerMeasurement, h5t.Payload("data")]

    h5t.load(Owner, result_file)  # does not raise
