"""The ``@h5t.dataset`` / ``@h5t.group`` decorator API, in place of the ``Payload`` marker."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pytest

import h5t

from .conftest import LazyMeasurement


def test_decorator_validates_eagerly_at_the_records_own_definition() -> None:
    with pytest.raises(h5t.SchemaError, match=r"Bad\.missing:"):

        @h5t.dataset(data="missing")
        @dataclasses.dataclass
        class Bad:
            unit: str
            data: np.ndarray


def test_decorated_class_is_usable_directly_with_no_annotated_or_payload(
    result_file: Path,
) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray

    class Owner(h5t.Group):
        measurement: Recording

    result = Owner.from_file(result_file)
    assert isinstance(result.measurement, Recording)
    assert result.measurement.unit == "m"
    assert np.array_equal(result.measurement.data, np.arange(5))


def test_attrs_binding_works_the_same_as_the_marker(result_file: Path) -> None:
    @h5t.dataset(data="data", attrs="attrs")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray
        attrs: Mapping[str, Any]

    class Owner(h5t.Group):
        measurement: Recording

    result = Owner.from_file(result_file)
    assert result.measurement.attrs["unit"] == "m"
    assert result.measurement.attrs["extra"] == "measurement"


def test_eager_marker_prefills_a_decorated_lazy_payload(result_file: Path) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: h5t.LazyArray

    class Owner(h5t.Group):
        measurement: Annotated[Recording, h5t.Eager()]

    result = Owner.from_file(result_file)
    assert isinstance(result.measurement.data, h5t.LazyArray)
    assert np.array_equal(result.measurement.data.data, np.arange(5))


def test_eager_marker_on_an_ndarray_decorated_payload_is_schema_error() -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray

    with pytest.raises(h5t.SchemaError, match="Eager and Payload"):

        class Owner(h5t.Group):
            measurement: Annotated[Recording, h5t.Eager()]


def test_attr_marker_cannot_annotate_a_decorated_record_field() -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray

    with pytest.raises(h5t.SchemaError, match="Attr cannot annotate"):

        class Owner(h5t.Group):
            measurement: Annotated[Recording, h5t.Attr()]


def test_explicit_payload_at_use_site_overrides_the_decorators_own_binding(
    result_file: Path,
) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray
        scale: float = 1.0

    class Owner(h5t.Group):
        # This owner's field wants attrs=; the decorator declared none.
        measurement: Annotated[Recording, h5t.Payload("data", attrs=None, extras="forbid")]

    with pytest.raises(h5t.ValidationError) as caught:
        Owner.from_file(result_file)
    assert caught.value.path == "/measurement@extra"


def test_decorator_composes_with_frozen_slots_dataclass(result_file: Path) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass(frozen=True, slots=True)
    class Recording:
        unit: str
        data: np.ndarray

    class Owner(h5t.Group):
        measurement: Recording

    result = Owner.from_file(result_file)
    assert result.measurement.unit == "m"


def test_decorated_lazy_measurement_matches_marker_based_loading(result_file: Path) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class DecoratedLazyMeasurement:
        unit: str
        data: h5t.LazyArray
        scale: float = 1.0

    class ViaDecorator(h5t.Group):
        measurement: DecoratedLazyMeasurement

    class ViaMarker(h5t.Group):
        measurement: Annotated[LazyMeasurement, h5t.Payload("data")]

    decorated = ViaDecorator.from_file(result_file).measurement
    marked = ViaMarker.from_file(result_file).measurement
    assert decorated.unit == marked.unit
    assert decorated.scale == marked.scale
    assert decorated.data.path == marked.data.path
