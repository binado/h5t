"""The ``@h5t.dataset`` / ``@h5t.group`` decorator API, in place of the ``Payload`` marker."""

from __future__ import annotations

import dataclasses
import gc
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest

import h5t
from h5t._spec import MemberKind

from .conftest import LazyMeasurement, Nested, Result

# Module-level, mirroring tests/test_compile.py's _ForwardRef/_DefinedLater: a decorated
# group record forward-references one defined later in this module, and another
# references itself. Both defer at decoration time (the referent is not bound yet) and
# resolve lazily against fresh module globals the first time they are actually used --
# a function-local equivalent would not resolve, since the snapshot @h5t.group keeps is
# taken before either name exists in the enclosing scope.


@h5t.group()
@dataclasses.dataclass
class _ForwardA:
    b: _ForwardB


@h5t.group()
@dataclasses.dataclass
class _ForwardB:
    value: int


@h5t.group()
@dataclasses.dataclass
class _RecursiveNode:
    value: int
    child: _RecursiveNode | None = None


def test_decorator_validates_eagerly_at_the_records_own_definition() -> None:
    with pytest.raises(h5t.SchemaError, match=r"Bad\.missing:"):

        @h5t.dataset(data="missing")
        @dataclasses.dataclass
        class Bad:
            unit: str
            data: np.ndarray


def test_foreign_cache_reuses_live_specs_without_retaining_record_classes() -> None:
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray

    @h5t.group()
    @dataclasses.dataclass
    class First:
        measurement: Annotated[Recording, h5t.Payload("data")]

    @h5t.group()
    @dataclasses.dataclass
    class Second:
        measurement: Annotated[Recording, h5t.Payload("data")]

    def measurement_field(record: type) -> Any:
        spec = record.__dict__["__h5t_record__"].spec
        return {field.py_name: field for field in spec.fields}["measurement"]

    assert measurement_field(First).foreign is measurement_field(Second).foreign

    def make_transient() -> weakref.ReferenceType[type]:
        @h5t.dataset(data="data")
        @dataclasses.dataclass
        class Transient:
            unit: str
            data: np.ndarray

        return weakref.ref(Transient)

    transient = make_transient()
    gc.collect()
    assert transient() is None

    def make_payload_transient() -> weakref.ReferenceType[type]:
        @dataclasses.dataclass
        class TransientRecord:
            unit: str
            data: np.ndarray

        h5t._compile._foreign_spec(
            TransientRecord,
            TransientRecord,
            "measurement",
            kind=h5t._spec.RecordKind.DATASET,
            data="data",
            attrs=None,
            extras_raw="ignore",
        )
        return weakref.ref(TransientRecord)

    record_ref = make_payload_transient()
    gc.collect()
    assert record_ref() is None


def test_decorated_record_defaults_are_instance_checked(result_file: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Child:
        answer: int

    @h5t.group()
    @dataclasses.dataclass
    class WithInvalidDefault:
        child: Child = 123  # type: ignore[assignment]

    @h5t.group()
    @dataclasses.dataclass
    class WithInvalidFactory:
        child: Child = dataclasses.field(default_factory=lambda: 123)  # type: ignore[arg-type]

    for schema in (WithInvalidDefault, WithInvalidFactory):
        with pytest.raises(h5t.ValidationError) as caught:
            h5t.load(schema, result_file)
        assert caught.value.path == "/child"


def test_decorated_class_is_usable_directly_with_no_annotated_or_payload(
    result_file: Path,
) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Recording

    result = h5t.load(Owner, result_file)
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

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Recording

    result = h5t.load(Owner, result_file)
    assert result.measurement.attrs["unit"] == "m"
    assert result.measurement.attrs["extra"] == "measurement"


def test_eager_marker_prefills_a_decorated_lazy_payload(result_file: Path) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: h5t.LazyArray

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[Recording, h5t.Eager()]

    result = h5t.load(Owner, result_file)
    assert isinstance(result.measurement.data, h5t.LazyArray)
    assert np.array_equal(result.measurement.data.data, np.arange(5))


def test_decorated_record_preserves_custom_lazyarray_subclass(result_file: Path) -> None:
    class CustomLazyArray(h5t.LazyArray):
        pass

    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: CustomLazyArray

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Annotated[Recording, h5t.Eager()]

    measurement = h5t.load(Owner, result_file).measurement
    assert type(measurement.data) is CustomLazyArray

    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10
    assert np.array_equal(measurement.data.data, np.arange(5))


def test_eager_marker_on_an_ndarray_decorated_payload_is_schema_error() -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray

    with pytest.raises(h5t.SchemaError, match="Eager and Payload"):

        @h5t.group()
        @dataclasses.dataclass
        class Owner:
            measurement: Annotated[Recording, h5t.Eager()]


def test_attr_marker_cannot_annotate_a_decorated_record_field() -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray

    with pytest.raises(h5t.SchemaError, match="Attr cannot annotate"):

        @h5t.group()
        @dataclasses.dataclass
        class Owner:
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

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        # This owner's field wants attrs=; the decorator declared none.
        measurement: Annotated[Recording, h5t.Payload("data", attrs=None, extras="forbid")]

    with pytest.raises(h5t.ValidationError) as caught:
        h5t.load(Owner, result_file)
    assert caught.value.path == "/measurement@extra"


def test_decorator_composes_with_frozen_slots_dataclass(result_file: Path) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass(frozen=True, slots=True)
    class Recording:
        unit: str
        data: np.ndarray

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Recording

    result = h5t.load(Owner, result_file)
    assert result.measurement.unit == "m"


def test_decorated_lazy_measurement_matches_marker_based_loading(result_file: Path) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class DecoratedLazyMeasurement:
        unit: str
        data: h5t.LazyArray
        scale: float = 1.0

    @h5t.group()
    @dataclasses.dataclass
    class ViaDecorator:
        measurement: DecoratedLazyMeasurement

    @h5t.group()
    @dataclasses.dataclass
    class ViaMarker:
        measurement: Annotated[LazyMeasurement, h5t.Payload("data")]

    decorated = h5t.load(ViaDecorator, result_file).measurement
    marked = h5t.load(ViaMarker, result_file).measurement
    assert decorated.unit == marked.unit
    assert decorated.scale == marked.scale
    assert decorated.data.path == marked.data.path


def test_decorated_group_record_is_usable_directly_with_no_annotated_or_payload(
    result_file: Path,
) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Root:
        version: int

    result = h5t.load(Root, result_file)
    assert isinstance(result, Root)
    assert result.version == 2


def test_forward_reference_to_a_record_defined_later_in_the_module_resolves() -> None:
    # The positive case the deferral mechanism exists for: _ForwardA names _ForwardB
    # before _ForwardB is defined, so decoration deferred it; by the time its fields
    # are read here, both module globals hold their final values.
    assert isinstance(_ForwardA.__dict__["__h5t_record__"], h5t._compile._PendingRecord)
    foreign = h5t._compile._ensure_record_compiled(_ForwardA)
    fields = {field.py_name: field for field in foreign.spec.fields}
    assert fields["b"].member_type is _ForwardB


def test_recursive_group_record_compiles_and_loads_a_nested_file(tmp_path: Path) -> None:
    # Regression for correction (a): a GROUP-kind member must resolve at load time
    # (mirroring how any GROUP-kind member is resolved), or this recurses forever at
    # compile time instead.
    path = tmp_path / "recursive.h5"
    with h5py.File(path, "w") as file:
        file.attrs["value"] = 1
        child = file.create_group("child")
        child.attrs["value"] = 2

    result = h5t.load(_RecursiveNode, path)
    assert result.value == 1
    assert result.child is not None
    assert result.child.value == 2
    assert result.child.child is None


def test_attrs_binding_on_a_group_record_matches_group_attrs(result_file: Path) -> None:
    @h5t.group(attrs="attrs")
    @dataclasses.dataclass
    class WithAttrs:
        version: int
        optional_note: str | None
        attrs: Mapping[str, Any]
        defaulted: int = 42

    result = h5t.load(WithAttrs, result_file)
    assert result.attrs["version"] == 2  # declared: validated value
    assert result.attrs["undeclared"] == "raw"  # undeclared: reachable raw
    # Absent declared names are recorded too, so `attrs` lists every declared name.
    assert result.attrs["optional_note"] is None
    assert result.attrs["defaulted"] == 42


def test_dataset_record_nested_inside_a_group_record(result_file: Path) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Recording

    result = h5t.load(Owner, result_file)
    assert isinstance(result.measurement, Recording)
    assert result.measurement.unit == "m"


def test_group_record_nested_inside_another_group_record(result_file: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Inner:
        answer: int

    @h5t.group()
    @dataclasses.dataclass
    class Outer:
        nested: Inner

    result = h5t.load(Outer, result_file)
    assert isinstance(result.nested, Inner)
    assert result.nested.answer == 42


def test_a_record_from_another_module_is_usable_as_a_field(result_file: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        nested: Nested

    result = h5t.load(Owner, result_file)
    assert isinstance(result.nested, Nested)
    assert result.nested.answer == 42


def test_still_unresolved_nested_group_record_is_a_schema_error(tmp_path: Path) -> None:
    # A nested @h5t.group record stays pending until load. _load_group_values compiles
    # it after the file is open; that retry must become SchemaError via _compiled_record,
    # matching a top-level h5t.load. Leaving _UnresolvedAnnotation uncaught would also
    # leak past `h5t check`, which only catches SchemaError.
    @h5t.group()
    @dataclasses.dataclass
    class Inner:
        later: _NeverDefined  # noqa: F821 -- never bound

    assert isinstance(Inner.__dict__["__h5t_record__"], h5t._compile._PendingRecord)

    @h5t.group()
    @dataclasses.dataclass
    class Outer:
        nested: Inner

    path = tmp_path / "nested.h5"
    with h5py.File(path, "w") as file:
        file.create_group("nested").attrs["later"] = 1

    for load in (
        lambda: h5t.load(Outer, path),
        lambda: h5t.load(Inner, path),
    ):
        with pytest.raises(h5t.SchemaError, match="_NeverDefined"):
            load()


def test_load_rejects_a_dataset_kind_record(tmp_path: Path) -> None:
    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Recording:
        unit: str
        data: np.ndarray

    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(h5t.SchemaError, match="dataset record"):
        h5t.load(Recording, path)


def test_load_rejects_a_plain_undecorated_class(tmp_path: Path) -> None:
    class Plain:
        pass

    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(h5t.SchemaError, match="not a decorated record"):
        h5t.load(Plain, path)


def test_eager_on_a_group_record_field_is_schema_error() -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Rec:
        value: int

    with pytest.raises(h5t.SchemaError, match="Eager applies only"):

        @h5t.group()
        @dataclasses.dataclass
        class Owner:
            nested: Annotated[Rec, h5t.Eager()]


def test_explicit_payload_on_a_group_record_type_with_non_attribute_fields() -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Rec:
        value: int
        arr: np.ndarray
        other: np.ndarray

    with pytest.raises(h5t.SchemaError, match="only attributes"):

        @h5t.group()
        @dataclasses.dataclass
        class Owner:
            nested: Annotated[Rec, h5t.Payload("arr")]


def test_extras_forbid_on_a_group_record_rejects_undeclared_members(result_file: Path) -> None:
    @h5t.group(extras="forbid")
    @dataclasses.dataclass
    class Strict:
        version: int

    with pytest.raises(h5t.ValidationError):
        h5t.load(Strict, result_file)


def test_the_shared_result_schema_loads_every_field_kind(result_file: Path) -> None:
    actual = h5t.load(Result, result_file)

    assert actual.version == 2
    assert actual.renamed == "current"
    assert actual.config == {"enabled": True}
    assert np.array_equal(actual.array_attr, np.arange(3))
    assert np.array_equal(actual.values, np.arange(4))
    assert actual.measurement.unit == "m"
    assert actual.measurement.scale == 1.0
    assert np.array_equal(actual.measurement.data.data, np.arange(5))
    assert actual.eager_measurement.unit == "m"
    assert np.array_equal(actual.eager_measurement.data.data, np.arange(5))
    assert isinstance(actual.nested, Nested)
    assert actual.nested.answer == 42
    assert actual.optional_note is None
    assert actual.defaulted == 42


def test_lazy_array_is_a_dataset_member_in_its_own_right(result_file: Path) -> None:
    # A child dataset whose attributes the schema does not care about needs no record
    # type at all: LazyArray names the payload directly.
    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: h5t.LazyArray
        eager_measurement: Annotated[h5t.LazyArray, h5t.Eager()]
        absent: h5t.LazyArray | None

    spec = Owner.__dict__["__h5t_record__"].spec
    field_kinds = {field.py_name: field.kind for field in spec.fields}
    assert field_kinds["measurement"] is MemberKind.DATASET
    assert field_kinds["absent"] is MemberKind.DATASET

    loaded = h5t.load(Owner, result_file)
    assert loaded.absent is None
    assert loaded.measurement.path == "/measurement"
    assert loaded.measurement.shape == (5,)
    assert loaded.measurement.dtype == np.dtype("int64")

    # Lazy: the payload reflects the file at first access, not at load time.
    with h5py.File(result_file, "a") as file:
        file["measurement"][...] = np.arange(5) + 10
        file["eager_measurement"][...] = np.arange(5) + 10
    assert np.array_equal(loaded.measurement.data, np.arange(5) + 10)
    # Eager: prefetched during the load, so the later write is invisible.
    assert np.array_equal(loaded.eager_measurement.data, np.arange(5))


def test_lazy_array_member_preserves_a_custom_subclass(result_file: Path) -> None:
    class Tracked(h5t.LazyArray):
        pass

    @h5t.group()
    @dataclasses.dataclass
    class Owner:
        measurement: Tracked

    loaded = h5t.load(Owner, result_file)
    assert type(loaded.measurement) is Tracked
    assert np.array_equal(loaded.measurement.read(), np.arange(5))


def test_lazy_array_member_rejects_attr_and_payload() -> None:
    with pytest.raises(h5t.SchemaError, match="Attr cannot annotate"):

        @h5t.group()
        @dataclasses.dataclass
        class WithAttr:
            measurement: Annotated[h5t.LazyArray, h5t.Attr()]

    with pytest.raises(h5t.SchemaError, match="Payload cannot annotate"):

        @h5t.group()
        @dataclasses.dataclass
        class WithPayload:
            measurement: Annotated[h5t.LazyArray, h5t.Payload("data")]
