"""Record compilation: field classification, deferral, and declaration errors."""

from __future__ import annotations

import dataclasses
import gc
import sys
import typing
import weakref
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pytest

import h5t
from h5t._compile import _PendingRecord
from h5t._spec import ForeignSpec, MemberKind


def compiled(record: type) -> ForeignSpec:
    """Return ``record``'s compiled spec without going through a load."""
    marker = record.__dict__["__h5t_record__"]
    assert isinstance(marker, ForeignSpec), "record is still pending"
    return marker


def field_map(record: type) -> dict[str, Any]:
    return {field.py_name: field for field in compiled(record).spec.fields}


def test_every_field_kind_and_markers_compile() -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Child:
        value: int

    @h5t.dataset(data="data")
    @dataclasses.dataclass
    class Typed:
        unit: str
        data: h5t.LazyArray

    @h5t.group()
    @dataclasses.dataclass
    class Schema:
        renamed: Annotated[int, h5t.Name("on-disk")]
        config: Annotated[dict[str, Any], h5t.Attr()]
        array: np.ndarray
        bare: h5t.LazyArray
        lazy: Typed
        eager: Annotated[Typed, h5t.Eager()]
        child: Child

    fields = field_map(Schema)
    assert fields["renamed"].kind is MemberKind.ATTRIBUTE
    assert fields["renamed"].h5_name == "on-disk"
    assert fields["config"].kind is MemberKind.ATTRIBUTE
    assert fields["array"].kind is MemberKind.ARRAY
    assert fields["bare"].kind is MemberKind.DATASET
    assert fields["bare"].foreign is None  # no record type to construct
    assert fields["lazy"].kind is MemberKind.DATASET
    assert fields["lazy"].foreign is not None
    assert fields["eager"].eager
    assert fields["child"].kind is MemberKind.GROUP
    # A GROUP-kind member stays unresolved until load time, so a self-referential
    # schema can compile at all; only the class is recorded here.
    assert fields["child"].foreign is None
    assert fields["child"].member_type is Child


def test_ndarray_typing_aliases_classify_as_arrays() -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Aliased:
        payload: npt.NDArray[np.float64]
        optional_payload: npt.NDArray[Any] | None

    fields = field_map(Aliased)
    assert fields["payload"].kind is MemberKind.ARRAY
    assert fields["optional_payload"].kind is MemberKind.ARRAY
    assert fields["optional_payload"].optional


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 aliases require Python 3.12")
def test_pep695_aliases_of_ndarray_classify_as_arrays() -> None:
    # A PEP 695 alias defers its right-hand side, so typing.get_origin stops at the
    # alias rather than reaching np.ndarray. numpy >= 2.5 defines npt.NDArray this
    # way itself; a user aliasing one -- or aliasing that alias -- must resolve too.
    # Built through the runtime constructor so this file needs no 3.12-only syntax.
    Coords = typing.TypeAliasType("Coords", npt.NDArray[np.float64])
    Chained = typing.TypeAliasType("Chained", Coords)

    @h5t.group()
    @dataclasses.dataclass
    class Aliased:
        direct: Coords
        chained: Chained

    fields = field_map(Aliased)
    assert fields["direct"].kind is MemberKind.ARRAY
    assert fields["chained"].kind is MemberKind.ARRAY
    # Normalized to the plain type, so an alias and a bare ndarray stay equivalent.
    assert fields["direct"].annotation is np.ndarray


def test_a_resolvable_record_compiles_at_the_decorator() -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Eagerly:
        value: int

    assert [field.py_name for field in compiled(Eagerly).spec.fields] == ["value"]


def test_a_record_declared_in_a_function_compiles_before_it_returns() -> None:
    def declare() -> type:
        @h5t.dataset(data="data")
        @dataclasses.dataclass
        class Payload:
            unit: str
            data: h5t.LazyArray

        @h5t.group()
        @dataclasses.dataclass
        class Local:
            payload: Payload

        # The decorator's defining frame is still live, so a function-local dependency
        # resolves without the scope being retained for later.
        assert isinstance(Local.__dict__["__h5t_record__"], ForeignSpec)
        return Local

    assert field_map(declare())["payload"].kind is MemberKind.DATASET


def test_eager_compilation_does_not_retain_the_defining_scope() -> None:
    class Tracked:  # object() cannot be weakly referenced; an instance can
        pass

    def declare() -> tuple[type, weakref.ref[Tracked]]:
        unrelated = Tracked()

        @h5t.group()
        @dataclasses.dataclass
        class Local:
            value: int

        return Local, weakref.ref(unrelated)

    record, ref = declare()
    gc.collect()
    assert ref() is None
    assert compiled(record).spec.fields[0].py_name == "value"


def test_deferred_path_snapshots_the_scope_weakly() -> None:
    # Accepted limitation: a record that both lives in a function scope *and* forward-
    # references a class defined later in that same function may find the referent
    # collected before first use. That is the intersection of two rare cases, and the
    # same tradeoff pydantic ships. Eager compilation makes it rarer than a lazy
    # default would, since ordinary function-local records never defer.
    @h5t.group()
    @dataclasses.dataclass
    class Dependency:
        value: int

    label = "on-disk"

    @h5t.group()
    @dataclasses.dataclass
    class Deferred:
        dependency: Dependency
        renamed: Annotated[int, h5t.Name(label)]
        undefined: _NeverDefined  # noqa: F821 -- forces the deferred path

    pending = Deferred.__dict__["__h5t_record__"]
    assert isinstance(pending, _PendingRecord)
    assert isinstance(pending.scope["Dependency"], weakref.ref)
    assert pending.scope["Dependency"]() is Dependency
    # A str rejects weakref.ref, so it is the one kind of entry held strongly.
    assert pending.scope["label"] == label


def test_deferred_path_does_not_retain_locals_the_annotations_never_name() -> None:
    # The snapshot must hold no more than annotation resolution can ask for: a
    # container rejects weakref.ref, so an unfiltered snapshot would pin whatever it
    # holds for as long as the deferred record lives.
    class Tracked:  # object() cannot be weakly referenced; an instance can
        pass

    def declare() -> tuple[type, weakref.ref[Tracked]]:
        tracked = Tracked()
        unrelated = [tracked]
        assert unrelated[0] is tracked

        @h5t.group()
        @dataclasses.dataclass
        class Local:
            undefined: _NeverDefined  # noqa: F821 -- forces the deferred path

        return Local, weakref.ref(tracked)

    record, ref = declare()
    gc.collect()
    assert record.__dict__["__h5t_record__"].scope == {}
    assert ref() is None
    # h5t.load is where a still-unresolved name becomes the user-visible SchemaError;
    # it retries the record before touching the filesystem.
    with pytest.raises(h5t.SchemaError, match="_NeverDefined"):
        h5t.load(record, "never-opened.h5")


def test_a_broken_annotation_helper_is_not_a_forward_reference() -> None:
    # A NameError escaping a function the annotation calls is a bug in that function,
    # not a name bound later, so it must raise at the decorator instead of silently
    # deferring until the record is used.
    def alias():
        return _typo_inside_the_helper  # noqa: F821

    with pytest.raises(h5t.SchemaError, match="_typo_inside_the_helper"):

        @h5t.group()
        @dataclasses.dataclass
        class Broken:
            value: alias()


@pytest.mark.parametrize(
    "declaration, match",
    [
        ("value: list[int]", "containers require Attr"),
        ("value: Annotated[int, h5t.Eager()]", "Eager applies only"),
        (
            "value: Annotated[int, h5t.Attr(), h5t.Eager()]",
            "Attr and Eager cannot be combined",
        ),
        ("value: Annotated[Child, h5t.Attr()]", "Attr cannot annotate"),
        ("value: Annotated[h5t.LazyArray, h5t.Attr()]", "Attr cannot annotate"),
        ("value: Annotated[np.ndarray, h5t.Name('a/b')]", "cannot contain"),
        ("value: Annotated[int, h5t.Name(''), h5t.Name('b')]", "may appear only once"),
    ],
)
def test_invalid_declarations(declaration: str, match: str) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Child:
        pass

    namespace = {
        "__name__": __name__,
        "dataclasses": dataclasses,
        "h5t": h5t,
        "np": np,
        "Annotated": Annotated,
        "Child": Child,
    }
    source = "@h5t.group()\n@dataclasses.dataclass\nclass Invalid:\n    " + declaration + "\n"
    with pytest.raises(h5t.SchemaError, match=match):
        exec(source, namespace)
    assert "Invalid" not in namespace  # the decorator itself raised


def test_dataset_record_only_accepts_attributes_and_extras_has_two_values() -> None:
    with pytest.raises(h5t.SchemaError, match="only attributes"):

        @h5t.dataset(data="data")
        @dataclasses.dataclass
        class BadDataset:
            data: np.ndarray
            payload: np.ndarray

    with pytest.raises(h5t.SchemaError, match="ignore.*forbid"):

        @h5t.group(extras="warn")  # type: ignore[arg-type]
        @dataclasses.dataclass
        class Warn:
            pass


def test_duplicate_hdf5_names_fail() -> None:
    with pytest.raises(h5t.SchemaError, match="duplicate HDF5"):

        @h5t.group()
        @dataclasses.dataclass
        class Duplicate:
            first: Annotated[str, h5t.Name("same")]
            second: Annotated[int, h5t.Name("same")]


def test_a_field_may_be_named_after_anything_h5t_exposes() -> None:
    # The inheritance API reserved every public Group/Dataset attribute against a field
    # name. A record inherits nothing from h5t, so nothing is reserved.
    @h5t.group()
    @dataclasses.dataclass
    class Unreserved:
        attrs: Annotated[str, h5t.Attr()]
        path: Annotated[str, h5t.Attr()]
        data: np.ndarray
        shape: Annotated[int, h5t.Attr()]

    assert set(field_map(Unreserved)) == {"attrs", "path", "data", "shape"}


def test_removed_api_is_absent() -> None:
    removed = ("Group", "Dataset", "File", "Keys", "Invalid", "ValidationReport", "f8")
    for name in removed:
        assert not hasattr(h5t, name)
        assert name not in h5t.__all__
