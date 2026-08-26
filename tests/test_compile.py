"""Class creation and field classification."""

from __future__ import annotations

import gc
import weakref
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pytest

import h5t
from h5t._spec import Extras, MemberKind


def test_every_field_kind_and_markers_compile() -> None:
    class Child(h5t.Group):
        value: int

    class Typed(h5t.Dataset):
        unit: str

    class Schema(h5t.Group):
        renamed: Annotated[int, h5t.Name("on-disk")]
        config: Annotated[dict[str, Any], h5t.Attr()]
        array: np.ndarray
        lazy: Typed
        eager: Annotated[Typed, h5t.Eager()]
        child: Child

    fields = {field.py_name: field for field in Schema.__h5spec__.fields}
    assert fields["renamed"].kind is MemberKind.ATTRIBUTE
    assert fields["renamed"].h5_name == "on-disk"
    assert fields["config"].kind is MemberKind.ATTRIBUTE
    assert fields["array"].kind is MemberKind.ARRAY
    assert fields["lazy"].kind is MemberKind.DATASET
    assert fields["eager"].eager
    assert fields["child"].kind is MemberKind.GROUP


def test_ndarray_typing_aliases_classify_as_arrays() -> None:
    class Aliased(h5t.Group):
        payload: npt.NDArray[np.float64]
        optional_payload: npt.NDArray[Any] | None

    fields = {field.py_name: field for field in Aliased.__h5spec__.fields}
    assert fields["payload"].kind is MemberKind.ARRAY
    assert fields["optional_payload"].kind is MemberKind.ARRAY
    assert fields["optional_payload"].optional


def test_plain_and_aliased_ndarray_are_equivalent_in_diamonds() -> None:
    class PlainBase(h5t.Group):
        values: np.ndarray

    class AliasBase(h5t.Group):
        values: npt.NDArray[np.float64]

    class Diamond(PlainBase, AliasBase):
        pass

    fields = {field.py_name: field for field in Diamond.__h5spec__.fields}
    assert fields["values"].kind is MemberKind.ARRAY


class _ForwardRef(h5t.Group):
    later: _DefinedLater


class _DefinedLater(h5t.Group):
    value: int


def test_annotations_may_name_classes_defined_later() -> None:
    # _DefinedLater was undefined at the class statement, so _ForwardRef took the
    # deferred path and is still uncompiled until its spec is read.
    assert "_h5t_spec" not in _ForwardRef.__dict__
    fields = {field.py_name: field for field in _ForwardRef.__h5spec__.fields}
    assert fields["later"].kind is MemberKind.GROUP
    assert fields["later"].member_type is _DefinedLater


def test_valid_schema_compiles_at_the_class_statement() -> None:
    class Eagerly(h5t.Group):
        value: int

    assert "_h5t_spec" in Eagerly.__dict__
    assert "_h5t_scope" not in Eagerly.__dict__
    assert [field.py_name for field in Eagerly.__h5spec__.fields] == ["value"]


def test_schema_declared_in_a_function_compiles_before_it_returns() -> None:
    def declare() -> type[h5t.Group]:
        class Payload(h5t.Dataset):
            unit: str

        class Local(h5t.Group):
            payload: Payload

        # The defining frame is still live, so a function-local dependency resolves
        # without the scope being retained for later.
        assert "_h5t_spec" in Local.__dict__
        assert "_h5t_scope" not in Local.__dict__
        return Local

    fields = {field.py_name: field for field in declare().__h5spec__.fields}
    assert fields["payload"].kind is MemberKind.DATASET


def test_eager_compilation_does_not_retain_the_defining_scope() -> None:
    class Tracked:  # object() cannot be weakly referenced; an instance can
        pass

    def declare() -> tuple[type[h5t.Group], weakref.ref[Tracked]]:
        unrelated = Tracked()

        class Local(h5t.Group):
            value: int

        return Local, weakref.ref(unrelated)

    schema, ref = declare()
    gc.collect()
    assert ref() is None
    assert schema.__h5spec__.fields[0].py_name == "value"


def test_deferred_path_snapshots_the_scope_weakly() -> None:
    # Accepted limitation: a schema that both lives in a function scope *and* forward-
    # references a class defined later in that same function may find the referent
    # collected before first use. That is the intersection of two rare cases, and the
    # same tradeoff pydantic ships. Eager compilation makes it rarer than a lazy
    # default would, since ordinary function-local schemas never defer.
    class Dependency(h5t.Group):
        value: int

    label = "text"
    count = 3

    class Deferred(h5t.Group):
        undefined: _NeverDefined  # noqa: F821 -- forces the deferred path

    snapshot = Deferred.__dict__["_h5t_scope"]
    assert isinstance(snapshot["Dependency"], weakref.ref)
    assert snapshot["Dependency"]() is Dependency
    assert snapshot["label"] == label
    assert snapshot["count"] == count


def test_inheritance_defaults_and_extras() -> None:
    class Base(h5t.Group, extras="forbid"):
        inherited: int

    class Derived(Base):
        own: str | None = "fallback"

    assert [field.py_name for field in Derived.__h5spec__.fields] == ["inherited", "own"]
    assert Derived.__h5spec__.extras is Extras.FORBID


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
    ],
)
def test_invalid_declarations(declaration: str, match: str) -> None:
    class Child(h5t.Group):
        pass

    namespace = {
        "__name__": __name__,
        "h5t": h5t,
        "Annotated": Annotated,
        "Child": Child,
    }
    with pytest.raises(h5t.SchemaError, match=match):
        exec(f"class Invalid(h5t.Group):\n    {declaration}", namespace)
    assert "Invalid" not in namespace  # the class statement itself raised


def test_dataset_only_accepts_attributes_and_extras_has_two_values() -> None:
    with pytest.raises(h5t.SchemaError, match="only attributes"):

        class BadDataset(h5t.Dataset):
            payload: np.ndarray

    with pytest.raises(h5t.SchemaError, match="ignore.*forbid"):

        class Warn(h5t.Group, extras="warn"):  # type: ignore[arg-type]
            pass


def test_reserved_api_names_and_duplicate_names_fail() -> None:
    with pytest.raises(h5t.SchemaError, match="shadows"):

        class Reserved(h5t.Group):
            attrs: str

    with pytest.raises(h5t.SchemaError, match="duplicate HDF5"):

        class Duplicate(h5t.Group):
            first: Annotated[str, h5t.Name("same")]
            second: Annotated[int, h5t.Name("same")]


def test_removed_api_is_absent() -> None:
    for name in ("File", "Keys", "Invalid", "ValidationReport", "f8"):
        assert not hasattr(h5t, name)
    with pytest.raises(TypeError):
        h5t.Dataset[int]  # type: ignore[index]
    with pytest.raises(TypeError):
        h5t.Group[int]  # type: ignore[index]
