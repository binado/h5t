"""Group compilation: field classification, deferral, and declaration errors."""

from __future__ import annotations

import gc
import sys
import typing
import weakref
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, ConfigDict, Field

import h5t
from h5t._compile import _rebuild_model
from h5t._spec import GroupSpec, MemberKind


def compiled(group_cls: type[h5t.Group[Any]]) -> GroupSpec:
    """Return ``group_cls``'s compiled spec without going through a load."""
    marker = group_cls.__dict__["__h5t_group__"]
    assert isinstance(marker, GroupSpec), "group is still pending"
    return marker


def field_map(group_cls: type[h5t.Group[Any]]) -> dict[str, Any]:
    return {field.py_name: field for field in compiled(group_cls).fields}


def test_every_field_kind_and_markers_compile() -> None:
    class ChildAttrs(BaseModel):
        value: int

    class Child(h5t.Group[ChildAttrs]):
        pass

    class TypedAttrs(BaseModel):
        unit: str

    class SchemaAttrs(BaseModel):
        renamed: str = Field(alias="on-disk")

    class Schema(h5t.Group[SchemaAttrs]):
        aliased_arr: np.ndarray = Field(alias="disk-arr")
        array: np.ndarray
        bare: h5t.Dataset
        typed: h5t.Dataset[TypedAttrs]
        child: Child

    spec = compiled(Schema)
    assert spec.attrs is not None
    assert spec.attrs.fields[0].h5_name == "on-disk"

    fields = field_map(Schema)
    assert fields["aliased_arr"].kind is MemberKind.ARRAY
    assert fields["aliased_arr"].h5_name == "disk-arr"
    assert fields["array"].kind is MemberKind.ARRAY
    assert fields["bare"].kind is MemberKind.DATASET
    assert fields["bare"].dataset_attrs is None
    assert fields["typed"].kind is MemberKind.DATASET
    assert fields["typed"].dataset_attrs is not None
    assert fields["child"].kind is MemberKind.GROUP
    assert fields["child"].member_type is Child


def test_ndarray_typing_aliases_classify_as_arrays() -> None:
    class Aliased(h5t.Group):
        payload: npt.NDArray[np.float64]
        optional_payload: npt.NDArray[Any] | None

    fields = field_map(Aliased)
    assert fields["payload"].kind is MemberKind.ARRAY
    assert fields["optional_payload"].kind is MemberKind.ARRAY
    assert fields["optional_payload"].optional


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 aliases require Python 3.12")
def test_pep695_aliases_of_ndarray_classify_as_arrays() -> None:
    Coords = typing.TypeAliasType("Coords", npt.NDArray[np.float64])
    Chained = typing.TypeAliasType("Chained", Coords)

    class Aliased(h5t.Group):
        direct: Coords
        chained: Chained

    fields = field_map(Aliased)
    assert fields["direct"].kind is MemberKind.ARRAY
    assert fields["chained"].kind is MemberKind.ARRAY
    assert fields["direct"].annotation is np.ndarray


def test_a_resolvable_group_compiles_at_class_definition() -> None:
    class Eagerly(h5t.Group):
        value: np.ndarray

    assert [field.py_name for field in compiled(Eagerly).fields] == ["value"]


def test_a_group_declared_in_a_function_compiles_before_it_returns() -> None:
    def declare() -> type[h5t.Group[Any]]:
        class PayloadAttrs(BaseModel):
            unit: str

        class Local(h5t.Group):
            payload: h5t.Dataset[PayloadAttrs]

        assert isinstance(Local.__dict__["__h5t_group__"], GroupSpec)
        return Local

    assert field_map(declare())["payload"].kind is MemberKind.DATASET


def test_eager_compilation_does_not_retain_the_defining_scope() -> None:
    class Tracked:
        pass

    def declare() -> tuple[type[h5t.Group[Any]], weakref.ref[Tracked]]:
        unrelated = Tracked()

        class Local(h5t.Group):
            value: np.ndarray

        return Local, weakref.ref(unrelated)

    record, ref = declare()
    gc.collect()
    assert ref() is None
    assert compiled(record).fields[0].py_name == "value"


def test_deferred_group_with_function_local_dependency_needs_localns_to_resolve() -> None:
    class Dependency(h5t.Group):
        value: np.ndarray

    def declare(**kwargs: Any) -> type[h5t.Group[Any]]:
        class Deferred(h5t.Group, **kwargs):
            dependency: Dependency

        return Deferred

    plain = declare()
    with_localns = declare(localns={"Dependency": Dependency})

    with pytest.raises(h5t.SchemaError, match="Dependency"):
        plain.model_validate_hdf5("never-opened.h5")
    fields = h5t._compile._compiled_group(with_localns).fields
    assert fields[0].member_type is Dependency


def test_function_local_group_can_reference_itself() -> None:
    class Node(h5t.Group):
        child: Node | None = None

    names = [f.py_name for f in h5t._compile._compiled_group(Node).fields]
    assert names == ["child"]


def test_deferred_group_retains_no_locals_the_annotations_never_name() -> None:
    class Tracked:
        pass

    def declare() -> tuple[type[h5t.Group[Any]], weakref.ref[Tracked]]:
        tracked = Tracked()
        unrelated = [tracked]
        assert unrelated[0] is tracked

        class Local(h5t.Group):
            undefined: _NeverDefined  # noqa: F821

        return Local, weakref.ref(tracked)

    record, ref = declare()
    gc.collect()
    assert record.__dict__["__h5t_group__"].localns is None
    assert ref() is None
    with pytest.raises(h5t.SchemaError, match="_NeverDefined"):
        record.model_validate_hdf5("never-opened.h5")


def test_a_broken_annotation_helper_is_not_a_forward_reference() -> None:
    def alias():
        return _typo_inside_the_helper  # noqa: F821

    with pytest.raises(h5t.SchemaError, match="_typo_inside_the_helper"):

        class Broken(h5t.Group):
            value: alias()


@pytest.mark.parametrize(
    "declaration, match",
    [
        ("value: list[int]", "unsupported child field annotation"),
        ("value: int", "unsupported child field annotation"),
        ("value: Annotated[np.ndarray, Field(alias='a/b')]", "cannot contain"),
        (
            "value: Annotated[dict[str, Child], h5t.Glob('a'), h5t.Glob('b')]",
            "may appear only once",
        ),
    ],
)
def test_invalid_declarations(declaration: str, match: str) -> None:
    class Child(h5t.Group):
        pass

    namespace = {
        "__name__": __name__,
        "h5t": h5t,
        "np": np,
        "Annotated": Annotated,
        "Field": Field,
        "Child": Child,
    }
    source = "class Invalid(h5t.Group):\n    " + declaration + "\n"
    with pytest.raises(h5t.SchemaError, match=match):
        exec(source, namespace)
    assert "Invalid" not in namespace


def test_invalid_extra_config_raises_schema_error() -> None:
    with pytest.raises(h5t.SchemaError, match="ignore.*forbid"):

        class Warn(h5t.Group):
            model_config = ConfigDict(extra="allow")


def test_duplicate_hdf5_names_fail() -> None:
    with pytest.raises(h5t.SchemaError, match="duplicate HDF5"):

        class Duplicate(h5t.Group):
            first: np.ndarray = Field(alias="same")
            second: h5t.Dataset = Field(alias="same")


def test_removed_api_is_absent() -> None:
    removed = (
        "group",
        "dataset",
        "load",
        "save",
        "LazyArray",
        "Name",
        "Attr",
        "Eager",
        "Payload",
        "ConversionError",
    )
    for name in removed:
        assert not hasattr(h5t, name)
        assert name not in h5t.__all__


class _Plain(BaseModel):
    value: int = 0


def test_rebuild_model_unresolved_annotation_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import PydanticUndefinedAnnotation

    def undefined(*args: object, **kwargs: object) -> None:
        raise PydanticUndefinedAnnotation("Later", "name 'Later' is not defined")

    monkeypatch.setattr(_Plain, "model_rebuild", undefined)
    _rebuild_model(_Plain, {})


def test_rebuild_model_schema_failure_raises_schema_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import PydanticUserError

    def broken(*args: object, **kwargs: object) -> None:
        raise PydanticUserError("bad schema", code=None)

    monkeypatch.setattr(_Plain, "model_rebuild", broken)
    with pytest.raises(h5t.SchemaError, match="_Plain.*bad schema") as caught:
        _rebuild_model(_Plain, {})
    assert isinstance(caught.value.__cause__, PydanticUserError)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, RuntimeError])
def test_rebuild_model_unrelated_exceptions_propagate(
    monkeypatch: pytest.MonkeyPatch, exc_type: type[BaseException]
) -> None:
    def interrupted(*args: object, **kwargs: object) -> None:
        raise exc_type()

    monkeypatch.setattr(_Plain, "model_rebuild", interrupted)
    with pytest.raises(exc_type):
        _rebuild_model(_Plain, {})
