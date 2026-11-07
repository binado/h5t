"""Group declarations, Pydantic rebuilding, and observable HDF5 behavior."""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

import h5t
from h5t._compile import _read_array


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
        namespace["Invalid"].model_rebuild(_types_namespace=namespace)
        namespace["Invalid"].model_validate_hdf5("never-opened.h5")


def test_invalid_extra_config_raises_schema_error() -> None:
    with pytest.raises(h5t.SchemaError, match="ignore.*forbid"):

        class Warn(h5t.Group):
            model_config = ConfigDict(extra="allow")

        Warn.model_validate_hdf5("never-opened.h5")


def test_duplicate_hdf5_names_fail() -> None:
    with pytest.raises(h5t.SchemaError, match="duplicate HDF5"):

        class Duplicate(h5t.Group):
            first: np.ndarray = Field(alias="same")
            second: h5t.Dataset = Field(alias="same")

        Duplicate.model_validate_hdf5("never-opened.h5")


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


T_attrs = typing.TypeVar("T_attrs", bound=BaseModel)


class _StrictAttrs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: int


class _GenericBase(h5t.Group[T_attrs], typing.Generic[T_attrs]):
    pass


class _StrictSubclass(_GenericBase[_StrictAttrs]):
    pass


def test_generic_group_subclass_enforces_attrs_extra_forbid(tmp_path: Path) -> None:
    path = tmp_path / "generic.h5"
    with h5py.File(path, "w") as file:
        file.attrs["answer"] = 7
        file.attrs["surprise"] = 1
    with pytest.raises(h5t.ValidationError) as caught:
        _StrictSubclass.model_validate_hdf5(path)
    assert caught.value.path == "/@surprise"


class _OpenDatasetGeneric(h5t.Group[T_attrs], typing.Generic[T_attrs]):
    ds: h5t.Dataset[T_attrs]


class _OpenDatasetAttrs(BaseModel):
    answer: int


def test_generic_group_with_open_typevar_dataset_compiles_and_loads(tmp_path: Path) -> None:
    path = tmp_path / "open-typevar.h5"
    with h5py.File(path, "w") as file:
        file.attrs["answer"] = 1
        file.create_dataset("ds", data=np.arange(3)).attrs["answer"] = 2
    loaded = _OpenDatasetGeneric[_OpenDatasetAttrs].model_validate_hdf5(path)
    assert loaded.ds.shape == (3,)
    assert loaded.ds.attrs.answer == 2


def test_plain_array_fields_are_read_from_raw_datasets(tmp_path: Path) -> None:
    class Plain(h5t.Group):
        values: np.ndarray
        runs: Annotated[dict[str, np.ndarray], h5t.Glob("r*")]

    class Sub(Plain):
        pass

    class Attrs(BaseModel):
        n: int

    class Generic(h5t.Group[Attrs]):
        values: np.ndarray

    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as f:
        f.attrs["n"] = 1
        f["values"] = np.arange(3)
        f.create_group("runs")["r1"] = np.ones(2)
    for cls in (Plain, Sub, Generic):
        loaded = cls.model_validate_hdf5(path)
        assert isinstance(loaded.values, np.ndarray)
        assert loaded.values.tolist() == [0, 1, 2]
    assert isinstance(Plain.model_validate_hdf5(path).runs["r1"], np.ndarray)
    assert _read_array(np.zeros(2)).shape == (2,)


def test_glob_filters_only_a_live_group_view(tmp_path: Path) -> None:
    from h5t._h5view import _GroupView

    path = tmp_path / "g.h5"
    with h5py.File(path, "w") as f:
        for name in ("a1", "a2", "b1"):
            f[name] = np.zeros(1)
    adapter = TypeAdapter(Annotated[dict[str, Any], h5t.Glob("a*")])
    with h5py.File(path, "r") as f:
        assert sorted(adapter.validate_python(_GroupView(f))) == ["a1", "a2"]
    assert sorted(adapter.validate_python({"a1": 1, "b1": 2})) == ["a1", "b1"]


def test_explicit_namespace_rebuild_and_self_reference(tmp_path: Path) -> None:
    def declare():
        class Deferred(h5t.Group):
            child: LocalDependency

        return Deferred

    deferred = declare()

    class LocalDependency(h5t.Group):
        child: LocalDependency | None = None
        values: np.ndarray

    deferred.model_rebuild(_types_namespace={"LocalDependency": LocalDependency})
    path = tmp_path / "local.h5"
    deferred(child=LocalDependency(values=np.arange(3))).model_dump_hdf5(path)
    loaded = deferred.model_validate_hdf5(path)
    assert loaded.child.child is None
    np.testing.assert_array_equal(loaded.child.values, np.arange(3))


def test_localns_keyword_is_removed() -> None:
    with pytest.raises(TypeError):

        class Invalid(h5t.Group, localns={}):
            pass


@pytest.mark.parametrize("collection", [False, True])
def test_nested_invalid_declaration_precedes_file_access(
    tmp_path: Path, collection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Bad(h5t.Group):
        value: int

    annotation = dict[str, Bad] if collection else Bad

    class Outer(h5t.Group):
        child: annotation

    def forbidden(*args, **kwargs):
        pytest.fail("declaration checking opened a file")

    monkeypatch.setattr(h5py, "File", forbidden)
    with pytest.raises(h5t.SchemaError, match="Bad.value"):
        Outer.model_validate_hdf5(tmp_path / "absent.h5")
    child = Bad(value=1)
    value = Outer(child={"one": child} if collection else child)
    with pytest.raises(h5t.SchemaError, match="Bad.value"):
        value.model_dump_hdf5(tmp_path / "absent.h5")


def test_ndarray_aliases_roundtrip(tmp_path: Path) -> None:
    import numpy.typing as npt

    class Arrays(h5t.Group):
        data: npt.NDArray[np.float64]
        optional: npt.NDArray[Any] | None = None

    path = tmp_path / "aliases.h5"
    Arrays(data=np.zeros(3)).model_dump_hdf5(path)
    assert Arrays.model_validate_hdf5(path).data.shape == (3,)
    assert Arrays.model_validate_hdf5(path).optional is None


@pytest.mark.skipif(not hasattr(typing, "TypeAliasType"), reason="requires Python 3.12")
def test_chained_pep695_array_aliases_roundtrip(tmp_path: Path) -> None:
    import numpy.typing as npt

    Coords = typing.TypeAliasType("Coords", npt.NDArray[np.float64])
    Chained = typing.TypeAliasType("Chained", Coords)

    class Arrays(h5t.Group):
        data: Chained

    path = tmp_path / "chained.h5"
    Arrays(data=np.zeros(3)).model_dump_hdf5(path)
    assert Arrays.model_validate_hdf5(path).data.shape == (3,)
