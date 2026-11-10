"""Tests for ``Group.model_dump_hdf5``."""

from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Float
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, computed_field

import h5t
from tests.conftest import (
    MeasurementAttrs,
    Nested,
    NestedAttrs,
    open_fd_count,
    write_result,
)


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class ItemAttrs(BaseModel):
    n: int


class Item(h5t.Group[ItemAttrs]):
    pass


class DocAttrs(BaseModel):
    version: int
    title: str
    color: Color
    ratio: float
    flag: bool
    note: str | None = None
    cfg: Annotated[
        dict[str, Any],
        BeforeValidator(json.loads),
        PlainSerializer(json.dumps),
    ]


class Doc(h5t.Group[DocAttrs]):
    fisher: Float[np.ndarray, "n n"]
    nested: Nested
    items: dict[str, Item]
    opt_item: Item | None = None


def make_doc(**overrides: Any) -> Doc:
    attr_keys = {"version", "title", "color", "ratio", "flag", "note", "cfg"}
    attr_values: dict[str, Any] = {
        "version": 3,
        "title": "hello",
        "color": Color.BLUE,
        "ratio": 0.5,
        "flag": True,
        "note": None,
        "cfg": {"k": [1, 2]},
    }
    group_values: dict[str, Any] = {
        "fisher": np.eye(3),
        "nested": Nested(attrs=NestedAttrs(answer=7)),
        "items": {"a": Item(attrs=ItemAttrs(n=1)), "b": Item(attrs=ItemAttrs(n=2))},
        "opt_item": None,
    }
    for k, v in overrides.items():
        if k in attr_keys:
            attr_values[k] = v
        else:
            group_values[k] = v
    doc = Doc.model_construct(
        attrs=DocAttrs.model_construct(**attr_values),
        **group_values,
    )
    return doc


def assert_doc_equal(a: Doc, b: Doc) -> None:
    assert a.attrs.version == b.attrs.version
    assert a.attrs.title == b.attrs.title
    assert a.attrs.color == b.attrs.color
    assert a.attrs.ratio == b.attrs.ratio
    assert a.attrs.flag == b.attrs.flag
    assert a.attrs.note == b.attrs.note
    assert a.attrs.cfg == b.attrs.cfg
    assert np.array_equal(a.fisher, b.fisher)
    assert a.nested == b.nested
    assert a.items == b.items
    assert a.opt_item == b.opt_item


def test_round_trip(tmp_path: Path) -> None:
    doc = make_doc(opt_item=Item(attrs=ItemAttrs(n=9)))
    path = tmp_path / "doc.h5"
    doc.model_dump_hdf5(path)
    assert_doc_equal(Doc.model_validate_hdf5(path), doc)


def test_enum_stored_as_value_and_none_omitted(tmp_path: Path) -> None:
    path = tmp_path / "doc.h5"
    make_doc().model_dump_hdf5(path)
    with h5py.File(path) as f:
        assert f.attrs["color"] == "blue"
        assert "note" not in f.attrs
        assert "opt_item" not in f
    assert Doc.model_validate_hdf5(path).attrs.note is None


@settings(max_examples=25, deadline=None)
@given(
    version=st.integers(-(2**31), 2**31),
    title=st.text(alphabet=st.characters(blacklist_characters="\x00", codec="utf-8"), max_size=10),
    ratio=st.floats(allow_nan=False, allow_infinity=False),
    flag=st.booleans(),
    note=st.none() | st.text(alphabet="abc", max_size=5),
    size=st.integers(1, 4),
    keys=st.sets(st.text(alphabet="abcxyz", min_size=1, max_size=4), max_size=4),
    answer=st.integers(-100, 100),
)
def test_round_trip_property(
    tmp_path_factory: pytest.TempPathFactory,
    version: int,
    title: str,
    ratio: float,
    flag: bool,
    note: str | None,
    size: int,
    keys: set[str],
    answer: int,
) -> None:
    doc = make_doc(
        version=version,
        title=title,
        ratio=ratio,
        flag=flag,
        note=note,
        fisher=np.ones((size, size)),
        nested=Nested(attrs=NestedAttrs(answer=answer)),
        items={k: Item(attrs=ItemAttrs(n=i)) for i, k in enumerate(sorted(keys))},
    )
    path = tmp_path_factory.mktemp("prop") / "doc.h5"
    doc.model_dump_hdf5(path)
    assert_doc_equal(Doc.model_validate_hdf5(path), doc)


def test_datasets_and_lazy_datasets(tmp_path: Path) -> None:
    class Holder(h5t.Group):
        typed: h5t.Dataset[MeasurementAttrs]
        samples: h5t.Dataset

    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        for name in ("typed", "samples"):
            ds = f.create_dataset(name, data=np.arange(4))
            if name != "samples":
                ds.attrs["unit"] = "m"
    holder = Holder.model_validate_hdf5(src)
    dst = tmp_path / "dst.h5"
    holder.model_dump_hdf5(dst)
    again = Holder.model_validate_hdf5(dst)
    assert np.array_equal(again.typed.data, np.arange(4))
    assert np.array_equal(again.samples.data, np.arange(4))
    assert again.typed.attrs.unit == "m"


def test_load_save_load_result_file(tmp_path: Path) -> None:
    class Snap(h5t.Group):
        values: np.ndarray
        measurement: h5t.Dataset[MeasurementAttrs]

    src = tmp_path / "result.h5"
    write_result(src)
    first = Snap.model_validate_hdf5(src)
    assert first.attrs["undeclared"] == "raw"

    # Same file, different root, append mode: lazy datasets are read before opening.
    first.model_dump_hdf5(src, root="/copy", mode="a")
    second = Snap.model_validate_hdf5(src, root="/copy")
    assert second.attrs["undeclared"] == "raw"
    assert np.array_equal(second.measurement.data, np.arange(5))
    assert np.array_equal(second.values, np.arange(4))

    dst = tmp_path / "dst.h5"
    first.model_dump_hdf5(dst)
    assert Snap.model_validate_hdf5(dst).attrs["version"] == 2


def test_untyped_forbid_rejects_undeclared_attrs_on_dump(tmp_path: Path) -> None:
    class Strict(h5t.Group):
        model_config = ConfigDict(extra="forbid")

    ok = Strict()
    ok.model_dump_hdf5(tmp_path / "ok.h5")
    bad = Strict.model_construct(attrs={"zzz": 1})
    with pytest.raises(h5t.ValidationError, match="unexpected attribute") as info:
        bad.model_dump_hdf5(tmp_path / "bad.h5")
    assert info.value.path == "/@zzz"
    assert not (tmp_path / "bad.h5").exists()


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"items": {"a/b": Item(attrs=ItemAttrs(n=1))}}, "/items/a/b"),
        ({"items": {"": Item(attrs=ItemAttrs(n=1))}}, "/items/"),
        ({"items": {".": Item(attrs=ItemAttrs(n=1))}}, "/items/."),
    ],
)
def test_dump_of_invalid_collection_key_raises_and_leaves_no_file(
    tmp_path: Path, overrides: dict[str, Any], path: str
) -> None:
    target = tmp_path / "doc.h5"
    with pytest.raises(h5t.ValidationError) as info:
        make_doc(**overrides).model_dump_hdf5(target)
    assert info.value.path == path
    assert not target.exists()


def test_failures_leave_existing_file_bytes_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    make_doc().model_dump_hdf5(target)
    before = target.read_bytes()
    with pytest.raises(h5t.ValidationError):
        make_doc(items={"a/b": Item(attrs=ItemAttrs(n=1))}).model_dump_hdf5(
            target, root="/other", mode="a"
        )
    assert target.read_bytes() == before


def test_unstorable_attribute_requires_serializer(tmp_path: Path) -> None:
    class RawAttrs(BaseModel):
        config: dict[str, Any]

    class Raw(h5t.Group[RawAttrs]):
        pass

    with pytest.raises(h5t.ValidationError, match="PlainSerializer") as info:
        Raw(attrs=RawAttrs(config={"a": 1})).model_dump_hdf5(tmp_path / "raw.h5")
    assert info.value.path == "/@config"


def test_apply_failure_closes_file(tmp_path: Path) -> None:
    class Strings(h5t.Group):
        names: np.ndarray

    bad = Strings(names=np.array(["a", "b"]))  # unicode arrays are not HDF5-writable
    target = tmp_path / "s.h5"
    before = open_fd_count()
    with pytest.raises(h5t.ValidationError) as info:
        bad.model_dump_hdf5(target)
    assert info.value.path == "/names"
    assert open_fd_count() == before


def test_modes(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    make_doc().model_dump_hdf5(target)
    with pytest.raises(OSError):
        make_doc().model_dump_hdf5(target)
    with pytest.raises(OSError):
        make_doc().model_dump_hdf5(target, mode="x")
    make_doc(version=5).model_dump_hdf5(target, mode="w")
    assert Doc.model_validate_hdf5(target).attrs.version == 5
    with pytest.raises(ValueError, match="mode"):
        make_doc().model_dump_hdf5(target, mode="r")  # ty: ignore[invalid-argument-type]


def test_save_nested_root_creates_intermediate_groups(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    make_doc().model_dump_hdf5(target, root="/a/b")
    assert Doc.model_validate_hdf5(target, root="/a/b").attrs.version == 3


def test_save_nonempty_root_without_overwrite_raises(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    make_doc().model_dump_hdf5(target, root="/a/b")
    with pytest.raises(h5t.ValidationError, match="not empty") as info:
        make_doc().model_dump_hdf5(target, root="/a/b", mode="a")
    assert info.value.path == "/a/b"


def test_save_nonempty_root_with_overwrite_replaces(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    make_doc().model_dump_hdf5(target, root="/a/b")
    make_doc(version=8).model_dump_hdf5(target, root="/a/b", mode="a", overwrite=True)
    assert Doc.model_validate_hdf5(target, root="/a/b").attrs.version == 8


def test_save_file_root_nonempty_without_overwrite_raises(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    make_doc().model_dump_hdf5(target)
    with pytest.raises(h5t.ValidationError, match="not empty"):
        make_doc().model_dump_hdf5(target, mode="a")


def test_save_file_root_overwrite_clears_old_children(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    make_doc().model_dump_hdf5(target, root="/a/b")
    make_doc(version=10).model_dump_hdf5(target, mode="a", overwrite=True)
    assert Doc.model_validate_hdf5(target).attrs.version == 10
    with h5py.File(target) as f:
        assert "a" not in f


def test_save_relative_root_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        make_doc().model_dump_hdf5(tmp_path / "x.h5", root="rel")


def test_root_occupied_by_dataset(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    with h5py.File(target, "w") as f:
        f.create_dataset("d", data=1)
    with pytest.raises(h5t.ValidationError, match="found a dataset"):
        make_doc().model_dump_hdf5(target, root="/d", mode="a")


def test_serializer_failure_is_validation_error(tmp_path: Path) -> None:
    def boom(_: Any) -> Any:
        raise RuntimeError("no")

    class BoomAttrs(BaseModel):
        x: Annotated[int, PlainSerializer(boom)]

    class Boom(h5t.Group[BoomAttrs]):
        pass

    with pytest.raises(h5t.ValidationError, match="serializer failed") as info:
        Boom(attrs=BoomAttrs(x=1)).model_dump_hdf5(tmp_path / "boom.h5")
    assert info.value.path == "/"


def test_no_fd_leak(tmp_path: Path) -> None:
    before = open_fd_count()
    make_doc().model_dump_hdf5(tmp_path / "ok.h5")
    with pytest.raises(h5t.ValidationError):
        make_doc(items={"a/b": Item(attrs=ItemAttrs(n=1))}).model_dump_hdf5(tmp_path / "bad.h5")
    Doc.model_validate_hdf5(tmp_path / "ok.h5")
    assert open_fd_count() == before


class _TypedDatasetGroup(h5t.Group):
    measurement: h5t.Dataset[MeasurementAttrs]


def test_dump_of_dataset_whose_source_file_was_deleted_leaves_no_file(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as file:
        file.create_dataset("measurement", data=np.arange(3)).attrs["unit"] = "m"
    loaded = _TypedDatasetGroup.model_validate_hdf5(source)
    source.unlink()
    target = tmp_path / "copy.h5"
    with pytest.raises(h5t.ValidationError, match="cannot read dataset data") as caught:
        loaded.model_dump_hdf5(target)
    assert caught.value.path == "/measurement"
    assert not target.exists()


@pytest.mark.parametrize("value", ["a\x00b", np.zeros(70_000)], ids=["embedded-nul", "too-large"])
def test_dump_attr_h5py_rejects_reports_attribute_path(tmp_path: Path, value: object) -> None:
    class Attrs(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        v: Any

    target = tmp_path / "doc.h5"
    with pytest.raises(h5t.ValidationError, match="cannot write attribute") as info:
        h5t.Group[Attrs](attrs=Attrs(v=value)).model_dump_hdf5(target)
    assert info.value.path == "/@v"


def test_dump_required_any_attr_with_none_value_reports_attribute_path(tmp_path: Path) -> None:
    class Attrs(BaseModel):
        v: Any

    target = tmp_path / "doc.h5"
    with pytest.raises(h5t.ValidationError, match="cannot store NoneType") as info:
        h5t.Group[Attrs](attrs=Attrs(v=None)).model_dump_hdf5(target)
    assert info.value.path == "/@v"


def test_dump_attr_alias_is_stored_under_the_on_disk_name(tmp_path: Path) -> None:
    class Attrs(BaseModel):
        renamed: str = Field(alias="on-disk")

    target = tmp_path / "doc.h5"
    h5t.Group[Attrs](attrs=Attrs(**{"on-disk": "x"})).model_dump_hdf5(target)
    with h5py.File(target) as f:
        assert list(f.attrs) == ["on-disk"]


def test_dump_stores_declared_attrs_only_not_computed_fields(tmp_path: Path) -> None:
    class Attrs(BaseModel):
        n: int

        @computed_field
        @property
        def doubled(self) -> int:
            return 2 * self.n

    target = tmp_path / "doc.h5"
    h5t.Group[Attrs](attrs=Attrs(n=2)).model_dump_hdf5(target)
    with h5py.File(target) as f:
        assert list(f.attrs) == ["n"]


@pytest.mark.parametrize("collection", [False, True])
def test_runtime_group_subclass_fields_are_written(tmp_path: Path, collection: bool) -> None:
    class Base(h5t.Group):
        values: np.ndarray

    class Sub(Base):
        extra_values: np.ndarray = Field(alias="extra-values")

    annotation = dict[str, Base] if collection else Base

    class Outer(h5t.Group):
        child: annotation

    child = Sub(values=np.arange(2), extra_values=np.arange(3))
    value = Outer(child={"one": child} if collection else child)
    path = tmp_path / "subclass.h5"
    value.model_dump_hdf5(path)
    with h5py.File(path) as file:
        node = file["child/one" if collection else "child"]
        np.testing.assert_array_equal(node["extra-values"][()], np.arange(3))


def test_runtime_attrs_subclass_optional_alias_and_serializer(tmp_path: Path) -> None:
    from pydantic import computed_field, field_serializer

    class BaseAttrs(BaseModel):
        version: int

    class SubAttrs(BaseAttrs):
        omitted: int | None = Field(default=None, serialization_alias="missing")
        label: int = Field(serialization_alias="label-on-disk")

        @field_serializer("label")
        def label_string(self, value: int) -> str:
            return f"v{value}"

        @computed_field
        @property
        def derived(self) -> int:
            return 1

    class Owner(h5t.Group[BaseAttrs]):
        dataset: h5t.Dataset[BaseAttrs]

    attrs = SubAttrs(version=1, label=2)
    value = Owner(attrs=attrs, dataset=h5t.Dataset[BaseAttrs](np.zeros(2), attrs))
    path = tmp_path / "attrs-subclass.h5"
    value.model_dump_hdf5(path)
    with h5py.File(path) as file:
        for node in (file, file["dataset"]):
            assert dict(node.attrs) == {"version": 1, "label-on-disk": "v2"}


def test_invalid_runtime_subclass_leaves_existing_file_unchanged(tmp_path: Path) -> None:
    class Base(h5t.Group):
        pass

    class Bad(Base):
        invalid: int

    class Outer(h5t.Group):
        child: Base

    path = tmp_path / "existing.h5"
    with h5py.File(path, "w") as file:
        file["original"] = np.arange(3)
    before = path.read_bytes()
    with pytest.raises(h5t.SchemaError, match="Bad.invalid"):
        Outer(child=Bad(invalid=1)).model_dump_hdf5(path, mode="w")
    assert path.read_bytes() == before


def test_invalid_runtime_attrs_leave_existing_file_unchanged(tmp_path: Path) -> None:
    class BaseAttrs(BaseModel):
        version: int

    class BadAttrs(BaseAttrs):
        first: int = Field(alias="same")
        second: int = Field(alias="same")

    class Owner(h5t.Group[BaseAttrs]):
        pass

    value = Owner(attrs=BadAttrs(version=1, same=2))
    path = tmp_path / "unchanged.h5"
    path.write_bytes(b"untouched")
    with pytest.raises(h5t.SchemaError, match="duplicate HDF5 attribute"):
        value.model_dump_hdf5(path, mode="w")
    assert path.read_bytes() == b"untouched"
