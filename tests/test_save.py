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
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer

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
    assert np.array_equal(again.typed.read(), np.arange(4))
    assert np.array_equal(again.samples.read(), np.arange(4))
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
    assert np.array_equal(second.measurement.read(), np.arange(5))
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
        ({"version": "nope"}, "/@version"),
        ({"fisher": np.ones((2, 3))}, "/fisher"),
        ({"fisher": np.ones((2, 2), dtype=int)}, "/fisher"),
        ({"items": {"a/b": Item(attrs=ItemAttrs(n=1))}}, "/items/a/b"),
        ({"items": {"": Item(attrs=ItemAttrs(n=1))}}, "/items/"),
        ({"items": {"a": Nested(attrs=NestedAttrs(answer=1))}}, "/items/a"),
        ({"nested": None}, "/nested"),
    ],
)
def test_failures_leave_no_file(tmp_path: Path, overrides: dict[str, Any], path: str) -> None:
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
        make_doc(fisher=np.ones((2, 3))).model_dump_hdf5(target, root="/other", mode="a")
    assert target.read_bytes() == before


def test_unstorable_attribute_requires_serializer(tmp_path: Path) -> None:
    class RawAttrs(BaseModel):
        config: dict[str, Any]

    class Raw(h5t.Group[RawAttrs]):
        pass

    with pytest.raises(h5t.ValidationError, match="PlainSerializer") as info:
        Raw(attrs=RawAttrs(config={"a": 1})).model_dump_hdf5(tmp_path / "raw.h5")
    assert info.value.path == "/@config"


def test_glob_mismatch_rejected(tmp_path: Path) -> None:
    class Globbed(h5t.Group):
        snaps: Annotated[dict[str, Item], h5t.Glob("snap_*")]

    Globbed(snaps={"snap_1": Item(attrs=ItemAttrs(n=1))}).model_dump_hdf5(tmp_path / "ok.h5")
    with pytest.raises(h5t.ValidationError, match="Glob") as info:
        Globbed(snaps={"other": Item(attrs=ItemAttrs(n=1))}).model_dump_hdf5(tmp_path / "bad.h5")
    assert info.value.path == "/snaps/other"


def test_apply_failure_cleans_up(tmp_path: Path) -> None:
    class Strings(h5t.Group):
        names: np.ndarray

    bad = Strings(names=np.array(["a", "b"]))  # unicode arrays are not HDF5-writable
    target = tmp_path / "s.h5"
    before = open_fd_count()
    with pytest.raises(h5t.ValidationError) as info:
        bad.model_dump_hdf5(target)
    assert info.value.path == "/names"
    assert not target.exists()
    assert open_fd_count() == before

    make_doc().model_dump_hdf5(target)
    snapshot = target.read_bytes()
    with pytest.raises(h5t.ValidationError):
        bad.model_dump_hdf5(target, root="/a/b", mode="a")
    with h5py.File(target) as f:
        assert "a" not in f
    assert Doc.model_validate_hdf5(target).attrs.version == 3
    assert snapshot


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
    assert info.value.path == "/@x"


def test_no_fd_leak(tmp_path: Path) -> None:
    before = open_fd_count()
    make_doc().model_dump_hdf5(tmp_path / "ok.h5")
    with pytest.raises(h5t.ValidationError):
        make_doc(fisher=np.ones((2, 3))).model_dump_hdf5(tmp_path / "bad.h5")
    Doc.model_validate_hdf5(tmp_path / "ok.h5")
    assert open_fd_count() == before


class _Names(h5t.Group):
    names: np.ndarray


@pytest.fixture
def unwritable_names() -> _Names:
    """A valid plan whose apply phase fails: unicode arrays are not HDF5-writable."""
    return _Names(names=np.array(["a", "b"]))


@pytest.mark.parametrize("root", ["/", "/a/b"])
def test_failed_overwrite_is_destructive_like_h5py(
    tmp_path: Path, root: str, unwritable_names: _Names
) -> None:
    target = tmp_path / "doc.h5"
    _Names(names=np.arange(3)).model_dump_hdf5(target, root=root)
    with h5py.File(target, "a") as f:
        f[root].attrs["marker"] = "old"
    with pytest.raises(h5t.ValidationError) as info:
        unwritable_names.model_dump_hdf5(target, root=root, mode="a", overwrite=True)
    assert info.value.path == f"{root.rstrip('/')}/names"
    with h5py.File(target) as f:
        assert list(f[root].keys()) == []
        assert list(f[root].attrs) == []


def test_rollback_failure_is_attached_to_the_original_error(
    tmp_path: Path, unwritable_names: _Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    from h5t import _write

    def broken_rollback(self: _write._Undo, file: h5py.File) -> None:
        raise KeyError("cannot undo")

    target = tmp_path / "doc.h5"
    make_doc().model_dump_hdf5(target)
    monkeypatch.setattr(_write._Undo, "rollback", broken_rollback)
    with pytest.raises(h5t.ValidationError) as info:
        unwritable_names.model_dump_hdf5(target, root="/a/b", mode="a")
    assert any("rollback failed" in note for note in info.value.__notes__)


def test_partial_file_removal_failure_is_attached_to_the_original_error(
    tmp_path: Path, unwritable_names: _Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_remove(path: object) -> None:
        raise PermissionError("read-only directory")

    monkeypatch.setattr("os.remove", broken_remove)
    with pytest.raises(h5t.ValidationError) as info:
        unwritable_names.model_dump_hdf5(tmp_path / "doc.h5")
    assert any("partially written file" in note for note in info.value.__notes__)


class _TypedDatasetGroup(h5t.Group):
    measurement: h5t.Dataset[MeasurementAttrs]


class _OtherDatasetAttrs(BaseModel):
    note: str = "n"


@pytest.mark.parametrize(
    "value, match",
    [
        (np.arange(3), "instance of Dataset"),
        (h5t.Dataset(np.arange(3), _OtherDatasetAttrs()), "MeasurementAttrs"),
    ],
    ids=["non-dataset", "wrong-specialization"],
)
def test_dump_rejects_bad_dataset_member(tmp_path: Path, value: Any, match: str) -> None:
    group = _TypedDatasetGroup.model_construct(measurement=value)
    target = tmp_path / "bad.h5"
    with pytest.raises(h5t.ValidationError, match=match) as caught:
        group.model_dump_hdf5(target)
    assert caught.value.path.startswith("/measurement")
    assert not target.exists()


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
