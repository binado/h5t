"""Tests for ``h5t.save``."""

from __future__ import annotations

import dataclasses
import enum
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Float

import h5t
from tests.conftest import (
    EagerMeasurement,
    LazyMeasurement,
    Nested,
    open_fd_count,
    write_result,
)


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


@h5t.group()
@dataclasses.dataclass
class Item:
    n: int


@h5t.group()
@dataclasses.dataclass
class Doc:
    version: int
    title: str
    color: Color
    ratio: float
    flag: bool
    note: str | None
    fisher: Float[np.ndarray, "n n"]
    nested: Nested
    items: dict[str, Item]
    cfg: Annotated[dict[str, Any], h5t.Attr(converter=json.loads, serializer=json.dumps)]
    opt_item: Item | None = None


def make_doc(**overrides: Any) -> Doc:
    values: dict[str, Any] = {
        "version": 3,
        "title": "hello",
        "color": Color.BLUE,
        "ratio": 0.5,
        "flag": True,
        "note": None,
        "fisher": np.eye(3),
        "nested": Nested(answer=7),
        "items": {"a": Item(1), "b": Item(2)},
        "cfg": {"k": [1, 2]},
    }
    values.update(overrides)
    return Doc(**values)


def assert_doc_equal(a: Doc, b: Doc) -> None:
    assert a.version == b.version
    assert a.title == b.title
    assert a.color == b.color
    assert a.ratio == b.ratio
    assert a.flag == b.flag
    assert a.note == b.note
    assert np.array_equal(a.fisher, b.fisher)
    assert a.nested == b.nested
    assert a.items == b.items
    assert a.cfg == b.cfg
    assert a.opt_item == b.opt_item


def test_round_trip(tmp_path: Path) -> None:
    doc = make_doc(opt_item=Item(9))
    path = tmp_path / "doc.h5"
    h5t.save(doc, path)
    assert_doc_equal(h5t.load(Doc, path), doc)


def test_enum_stored_as_value_and_none_omitted(tmp_path: Path) -> None:
    path = tmp_path / "doc.h5"
    h5t.save(make_doc(), path)
    with h5py.File(path) as f:
        assert f.attrs["color"] == "blue"
        assert "note" not in f.attrs
        assert "opt_item" not in f
    assert h5t.load(Doc, path).note is None


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
        nested=Nested(answer=answer),
        items={k: Item(i) for i, k in enumerate(sorted(keys))},
    )
    path = tmp_path_factory.mktemp("prop") / "doc.h5"
    h5t.save(doc, path)
    assert_doc_equal(h5t.load(Doc, path), doc)


def test_dataset_records_and_lazy_payloads(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Holder:
        eager: EagerMeasurement
        lazy: Annotated[LazyMeasurement, h5t.Payload("data")]
        samples: h5t.LazyArray

    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        for name in ("eager", "lazy", "samples"):
            ds = f.create_dataset(name, data=np.arange(4))
            if name != "samples":
                ds.attrs["unit"] = "m"
    holder = h5t.load(Holder, src)
    assert holder.lazy.data._h5t_data is not None
    dst = tmp_path / "dst.h5"
    h5t.save(holder, dst)
    again = h5t.load(Holder, dst)
    assert np.array_equal(again.lazy.data.read(), np.arange(4))
    assert np.array_equal(again.samples.read(), np.arange(4))
    assert again.eager.unit == "m"


def test_load_save_load_result_file(tmp_path: Path) -> None:
    @h5t.group(attrs="snapshot")
    @dataclasses.dataclass
    class Snap:
        version: int
        values: np.ndarray
        measurement: Annotated[LazyMeasurement, h5t.Payload("data")]
        snapshot: Mapping[str, Any]

    src = tmp_path / "result.h5"
    write_result(src)
    first = h5t.load(Snap, src)
    assert first.snapshot["undeclared"] == "raw"

    # Same file, different root, append mode: lazy payloads are read before opening.
    h5t.save(first, src, root="/copy", mode="a")
    second = h5t.load(Snap, src, root="/copy")
    assert second.snapshot["undeclared"] == "raw"
    assert np.array_equal(second.measurement.data.read(), np.arange(5))
    assert np.array_equal(second.values, np.arange(4))

    dst = tmp_path / "dst.h5"
    h5t.save(first, dst)
    assert h5t.load(Snap, dst).version == 2


def test_snapshot_forbid_rejects_undeclared(tmp_path: Path) -> None:
    @h5t.group(attrs="snap", extras="forbid")
    @dataclasses.dataclass
    class Strict:
        a: int
        snap: Mapping[str, Any]

    ok = Strict(a=1, snap={"a": 1})
    h5t.save(ok, tmp_path / "ok.h5")
    with pytest.raises(h5t.ValidationError, match="unexpected attribute") as info:
        h5t.save(Strict(a=1, snap={"zzz": 1}), tmp_path / "bad.h5")
    assert info.value.path == "/@zzz"
    assert not (tmp_path / "bad.h5").exists()


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"version": "nope"}, "/@version"),
        ({"fisher": np.ones((2, 3))}, "/fisher"),
        ({"fisher": np.ones((2, 2), dtype=int)}, "/fisher"),
        ({"items": {"a/b": Item(1)}}, "/items/a/b"),
        ({"items": {"": Item(1)}}, "/items/"),
        ({"items": {"a": Nested(answer=1)}}, "/items/a"),
        ({"nested": None}, "/nested"),
    ],
)
def test_failures_leave_no_file(tmp_path: Path, overrides: dict[str, Any], path: str) -> None:
    target = tmp_path / "doc.h5"
    with pytest.raises(h5t.ValidationError) as info:
        h5t.save(make_doc(**overrides), target)
    assert info.value.path == path
    assert not target.exists()


def test_failures_leave_existing_file_bytes_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    h5t.save(make_doc(), target)
    before = target.read_bytes()
    with pytest.raises(h5t.ValidationError):
        h5t.save(make_doc(fisher=np.ones((2, 3))), target, root="/other", mode="a")
    assert target.read_bytes() == before


def test_unstorable_attribute_requires_serializer(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Raw:
        config: Annotated[dict[str, Any], h5t.Attr()]

    with pytest.raises(h5t.ValidationError, match="serializer") as info:
        h5t.save(Raw(config={"a": 1}), tmp_path / "raw.h5")
    assert info.value.path == "/@config"


def test_glob_mismatch_rejected(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Globbed:
        snaps: Annotated[dict[str, Item], h5t.Glob("snap_*")]

    h5t.save(Globbed(snaps={"snap_1": Item(1)}), tmp_path / "ok.h5")
    with pytest.raises(h5t.ValidationError, match="Glob") as info:
        h5t.save(Globbed(snaps={"other": Item(1)}), tmp_path / "bad.h5")
    assert info.value.path == "/snaps/other"


def test_apply_failure_cleans_up(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Strings:
        names: np.ndarray

    bad = Strings(names=np.array(["a", "b"]))  # unicode arrays are not HDF5-writable
    target = tmp_path / "s.h5"
    before = open_fd_count()
    with pytest.raises(h5t.ValidationError) as info:
        h5t.save(bad, target)
    assert info.value.path == "/names"
    assert not target.exists()
    assert open_fd_count() == before

    h5t.save(make_doc(), target)
    snapshot = target.read_bytes()
    with pytest.raises(h5t.ValidationError):
        h5t.save(bad, target, root="/a/b", mode="a")
    with h5py.File(target) as f:
        assert "a" not in f  # the groups save created were removed
    assert h5t.load(Doc, target).version == 3
    assert snapshot  # the original file still opens and loads


def test_modes(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    h5t.save(make_doc(), target)
    with pytest.raises(OSError):
        h5t.save(make_doc(), target)
    with pytest.raises(OSError):
        h5t.save(make_doc(), target, mode="x")
    h5t.save(make_doc(version=5), target, mode="w")
    assert h5t.load(Doc, target).version == 5
    with pytest.raises(ValueError, match="mode"):
        h5t.save(make_doc(), target, mode="r")  # ty: ignore[invalid-argument-type]


def test_save_nested_root_creates_intermediate_groups(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    h5t.save(make_doc(), target, root="/a/b")
    assert h5t.load(Doc, target, root="/a/b").version == 3


def test_save_nonempty_root_without_overwrite_raises(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    h5t.save(make_doc(), target, root="/a/b")
    with pytest.raises(h5t.ValidationError, match="not empty") as info:
        h5t.save(make_doc(), target, root="/a/b", mode="a")
    assert info.value.path == "/a/b"


def test_save_nonempty_root_with_overwrite_replaces(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    h5t.save(make_doc(), target, root="/a/b")
    h5t.save(make_doc(version=8), target, root="/a/b", mode="a", overwrite=True)
    assert h5t.load(Doc, target, root="/a/b").version == 8


def test_save_file_root_nonempty_without_overwrite_raises(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    h5t.save(make_doc(), target)
    with pytest.raises(h5t.ValidationError, match="not empty"):
        h5t.save(make_doc(), target, mode="a")


def test_save_file_root_overwrite_clears_old_children(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    h5t.save(make_doc(), target, root="/a/b")
    h5t.save(make_doc(version=10), target, mode="a", overwrite=True)
    assert h5t.load(Doc, target).version == 10
    with h5py.File(target) as f:
        assert "a" not in f


def test_save_relative_root_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        h5t.save(make_doc(), tmp_path / "x.h5", root="rel")


def test_root_occupied_by_dataset(tmp_path: Path) -> None:
    target = tmp_path / "doc.h5"
    with h5py.File(target, "w") as f:
        f.create_dataset("d", data=1)
    with pytest.raises(h5t.ValidationError, match="found a dataset"):
        h5t.save(make_doc(), target, root="/d", mode="a")


def test_schema_errors(tmp_path: Path) -> None:
    with pytest.raises(h5t.SchemaError, match="dataset record"):
        h5t.save(EagerMeasurement("m", np.arange(2)), tmp_path / "a.h5")

    class Plain:
        pass

    with pytest.raises(h5t.SchemaError, match="not a decorated"):
        h5t.save(Plain(), tmp_path / "b.h5")

    with pytest.raises(h5t.SchemaError, match="serializer must be callable"):

        @h5t.group()
        @dataclasses.dataclass
        class BadSerializer:
            x: Annotated[int, h5t.Attr(serializer=3)]  # ty: ignore[invalid-argument-type]


def test_serializer_failure_is_conversion_error(tmp_path: Path) -> None:
    def boom(_: Any) -> Any:
        raise RuntimeError("no")

    @h5t.group()
    @dataclasses.dataclass
    class Boom:
        x: Annotated[int, h5t.Attr(serializer=boom)]

    with pytest.raises(h5t.ConversionError, match="serializer failed") as info:
        h5t.save(Boom(x=1), tmp_path / "boom.h5")
    assert info.value.path == "/@x"


def test_no_fd_leak(tmp_path: Path) -> None:
    before = open_fd_count()
    h5t.save(make_doc(), tmp_path / "ok.h5")
    with pytest.raises(h5t.ValidationError):
        h5t.save(make_doc(fisher=np.ones((2, 3))), tmp_path / "bad.h5")
    h5t.load(Doc, tmp_path / "ok.h5")
    assert open_fd_count() == before
