"""Dynamic collections: ``dict[str, T]`` fields and the ``h5t.Glob`` filter."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest
from jaxtyping import Float

import h5t
from h5t._array import _DATA_NOT_LOADED


@h5t.group()
@dataclasses.dataclass
class Inner:
    label: str


@h5t.group()
@dataclasses.dataclass
class Snapshot:
    time: float
    inner: Inner | None = None


@h5t.dataset(data="values")
@dataclasses.dataclass
class Series:
    unit: str
    values: np.ndarray


@dataclasses.dataclass
class Plain:
    unit: str
    values: np.ndarray


def _write(path: Path, build: Any) -> Path:
    with h5py.File(path, "w") as f:
        build(f)
    return path


def _snapshots(f: h5py.File) -> None:
    run = f.create_group("run")
    snaps = run.create_group("snapshots")
    for i in range(2):
        g = snaps.create_group(f"snap_{i}")
        g.attrs["time"] = float(i)
        g.create_group("inner").attrs["label"] = f"l{i}"


def test_group_record_items(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Run:
        snapshots: dict[str, Snapshot]

    @h5t.group()
    @dataclasses.dataclass
    class Root:
        run: Run

    out = h5t.load(Root, _write(tmp_path / "a.h5", _snapshots))
    snaps = out.run.snapshots
    assert type(snaps) is dict
    assert list(snaps) == ["snap_0", "snap_1"]
    assert snaps["snap_1"] == Snapshot(1.0, Inner("l1"))


def test_mapping_annotation_and_dataset_and_payload_items(tmp_path: Path) -> None:
    def build(f: h5py.File) -> None:
        g = f.create_group("series")
        for n in ("a", "b"):
            d = g.create_dataset(n, data=np.arange(3.0))
            d.attrs["unit"] = "m"
        f.create_group("marked")
        for n in ("a", "b"):
            d = f["marked"].create_dataset(n, data=np.arange(2.0))
            d.attrs["unit"] = "s"

    @h5t.group()
    @dataclasses.dataclass
    class Root:
        series: Mapping[str, Series]
        marked: dict[str, Annotated[Plain, h5t.Payload("values")]]

    out = h5t.load(Root, _write(tmp_path / "a.h5", build))
    assert set(out.series) == {"a", "b"}
    assert out.series["a"].unit == "m"
    assert out.marked["b"].unit == "s"


def test_lazy_array_items_read_no_payload(tmp_path: Path) -> None:
    def build(f: h5py.File) -> None:
        g = f.create_group("arrs")
        g.create_dataset("x", data=np.arange(4.0))
        g.create_dataset("y", data=np.arange(5.0))

    @h5t.group()
    @dataclasses.dataclass
    class Root:
        arrs: dict[str, h5t.LazyArray]

    @h5t.group()
    @dataclasses.dataclass
    class EagerRoot:
        arrs: dict[str, Annotated[h5t.LazyArray, h5t.Eager()]]

    path = _write(tmp_path / "a.h5", build)
    out = h5t.load(Root, path)
    assert out.arrs["x"].shape == (4,)
    assert out.arrs["x"]._h5t_data is _DATA_NOT_LOADED
    assert h5t.load(EagerRoot, path).arrs["y"]._h5t_data is not _DATA_NOT_LOADED


def test_ndarray_and_jaxtyping_items_share_scope_with_owner(tmp_path: Path) -> None:
    def build(f: h5py.File, third: int = 3) -> None:
        g = f.create_group("arrs")
        g.create_dataset("x", data=np.zeros((3, 3)))
        g.create_dataset("y", data=np.zeros((third, third)))
        f.create_dataset("own", data=np.zeros(3))
        f.create_group("plain").create_dataset("p", data=np.arange(2))

    @h5t.group()
    @dataclasses.dataclass
    class Root:
        arrs: dict[str, Float[np.ndarray, "n n"]]
        own: Float[np.ndarray, "n"]
        plain: dict[str, np.ndarray]

    out = h5t.load(Root, _write(tmp_path / "ok.h5", build))
    assert out.plain["p"].tolist() == [0, 1]

    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        build(f, third=4)
    with pytest.raises(h5t.ValidationError, match=r"/arrs/y"):
        h5t.load(Root, bad)


def test_record_items_bind_independently(tmp_path: Path) -> None:
    @h5t.dataset(data="m")
    @dataclasses.dataclass
    class Mat:
        m: Float[np.ndarray, "n n"]

    def build(f: h5py.File) -> None:
        g = f.create_group("mats")
        g.create_dataset("a", data=np.zeros((2, 2)))
        g.create_dataset("b", data=np.zeros((3, 3)))

    @h5t.group()
    @dataclasses.dataclass
    class Root:
        mats: dict[str, Mat]

    assert set(h5t.load(Root, _write(tmp_path / "a.h5", build)).mats) == {"a", "b"}


def test_glob_filters_case_sensitively_and_skips_unmatched(tmp_path: Path) -> None:
    def build(f: h5py.File) -> None:
        g = f.create_group("chains")
        for n in ("chain_1", "chain_2", "Chain_3", "other", "chain_10"):
            g.create_dataset(n, data=np.arange(2))

    @h5t.group()
    @dataclasses.dataclass
    class Root:
        chains: Annotated[dict[str, h5t.LazyArray], h5t.Glob("chain_?")]

    out = h5t.load(Root, _write(tmp_path / "a.h5", build))
    assert set(out.chains) == {"chain_1", "chain_2"}


def test_empty_group_gives_empty_dict(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Root:
        snaps: dict[str, Snapshot]

    out = h5t.load(Root, _write(tmp_path / "a.h5", lambda f: f.create_group("snaps")))
    assert out.snaps == {}


def test_name_on_collection_field(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Root:
        raw: Annotated[dict[str, h5t.LazyArray], h5t.Name("Raw Data")]

    def build(f: h5py.File) -> None:
        f.create_group("Raw Data").create_dataset("x", data=np.arange(2))

    assert list(h5t.load(Root, _write(tmp_path / "a.h5", build)).raw) == ["x"]


def test_missing_collection_group(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.h5", lambda f: f.attrs.__setitem__("v", 1))

    @h5t.group()
    @dataclasses.dataclass
    class Required:
        snaps: dict[str, Snapshot]

    with pytest.raises(h5t.ValidationError, match=r"/snaps.*required collection 'snaps'"):
        h5t.load(Required, path)

    @h5t.group()
    @dataclasses.dataclass
    class Optional_:
        snaps: dict[str, Snapshot] | None

    assert h5t.load(Optional_, path).snaps is None

    @h5t.group()
    @dataclasses.dataclass
    class Defaulted:
        snaps: dict[str, Snapshot] = dataclasses.field(default_factory=dict)

    assert h5t.load(Defaulted, path).snaps == {}


def test_dataset_at_collection_path(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Root:
        snaps: dict[str, Snapshot]

    path = _write(tmp_path / "a.h5", lambda f: f.create_dataset("snaps", data=1))
    with pytest.raises(h5t.ValidationError, match="expected a group, found a dataset"):
        h5t.load(Root, path)


def test_item_failure_reports_item_path(tmp_path: Path) -> None:
    def build(f: h5py.File) -> None:
        _snapshots(f)
        f["run/snapshots/snap_1"].attrs["time"] = "late"

    @h5t.group()
    @dataclasses.dataclass
    class Run:
        snapshots: dict[str, Snapshot]

    @h5t.group()
    @dataclasses.dataclass
    class Root:
        run: Run

    with pytest.raises(h5t.ValidationError) as info:
        h5t.load(Root, _write(tmp_path / "a.h5", build))
    assert "/run/snapshots/snap_1@time" in str(info.value)


def test_self_referential_collection_terminates(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Node:
        value: int
        children: dict[str, Node] = dataclasses.field(default_factory=dict)

    def build(f: h5py.File) -> None:
        f.attrs["value"] = 0
        c = f.create_group("children").create_group("a")
        c.attrs["value"] = 1
        c.create_group("children").create_group("b").attrs["value"] = 2

    out = h5t.load(Node, _write(tmp_path / "a.h5", build))
    assert out.children["a"].children["b"].value == 2
    assert out.children["a"].children["b"].children == {}


def test_extras_forbid_allows_collection(tmp_path: Path) -> None:
    @h5t.group(extras="forbid")
    @dataclasses.dataclass
    class Root:
        snaps: dict[str, Snapshot]

    out = h5t.load(Root, _write(tmp_path / "a.h5", lambda f: f.create_group("snaps")))
    assert out.snaps == {}


def test_attr_dict_is_still_an_attribute(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Root:
        cfg: Annotated[dict[str, Any], h5t.Attr()]

    def build(f: h5py.File) -> None:
        f.attrs["cfg"] = '{"a": 1}'

    @h5t.group()
    @dataclasses.dataclass
    class Conv:
        cfg: Annotated[dict[str, Any], h5t.Attr(converter=__import__("json").loads)]

    assert h5t.load(Conv, _write(tmp_path / "a.h5", build)).cfg == {"a": 1}


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        (dict, "dict\\[str, T\\]"),
        (dict[int, Snapshot], "key type must be str"),
        (dict[str, int], "groups or datasets"),
        (dict[str, dict[str, Snapshot]], "nested collections"),
        (dict[str, Snapshot | None], "cannot be optional"),
        (dict[str, Annotated[Snapshot, h5t.Name("x")]], "Name cannot annotate"),
        (dict[str, Annotated[Snapshot, h5t.Glob("x")]], "Glob cannot annotate"),
        (Annotated[int, h5t.Glob("x")], "Glob applies only"),
        (Annotated[dict[str, Snapshot], h5t.Glob("a/b")], "cannot contain"),
        (Annotated[dict[str, Snapshot], h5t.Glob("")], "non-empty"),
    ],
)
def test_schema_errors(annotation: Any, message: str) -> None:
    with pytest.raises(h5t.SchemaError, match=message):

        @h5t.group()
        @dataclasses.dataclass
        class Bad:
            field: annotation  # ty: ignore[invalid-type-form]


def test_collection_on_dataset_record_is_rejected() -> None:
    with pytest.raises(h5t.SchemaError, match="only attributes"):

        @h5t.dataset(data="values")
        @dataclasses.dataclass
        class Bad:
            values: np.ndarray
            snaps: Annotated[dict[str, Snapshot], h5t.Glob("s*")]
