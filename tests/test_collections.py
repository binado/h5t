"""Dynamic collections: ``dict[str, T]`` fields and the ``h5t.Glob`` filter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field

import h5t
from h5t._array import _DATA_NOT_LOADED


class InnerAttrs(BaseModel):
    label: str


class Inner(h5t.Group[InnerAttrs]):
    pass


class SnapshotAttrs(BaseModel):
    time: float


class Snapshot(h5t.Group[SnapshotAttrs]):
    inner: Inner | None = None


class SeriesAttrs(BaseModel):
    unit: str


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
    class Run(h5t.Group):
        snapshots: dict[str, Snapshot]

    class Root(h5t.Group):
        run: Run

    out = Root.model_validate_hdf5(_write(tmp_path / "a.h5", _snapshots))
    snaps = out.run.snapshots
    assert type(snaps) is dict
    assert list(snaps) == ["snap_0", "snap_1"]
    assert snaps["snap_1"] == Snapshot(
        attrs=SnapshotAttrs(time=1.0), inner=Inner(attrs=InnerAttrs(label="l1"))
    )


def test_mapping_annotation_and_dataset_items(tmp_path: Path) -> None:
    def build(f: h5py.File) -> None:
        g = f.create_group("series")
        for n in ("a", "b"):
            d = g.create_dataset(n, data=np.arange(3.0))
            d.attrs["unit"] = "m"

    class Root(h5t.Group):
        series: Mapping[str, h5t.Dataset[SeriesAttrs]]

    out = Root.model_validate_hdf5(_write(tmp_path / "a.h5", build))
    assert set(out.series) == {"a", "b"}
    assert out.series["a"].attrs.unit == "m"


def test_dataset_items_read_no_data(tmp_path: Path) -> None:
    def build(f: h5py.File) -> None:
        g = f.create_group("arrs")
        g.create_dataset("x", data=np.arange(4.0))
        g.create_dataset("y", data=np.arange(5.0))

    class Root(h5t.Group):
        arrs: dict[str, h5t.Dataset]

    path = _write(tmp_path / "a.h5", build)
    out = Root.model_validate_hdf5(path)
    assert out.arrs["x"].shape == (4,)
    assert out.arrs["x"]._h5t_data is _DATA_NOT_LOADED


def test_ndarray_and_jaxtyping_items_share_scope_with_owner(tmp_path: Path) -> None:
    def build(f: h5py.File, third: int = 3) -> None:
        g = f.create_group("arrs")
        g.create_dataset("x", data=np.zeros((3, 3)))
        g.create_dataset("y", data=np.zeros((third, third)))
        f.create_dataset("own", data=np.zeros(3))
        f.create_group("plain").create_dataset("p", data=np.arange(2))

    class Root(h5t.Group):
        arrs: dict[str, Float[np.ndarray, "n n"]]
        own: Float[np.ndarray, "n"]
        plain: dict[str, np.ndarray]

    out = Root.model_validate_hdf5(_write(tmp_path / "ok.h5", build))
    assert out.plain["p"].tolist() == [0, 1]

    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        build(f, third=4)
    with pytest.raises(h5t.ValidationError, match=r"/arrs/y"):
        Root.model_validate_hdf5(bad)


def test_group_items_bind_independently(tmp_path: Path) -> None:
    class Mat(h5t.Group):
        m: Float[np.ndarray, "n n"]

    def build(f: h5py.File) -> None:
        g = f.create_group("mats")
        g.create_group("a").create_dataset("m", data=np.zeros((2, 2)))
        g.create_group("b").create_dataset("m", data=np.zeros((3, 3)))

    class Root(h5t.Group):
        mats: dict[str, Mat]

    assert set(Root.model_validate_hdf5(_write(tmp_path / "a.h5", build)).mats) == {"a", "b"}


def test_glob_filters_case_sensitively_and_skips_unmatched(tmp_path: Path) -> None:
    def build(f: h5py.File) -> None:
        g = f.create_group("chains")
        for n in ("chain_1", "chain_2", "Chain_3", "other", "chain_10"):
            g.create_dataset(n, data=np.arange(2))

    class Root(h5t.Group):
        chains: Annotated[dict[str, h5t.Dataset], h5t.Glob("chain_?")]

    out = Root.model_validate_hdf5(_write(tmp_path / "a.h5", build))
    assert set(out.chains) == {"chain_1", "chain_2"}


def test_empty_group_gives_empty_dict(tmp_path: Path) -> None:
    class Root(h5t.Group):
        snaps: dict[str, Snapshot]

    out = Root.model_validate_hdf5(_write(tmp_path / "a.h5", lambda f: f.create_group("snaps")))
    assert out.snaps == {}


def test_alias_on_collection_field(tmp_path: Path) -> None:
    class Root(h5t.Group):
        raw: dict[str, h5t.Dataset] = Field(alias="Raw Data")

    def build(f: h5py.File) -> None:
        f.create_group("Raw Data").create_dataset("x", data=np.arange(2))

    assert list(Root.model_validate_hdf5(_write(tmp_path / "a.h5", build)).raw) == ["x"]


def test_missing_collection_group(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.h5", lambda f: f.attrs.__setitem__("v", 1))

    class Required(h5t.Group):
        snaps: dict[str, Snapshot]

    with pytest.raises(h5t.ValidationError, match=r"/snaps.*required collection 'snaps'"):
        Required.model_validate_hdf5(path)

    class Optional_(h5t.Group):
        snaps: dict[str, Snapshot] | None

    assert Optional_.model_validate_hdf5(path).snaps is None

    class Defaulted(h5t.Group):
        snaps: dict[str, Snapshot] = Field(default_factory=dict)

    assert Defaulted.model_validate_hdf5(path).snaps == {}


def test_dataset_at_collection_path(tmp_path: Path) -> None:
    class Root(h5t.Group):
        snaps: dict[str, Snapshot]

    path = _write(tmp_path / "a.h5", lambda f: f.create_dataset("snaps", data=1))
    with pytest.raises(h5t.ValidationError, match="expected a group, found a dataset"):
        Root.model_validate_hdf5(path)


def test_item_failure_reports_item_path(tmp_path: Path) -> None:
    def build(f: h5py.File) -> None:
        _snapshots(f)
        f["run/snapshots/snap_1"].attrs["time"] = "late"

    class Run(h5t.Group):
        snapshots: dict[str, Snapshot]

    class Root(h5t.Group):
        run: Run

    with pytest.raises(h5t.ValidationError) as info:
        Root.model_validate_hdf5(_write(tmp_path / "a.h5", build))
    assert "/run/snapshots/snap_1@time" in str(info.value)


def test_self_referential_collection_terminates(tmp_path: Path) -> None:
    class NodeAttrs(BaseModel):
        value: int

    class Node(h5t.Group[NodeAttrs]):
        children: dict[str, Node] = Field(default_factory=dict)

    def build(f: h5py.File) -> None:
        f.attrs["value"] = 0
        c = f.create_group("children").create_group("a")
        c.attrs["value"] = 1
        c.create_group("children").create_group("b").attrs["value"] = 2

    out = Node.model_validate_hdf5(_write(tmp_path / "a.h5", build))
    assert out.children["a"].children["b"].attrs.value == 2
    assert out.children["a"].children["b"].children == {}


def test_extras_forbid_allows_collection(tmp_path: Path) -> None:
    class Root(h5t.Group):
        model_config = ConfigDict(extra="forbid")

        snaps: dict[str, Snapshot]

    out = Root.model_validate_hdf5(_write(tmp_path / "a.h5", lambda f: f.create_group("snaps")))
    assert out.snaps == {}


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        (dict, "dict\\[str, T\\]"),
        (dict[int, Snapshot], "key type must be str"),
        (dict[str, int], "unsupported child field annotation"),
        (dict[str, dict[str, Snapshot]], "nested collections"),
        (dict[str, Snapshot | None], "cannot be optional"),
        (dict[str, Annotated[Snapshot, Field(alias="x")]], "alias cannot annotate"),
        (dict[str, Annotated[Snapshot, h5t.Glob("x")]], "Glob cannot annotate"),
        (Annotated[np.ndarray, h5t.Glob("x")], "Glob applies only"),
        (Annotated[dict[str, Snapshot], h5t.Glob("a/b")], "cannot contain"),
        (Annotated[dict[str, Snapshot], h5t.Glob("")], "non-empty"),
    ],
)
def test_schema_errors(annotation: Any, message: str) -> None:
    with pytest.raises(h5t.SchemaError, match=message):

        class Bad(h5t.Group):
            field: annotation  # ty: ignore[invalid-type-form]
