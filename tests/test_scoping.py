"""Dimension scoping: declared vs free dims, class reuse, dynamic scopes.

Note: no ``from __future__ import annotations`` here — schema classes are
defined inside test functions and reference local names, which PEP 563
string annotations cannot resolve (a documented v0.1 limitation).
"""

from pathlib import Path

import h5py
import numpy as np

import h5t
from h5t._validate import run_validation


def check(schema: type[h5t.File], path: Path) -> h5t.ValidationReport:
    with h5py.File(path, "r") as f:
        return run_validation(schema.__h5spec__, f, filename=str(path), schema_name=schema.__name__)


class FreeDetector(h5t.Group):
    """`n_time` is free: it unifies upward to the file root."""

    strain: h5t.Dataset[h5t.f8, "n_time"]


class LocalDetector(h5t.Group, dims={"n_time": None}):
    """`n_time` is declared (source-less): each use binds independently."""

    strain: h5t.Dataset[h5t.f8, "n_time"]


def write_two_detectors(path: Path, n_h1: int, n_l1: int) -> None:
    with h5py.File(path, "w") as f:
        f.create_group("h1").create_dataset("strain", data=np.zeros(n_h1))
        f.create_group("l1").create_dataset("strain", data=np.zeros(n_l1))


def test_free_dims_share_across_static_reuse(tmp_path: Path):
    class F(h5t.File):
        h1: FreeDetector
        l1: FreeDetector

    path = tmp_path / "t.h5"
    write_two_detectors(path, 5, 6)
    report = check(F, path)
    (problem,) = report.errors
    assert problem.path == "/"
    assert "'n_time' has inconsistent values" in problem.message
    assert "5 at ./h1/strain" in problem.message
    assert "6 at ./l1/strain" in problem.message


def test_free_dims_agreeing_are_clean(tmp_path: Path):
    class F(h5t.File):
        h1: FreeDetector
        l1: FreeDetector

    path = tmp_path / "t.h5"
    write_two_detectors(path, 5, 5)
    assert check(F, path).ok


def test_declared_dims_isolate_per_use(tmp_path: Path):
    class F(h5t.File):
        h1: LocalDetector
        l1: LocalDetector

    path = tmp_path / "t.h5"
    write_two_detectors(path, 5, 6)
    assert check(F, path).ok


def test_free_dims_unify_between_siblings_that_never_mention_each_other(
    tmp_path: Path,
):
    class A(h5t.Group):
        x: h5t.Dataset[h5t.f8, "n_freq 2"]

    class B(h5t.Group):
        y: h5t.Dataset[h5t.f8, "n_freq"]

    class F(h5t.File):
        a: A
        b: B

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_group("a").create_dataset("x", data=np.zeros((8, 2)))
        f.create_group("b").create_dataset("y", data=np.zeros(9))
    report = check(F, path)
    (problem,) = report.errors
    assert "'n_freq' has inconsistent values" in problem.message
    assert "8 at ./a/x" in problem.message
    assert "9 at ./b/y" in problem.message


def test_dim_unifies_within_one_dataset_across_axes(tmp_path: Path):
    class F(h5t.File):
        m: h5t.Dataset[h5t.f8, "n n"]

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("m", data=np.zeros((3, 4)))
    report = check(F, path)
    (problem,) = report.errors
    assert "'n' has inconsistent values" in problem.message
    assert "4 at ./m (axis 1)" in problem.message


def test_dynamic_children_never_unify(tmp_path: Path):
    class Item(h5t.Group):
        x: h5t.Dataset[h5t.f8, "n"]

    class F(h5t.File):
        items: h5t.Group[Item]

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_group("items/a").create_dataset("x", data=np.zeros(3))
        f.create_group("items/b").create_dataset("x", data=np.zeros(7))
    assert check(F, path).ok


def test_dims_do_not_escape_dynamic_children_to_parent(tmp_path: Path):
    class Item(h5t.Group):
        x: h5t.Dataset[h5t.f8, "n"]

    class F(h5t.File):
        top: h5t.Dataset[h5t.f8, "n"]
        items: h5t.Group[Item]

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("top", data=np.zeros(100))
        f.create_group("items/a").create_dataset("x", data=np.zeros(3))
    assert check(F, path).ok


def test_declared_dim_with_source_binds_per_class_use(tmp_path: Path):
    class Sized(h5t.Group, dims={"n": h5t.FromAttr("n")}):
        x: h5t.Dataset[h5t.f8, "n"]

    class F(h5t.File):
        a: Sized
        b: Sized

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        ga = f.create_group("a")
        ga.attrs["n"] = 3
        ga.create_dataset("x", data=np.zeros(3))
        gb = f.create_group("b")
        gb.attrs["n"] = 8
        gb.create_dataset("x", data=np.zeros(8))
    assert check(F, path).ok


def test_free_dim_conflict_inside_one_group_reports_at_that_scope(tmp_path: Path):
    class Inner(h5t.Group, dims={"n": None}):
        x: h5t.Dataset[h5t.f8, "n"]
        y: h5t.Dataset[h5t.f8, "n"]

    class F(h5t.File):
        inner: Inner

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("inner")
        g.create_dataset("x", data=np.zeros(2))
        g.create_dataset("y", data=np.zeros(5))
    report = check(F, path)
    (problem,) = report.errors
    assert problem.path == "/inner"
    assert "2 at ./x" in problem.message
    assert "5 at ./y" in problem.message
