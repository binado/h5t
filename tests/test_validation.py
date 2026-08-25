"""The validation matrix: mutate one thing, assert the *specific* error."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest

import h5t

from .conftest import PEResult, Posterior, write_pe_result, write_run


class Pair(h5t.File):
    x: h5t.Dataset[h5t.f8]
    y: h5t.Dataset[h5t.f8]
    label: str

    def validate(self) -> None:
        if self.x.shape != self.y.shape:
            raise h5t.Invalid("x and y must have the same shape")


def check(schema: type[h5t.File], path: Path) -> h5t.ValidationReport:
    with schema.open(path, validate=False) as view:
        return view.check()


def write_pair(path: Path, nx: int = 10, ny: int = 10, dtype: str = "f8") -> None:
    with h5py.File(path, "w") as f:
        f.attrs["label"] = "ok"
        f.create_dataset("x", data=np.zeros(nx, dtype=dtype))
        f.create_dataset("y", data=np.zeros(ny))


def test_conforming_file_is_clean(pe_file: Path):
    report = check(PEResult, pe_file)
    assert report.ok
    assert report.problems == []
    assert "ok" in report.render()


def test_dtype_mutation(tmp_path: Path):
    path = tmp_path / "t.h5"
    write_pair(path, dtype="f4")
    report = check(Pair, path)
    assert [p.message for p in report.errors] == ["dtype mismatch: expected f8, found float32"]
    assert report.errors[0].path == "/x"


def test_byte_order_is_not_an_error(tmp_path: Path):
    path = tmp_path / "t.h5"
    write_pair(path, dtype=">f8")
    assert check(Pair, path).ok


def test_group_validator_detects_mismatched_dataset_shapes(tmp_path: Path):
    path = tmp_path / "t.h5"
    write_pair(path, nx=10, ny=9)
    report = check(Pair, path)
    (problem,) = report.errors
    assert problem.path == "/"
    assert problem.message == "x and y must have the same shape"


def test_dropped_attr(tmp_path: Path):
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.zeros(3))
        f.create_dataset("y", data=np.zeros(3))
    report = check(Pair, path)
    assert [p.message for p in report.errors] == ["required attr 'label' missing"]


def test_mistaken_kind_hint_attr_vs_dataset(tmp_path: Path):
    class S(h5t.File):
        mass_1: np.ndarray  # someone meant a dataset

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("mass_1", data=np.zeros(3))
    report = check(S, path)
    (problem,) = report.errors
    assert "required attr 'mass_1' missing" in problem.message
    assert "but a dataset '/mass_1' exists here" in problem.message
    assert "Did you mean h5t.Dataset" in problem.message


def test_mistaken_kind_hint_dataset_vs_attr(tmp_path: Path):
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.attrs["label"] = "ok"
        f.attrs["x"] = np.zeros(3)
        f.create_dataset("y", data=np.zeros(3))
    report = check(Pair, path)
    (problem,) = report.errors
    assert "required dataset 'x' missing" in problem.message
    assert "but an attr 'x' exists here" in problem.message


def test_kind_mismatch_group_for_dataset(tmp_path: Path):
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.attrs["label"] = "ok"
        f.create_group("x")
        f.create_dataset("y", data=np.zeros(3))
    report = check(Pair, path)
    assert any("expected a dataset, found a group" in p.message for p in report.errors)


def test_fixed_dim(tmp_path: Path):
    class S(h5t.File):
        psd: h5t.Dataset[h5t.f8]

        def validate(self) -> None:
            if self.psd.ndim != 2 or self.psd.shape[1] != 2:
                raise h5t.Invalid("psd must have shape (n_freq, 2)")

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("psd", data=np.zeros((8, 3)))
    report = check(S, path)
    assert [p.message for p in report.errors] == ["psd must have shape (n_freq, 2)"]


def test_rank_mismatch(tmp_path: Path):
    class S(h5t.File):
        x: h5t.Dataset[h5t.f8]

        def validate(self) -> None:
            if self.x.ndim != 2:
                raise h5t.Invalid("x must have two axes")

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.zeros(5))
    report = check(S, path)
    (problem,) = report.errors
    assert problem.message == "x must have two axes"


def test_scalar_dataspace_vs_shape_one(tmp_path: Path):
    class S(h5t.File):
        x: h5t.Dataset[h5t.f8]

        def validate(self) -> None:
            if self.x.shape != ():
                raise h5t.Invalid("x must be a scalar dataspace")

    path = tmp_path / "scalar.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.float64(1.0))
    assert check(S, path).ok

    path2 = tmp_path / "one.h5"
    with h5py.File(path2, "w") as f:
        f.create_dataset("x", data=np.zeros(1))
    report = check(S, path2)
    assert [problem.message for problem in report.errors] == ["x must be a scalar dataspace"]


def test_anonymous_axis_accepts_anything(tmp_path: Path):
    class S(h5t.File):
        x: h5t.Dataset[h5t.f8]

        def validate(self) -> None:
            if self.x.ndim != 2:
                raise h5t.Invalid("x must have two axes")

    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.zeros((4, 17)))
    assert check(S, path).ok


class TestStringFlavours:
    def _write(self, path: Path, value: object) -> None:
        with h5py.File(path, "w") as f:
            f.attrs["label"] = value
            f.create_dataset("x", data=np.zeros(3))
            f.create_dataset("y", data=np.zeros(3))

    def test_fixed_bytes(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        self._write(path, np.bytes_(b"IMRPhenomXPHM"))
        assert check(Pair, path).ok

    def test_vlen_str(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        self._write(path, "utf-8 text")
        assert check(Pair, path).ok

    def test_undecodable_bytes_are_an_error(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        self._write(path, np.void(b"\xff\xfe"))
        report = check(Pair, path)
        assert any("attr 'label'" in p.message for p in report.errors)

    def test_non_string_attr_is_an_error(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        self._write(path, 3.5)
        report = check(Pair, path)
        assert any("expected a string" in p.message for p in report.errors)


def test_literal_mismatch(tmp_path: Path):
    path = tmp_path / "t.h5"
    write_pe_result(path)
    with h5py.File(path, "a") as f:
        f.attrs["version"] = "2.0"
    report = check(PEResult, path)
    (problem,) = report.errors
    assert problem.path == "/"
    assert "not in Literal['1.0']" in problem.message


class TestGroupValidator:
    def test_authoritative_source_beats_agreement(self, tmp_path: Path):
        # All datasets agree with each other but not with the claimed size:
        # inference alone cannot detect a uniformly truncated file.
        path = tmp_path / "t.h5"
        write_pe_result(path)
        with h5py.File(path, "a") as f:
            g = f["runs/C01:XPHM"]
            assert isinstance(g, h5py.Group)
            g.attrs["n_samples"] = 200
        report = check(PEResult, path)
        (problem,) = report.errors
        assert problem.path == "/runs/C01:XPHM"
        assert problem.message == "mass_1 must have shape (200,)"

    def test_missing_source_attr(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        write_pe_result(path)
        with h5py.File(path, "a") as f:
            del f["runs/C01:XPHM"].attrs["n_samples"]
        report = check(PEResult, path)
        assert any("required attr 'n_samples' missing" in p.message for p in report.errors)

    def test_non_integer_source_attr(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        write_pe_result(path)
        with h5py.File(path, "a") as f:
            f["runs/C01:XPHM"].attrs["n_samples"] = "many"
        report = check(PEResult, path)
        assert any("attr 'n_samples': expected int" in p.message for p in report.errors)


class TestExtras:
    def _write_with_junk(self, path: Path) -> None:
        with h5py.File(path, "w") as f:
            f.attrs["label"] = "ok"
            f.attrs["junk_attr"] = 1
            f.create_dataset("x", data=np.zeros(3))
            f.create_dataset("y", data=np.zeros(3))
            f.create_dataset("junk_ds", data=np.zeros(3))

    def test_ignore_is_default(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        self._write_with_junk(path)
        assert check(Pair, path).ok

    def test_warn(self, tmp_path: Path):
        class S(Pair, extras="warn"):
            pass

        path = tmp_path / "t.h5"
        self._write_with_junk(path)
        report = check(S, path)
        assert report.ok
        messages = {p.message for p in report.warnings}
        assert "extra dataset 'junk_ds' not in schema" in messages
        assert "extra attr 'junk_attr' not in schema" in messages

    def test_forbid(self, tmp_path: Path):
        class S(Pair, extras="forbid"):
            pass

        path = tmp_path / "t.h5"
        self._write_with_junk(path)
        report = check(S, path)
        assert not report.ok
        assert len(report.errors) == 2

    def test_open_emits_python_warnings(self, tmp_path: Path):
        class S(Pair, extras="warn"):
            pass

        path = tmp_path / "t.h5"
        self._write_with_junk(path)
        with pytest.warns(UserWarning, match="junk"):
            S.open(path).close()


class TestDynamicGroups:
    def test_missing_collection_group(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        with h5py.File(path, "w") as f:
            f.attrs["version"] = "1.0"
        report = check(PEResult, path)
        assert any("required group 'runs' missing" in p.message for p in report.errors)

    def test_empty_collection_is_valid(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        with h5py.File(path, "w") as f:
            f.attrs["version"] = "1.0"
            f.create_group("runs")
        assert check(PEResult, path).ok

    def test_each_matching_child_is_validated(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        write_pe_result(path, run_keys=("C01:XPHM", "C02:SEOB"))
        with h5py.File(path, "a") as f:
            del f["runs/C02:SEOB"].attrs["approximant"]
        report = check(PEResult, path)
        (problem,) = report.errors
        assert problem.path == "/runs/C02:SEOB"
        assert "required attr 'approximant' missing" in problem.message

    def test_nonmatching_children_follow_extras_policy(self, tmp_path: Path):
        class Strict(h5t.File):
            runs: Annotated[h5t.Group[Posterior], h5t.Keys(pattern=r"C\d+:.*")]

        path = tmp_path / "t.h5"
        write_pe_result(path)
        with h5py.File(path, "a") as f:
            f.create_group("runs/not_a_run")
        # default extras=ignore: nonmatching child ignored
        assert check(Strict, path).ok

        class Forbidding(h5t.File, extras="forbid"):
            runs: Annotated[h5t.Group[Posterior], h5t.Keys(pattern=r"C\d+:.*")]

        report = check(Forbidding, path)
        assert any("extra group 'not_a_run'" in p.message for p in report.errors)


class TestReportRendering:
    def test_report_batches_builtin_and_custom_findings(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        write_pe_result(path)
        with h5py.File(path, "a") as f:
            g = f["runs/C01:XPHM"]
            assert isinstance(g, h5py.Group)
            del g["mass_2"]
            g.create_dataset("mass_2", data=np.zeros(99))
            f.attrs["version"] = "2.0"
        report = check(PEResult, path)
        text = report.render()
        assert text.startswith("2 problems in")
        assert "/runs/C01:XPHM" in text
        assert "\u2717 attr 'version': value '2.0' not in Literal['1.0']" in text
        assert "\u2717 mass_2 must have shape (100,)" in text

    def test_validation_error_mentions_reopening(self, tmp_path: Path):
        path = tmp_path / "t.h5"
        write_pe_result(path)
        with h5py.File(path, "a") as f:
            del f.attrs["version"]
        with pytest.raises(h5t.ValidationError) as excinfo:
            PEResult.open(path)
        assert "File was closed" in str(excinfo.value)
        assert "PEResult.open(path, validate=False)" in str(excinfo.value)
        assert not excinfo.value.report.ok


def test_multiple_runs_do_not_share_n_samples(tmp_path: Path):
    # Each dynamic item validates itself, so runs may have different sizes.
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.attrs["version"] = "1.0"
        write_run(f.create_group("runs/C01:XPHM"), n_samples=100)
        write_run(f.create_group("runs/C02:SEOB"), n_samples=50)
    assert check(PEResult, path).ok
