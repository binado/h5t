"""Lazy views: typed access, mapping API, caching, and targeted errors."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import h5t

from .conftest import PEResult, Posterior, write_pe_result


def test_typed_member_access(pe_file: Path):
    with PEResult.open(pe_file) as f:
        run = f.runs["C01:XPHM"]
        assert isinstance(run, Posterior)
        assert isinstance(run.mass_1, h5t.Dataset)
        assert isinstance(f.runs, h5t.Group)


def test_attrs_are_eager_and_normalised(pe_file: Path):
    with PEResult.open(pe_file) as f:
        run = f.runs["C01:XPHM"]
        assert run.approximant == "IMRPhenomXPHM"
        assert type(run.approximant) is str
        assert type(run.f_ref) is float  # np.float64 in the file
        assert run.f_ref == 20.0


def test_bytes_attr_normalises_to_str(tmp_path: Path):
    path = tmp_path / "t.h5"
    write_pe_result(path)
    with h5py.File(path, "a") as f:
        f["runs/C01:XPHM"].attrs["approximant"] = np.bytes_(b"IMRPhenomXPHM")
    with PEResult.open(path) as f:
        run = f.runs["C01:XPHM"]
        assert run.approximant == "IMRPhenomXPHM"
        assert run.approximant != np.bytes_(b"OTHER")


def test_datasets_are_lazy_and_sliceable(pe_file: Path):
    with PEResult.open(pe_file) as f:
        ds = f.runs["C01:XPHM"].mass_1
        assert ds.shape == (100,)
        assert ds.dtype == np.dtype("f8")
        assert ds.ndim == 1
        assert len(ds) == 100
        chunk = ds[:5]
        assert isinstance(chunk, np.ndarray)
        assert chunk.shape == (5,)
        assert ds[:].shape == (100,)


def test_optional_missing_members_read_as_none(pe_file: Path):
    with PEResult.open(pe_file) as f:
        assert f.runs["C01:XPHM"].spins is None


def test_author_properties_work(pe_file: Path):
    with PEResult.open(pe_file) as f:
        run = f.runs["C01:XPHM"]
        cm = run.chirp_mass
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (100,)


def test_views_cache_per_instance(pe_file: Path):
    with PEResult.open(pe_file) as f:
        assert f.runs is f.runs
        run = f.runs["C01:XPHM"]
        assert run.mass_1 is run.mass_1


def test_mapping_api_two_namespaces(pe_file: Path):
    with PEResult.open(pe_file) as f:
        assert "runs" in f
        assert list(f.keys()) == ["runs"]
        assert len(f) == 1
        assert list(iter(f)) == ["runs"]
        # attrs namespace is separate and raw
        assert "version" in f.attrs
        raw = f.attrs["version"]
        assert raw == "1.0"


def test_bracket_access_is_untyped_escape_hatch(pe_file: Path):
    with PEResult.open(pe_file) as f:
        # PEResult has no dynamic children: bracket access returns raw h5py.
        raw = f["runs"]
        assert isinstance(raw, h5py.Group)
        # A dynamic Group[T] returns typed views for matching keys...
        assert isinstance(f.runs["C01:XPHM"], Posterior)
        # ...and raw h5py objects for nonmatching keys.


def test_dynamic_group_nonmatching_key_returns_raw(tmp_path: Path):
    path = tmp_path / "t.h5"
    write_pe_result(path)
    with h5py.File(path, "a") as f:
        f.create_group("runs/junk")
    with PEResult.open(path) as f:
        assert isinstance(f.runs["junk"], h5py.Group)
        assert isinstance(f.runs["C01:XPHM"], Posterior)


def test_dynamic_group_missing_key_raises_schema_mismatch(pe_file: Path):
    with PEResult.open(pe_file) as f:
        with pytest.raises(h5t.SchemaMismatchError, match="no such child"):
            f.runs["C99:NOPE"]


def test_subtree_check_and_validate(pe_file: Path):
    with PEResult.open(pe_file) as f:
        run = f.runs["C01:XPHM"]
        report = run.check()
        assert report.ok
        run.validate()  # should not raise


def test_repr_names_path_and_state(pe_file: Path):
    f = PEResult.open(pe_file)
    assert "'/'" in repr(f)
    assert "open" in repr(f)
    f.close()
    assert "closed" in repr(f)


class TestValidateFalse:
    """`validate=False` still has designed failure modes."""

    def _broken(self, tmp_path: Path) -> Path:
        path = tmp_path / "t.h5"
        write_pe_result(path)
        with h5py.File(path, "a") as f:
            g = f["runs/C01:XPHM"]
            assert isinstance(g, h5py.Group)
            del g.attrs["approximant"]
            del g["mass_1"]
            g.create_group("mass_1")  # wrong kind
        return path

    def test_open_does_not_raise(self, tmp_path: Path):
        path = self._broken(tmp_path)
        with PEResult.open(path, validate=False) as f:
            assert isinstance(f, PEResult)

    def test_missing_attr_is_targeted(self, tmp_path: Path):
        path = self._broken(tmp_path)
        with PEResult.open(path, validate=False) as f:
            run = f.runs["C01:XPHM"]
            with pytest.raises(h5t.SchemaMismatchError) as excinfo:
                run.approximant
            assert "/runs/C01:XPHM" in str(excinfo.value)
            assert "required attr 'approximant' missing" in str(excinfo.value)

    def test_wrong_kind_is_targeted(self, tmp_path: Path):
        path = self._broken(tmp_path)
        with PEResult.open(path, validate=False) as f:
            run = f.runs["C01:XPHM"]
            with pytest.raises(h5t.SchemaMismatchError, match="expected a dataset"):
                run.mass_1

    def test_conforming_members_still_work(self, tmp_path: Path):
        path = self._broken(tmp_path)
        with PEResult.open(path, validate=False) as f:
            run = f.runs["C01:XPHM"]
            assert run.mass_2[:].shape == (100,)

    def test_check_reports_everything(self, tmp_path: Path):
        path = self._broken(tmp_path)
        with PEResult.open(path, validate=False) as f:
            report = f.check()
        assert not report.ok
        assert len(report.errors) >= 2
