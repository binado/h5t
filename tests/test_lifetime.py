"""Handle lifetime: fd hygiene and closed-file behaviour."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import pytest

import h5t

from .conftest import PEResult, open_fd_count, write_pe_result

needs_dev_fd = pytest.mark.skipif(
    not os.path.isdir("/dev/fd"), reason="/dev/fd not available on this platform"
)


def _broken_file(tmp_path: Path) -> Path:
    path = tmp_path / "broken.h5"
    write_pe_result(path)
    with h5py.File(path, "a") as f:
        del f.attrs["version"]
    return path


@needs_dev_fd
def test_failed_validation_does_not_leak_fds(tmp_path: Path):
    path = _broken_file(tmp_path)
    # Warm up any lazily opened resources before taking the baseline.
    with pytest.raises(h5t.ValidationError):
        PEResult.open(path)
    baseline = open_fd_count()
    for _ in range(25):
        with pytest.raises(h5t.ValidationError):
            PEResult.open(path)
    assert open_fd_count() == baseline


@needs_dev_fd
def test_context_manager_closes_the_handle(pe_file: Path):
    with PEResult.open(pe_file):
        pass
    baseline = open_fd_count()
    for _ in range(25):
        with PEResult.open(pe_file) as f:
            f.runs["C01:XPHM"].mass_1[:]
    assert open_fd_count() == baseline


def test_retained_views_raise_after_exit(pe_file: Path):
    with PEResult.open(pe_file) as f:
        run = f.runs["C01:XPHM"]
        ds = run.mass_1
    with pytest.raises(h5t.ClosedFileError):
        ds[:]
    with pytest.raises(h5t.ClosedFileError):
        run.check()
    with pytest.raises(h5t.ClosedFileError):
        f.keys()


def test_cached_attr_values_survive_close(pe_file: Path):
    # Attrs are eager: once read, the Python value outlives the handle.
    with PEResult.open(pe_file) as f:
        run = f.runs["C01:XPHM"]
        approximant = run.approximant
    assert approximant == "IMRPhenomXPHM"


def test_explicit_close_is_idempotent(pe_file: Path):
    f = PEResult.open(pe_file)
    f.close()
    f.close()
    with pytest.raises(h5t.ClosedFileError):
        f.keys()


def test_validation_failure_closes_before_raising(tmp_path: Path):
    path = _broken_file(tmp_path)
    with pytest.raises(h5t.ValidationError) as excinfo:
        PEResult.open(path)
    assert excinfo.value.file_closed
    # The file is not locked: it can be reopened for writing immediately.
    with h5py.File(path, "a") as f:
        f.attrs["version"] = "1.0"
    with PEResult.open(path) as f:
        assert f.format_version == "1.0"
