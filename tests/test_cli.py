"""The `h5t check` command line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import pytest

from h5t._cli import main

from .conftest import write_pe_result

SCHEMA_MODULE = """
from typing import Annotated, Literal

import h5t


class Posterior(h5t.Group):
    mass_1: h5t.Dataset[h5t.f8]
    mass_2: h5t.Dataset[h5t.f8]
    log_likelihood: h5t.Dataset[h5t.f8]
    psd: h5t.Dataset[h5t.f8]
    n_samples: int
    approximant: str
    f_ref: float = 20.0

    def validate(self) -> None:
        if self.mass_1.shape != (self.n_samples,):
            raise h5t.Invalid("mass_1 must match n_samples")


class PEResult(h5t.File):
    runs: Annotated[h5t.Group[Posterior], h5t.Keys(pattern=r"C\\d+:.*")]
    format_version: Annotated[Literal["1.0"], h5t.Name("version")]
"""


@pytest.fixture
def schema_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    module_name = f"cli_schema_{tmp_path.name}"
    (tmp_path / f"{module_name}.py").write_text(SCHEMA_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return f"{module_name}:PEResult"


def test_check_ok_exits_zero(tmp_path, schema_ref, capsys):
    path = tmp_path / "good.h5"
    write_pe_result(path)
    assert main(["check", str(path), "--schema", schema_ref]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ok:")
    assert "PEResult" in out


def test_check_failure_exits_one_and_prints_report(tmp_path, schema_ref, capsys):
    path = tmp_path / "bad.h5"
    write_pe_result(path)
    with h5py.File(path, "a") as f:
        f.attrs["version"] = "0.9"
    assert main(["check", str(path), "--schema", schema_ref]) == 1
    out = capsys.readouterr().out
    assert "1 problem in" in out
    assert "not in Literal['1.0']" in out


def test_bad_schema_ref_exits_two(tmp_path, capsys):
    path = tmp_path / "x.h5"
    write_pe_result(path)
    with pytest.raises(SystemExit) as excinfo:
        main(["check", str(path), "--schema", "not-a-ref"])
    assert excinfo.value.code == 2
    assert "pkg.module:ClassName" in capsys.readouterr().err


def test_missing_module_exits_two(tmp_path, capsys):
    path = tmp_path / "x.h5"
    write_pe_result(path)
    with pytest.raises(SystemExit) as excinfo:
        main(["check", str(path), "--schema", "no_such_module:X"])
    assert excinfo.value.code == 2


def test_non_schema_attribute_exits_two(tmp_path, schema_ref):
    path = tmp_path / "x.h5"
    write_pe_result(path)
    module_name = schema_ref.split(":")[0]
    with pytest.raises(SystemExit) as excinfo:
        main(["check", str(path), "--schema", f"{module_name}:Posterior"])
    assert excinfo.value.code == 2


def test_unreadable_file_exits_two(tmp_path, schema_ref, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["check", str(tmp_path / "missing.h5"), "--schema", schema_ref])
    assert excinfo.value.code == 2
    assert "could not open" in capsys.readouterr().err
