"""The ``h5t check`` command."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import pytest

from h5t._cli import main

SCHEMA = """
import dataclasses

import h5t


@h5t.group()
@dataclasses.dataclass
class Result:
    version: int
"""


@pytest.fixture
def schema_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    module_name = f"schema_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(SCHEMA)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return f"{module_name}:Result"


def test_success_and_custom_root(
    tmp_path: Path, schema_ref: str, capsys: pytest.CaptureFixture
) -> None:
    path = tmp_path / "ok.h5"
    with h5py.File(path, "w") as file:
        file.create_group("result").attrs["version"] = 2
    assert main(["check", str(path), "--schema", schema_ref, "--root", "/result"]) == 0
    assert capsys.readouterr().out.startswith("ok:")


def test_first_validation_failure_exits_one(
    tmp_path: Path, schema_ref: str, capsys: pytest.CaptureFixture
) -> None:
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w"):
        pass
    assert main(["check", str(path), "--schema", schema_ref]) == 1
    output = capsys.readouterr().out
    assert output.startswith("invalid:")
    assert "/@version" in output


@pytest.mark.parametrize("schema", ["bad-ref", "missing_module:Result"])
def test_import_failures_exit_two(
    tmp_path: Path, schema: str, capsys: pytest.CaptureFixture
) -> None:
    path = tmp_path / "file.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(SystemExit) as caught:
        main(["check", str(path), "--schema", schema])
    assert caught.value.code == 2
    assert capsys.readouterr().err.startswith("error:")


def test_group_record_schema_succeeds(result_file: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["check", str(result_file), "--schema", "tests.conftest:Result"]) == 0
    assert capsys.readouterr().out.startswith("ok:")


def test_dataset_record_schema_exits_two(result_file: Path, capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["check", str(result_file), "--schema", "tests.conftest:EagerMeasurement"])
    assert caught.value.code == 2
    assert "dataset record" in capsys.readouterr().err


def test_unresolved_nested_group_record_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module_name = f"schema_{tmp_path.name.replace('-', '_')}_nested"
    (tmp_path / f"{module_name}.py").write_text(
        """\
from __future__ import annotations

import dataclasses

import h5t


@h5t.group()
@dataclasses.dataclass
class Inner:
    later: Later


@h5t.group()
@dataclasses.dataclass
class Owner:
    nested: Inner
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    path = tmp_path / "file.h5"
    with h5py.File(path, "w") as file:
        file.create_group("nested").attrs["later"] = 1
    with pytest.raises(SystemExit) as caught:
        main(["check", str(path), "--schema", f"{module_name}:Owner"])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Later" in err


def test_io_and_invalid_root_exit_two(
    tmp_path: Path, schema_ref: str, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["check", str(tmp_path / "missing.h5"), "--schema", schema_ref])
    assert caught.value.code == 2
    assert "could not check" in capsys.readouterr().err

    path = tmp_path / "file.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(SystemExit) as caught:
        main(["check", str(path), "--schema", schema_ref, "--root", "relative"])
    assert caught.value.code == 2
