"""The ``h5t check`` command."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import pytest

from h5t._cli import main

SCHEMA = """
from pydantic import BaseModel
import h5t


class ResultAttrs(BaseModel):
    version: int


class Result(h5t.Group[ResultAttrs]):
    pass
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


def test_non_group_schema_exits_two(result_file: Path, capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["check", str(result_file), "--schema", "tests.conftest:MeasurementAttrs"])
    assert caught.value.code == 2
    assert "Group subclass" in capsys.readouterr().err


def test_unresolved_nested_group_record_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module_name = f"schema_{tmp_path.name.replace('-', '_')}_nested"
    (tmp_path / f"{module_name}.py").write_text(
        """\
from __future__ import annotations

import h5t


class Inner(h5t.Group):
    later: Later


class Owner(h5t.Group):
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


BUGGY_SCHEMA = """
from pydantic import BaseModel, field_validator
import h5t


class BuggyAttrs(BaseModel):
    version: int

    @field_validator("version")
    @classmethod
    def explode(cls, value):
        raise KeyError("validator bug")


class Buggy(h5t.Group[BuggyAttrs]):
    pass
"""


@pytest.fixture
def buggy_schema_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    module_name = f"buggy_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(BUGGY_SCHEMA)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return f"{module_name}:Buggy"


@pytest.fixture
def versioned_file(tmp_path: Path) -> Path:
    path = tmp_path / "versioned.h5"
    with h5py.File(path, "w") as file:
        file.attrs["version"] = 1
    return path


def test_validator_bug_exits_two_and_names_the_exception_type(
    versioned_file: Path, buggy_schema_ref: str, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["check", str(versioned_file), "--schema", buggy_schema_ref])
    assert caught.value.code == 2
    assert "KeyError" in capsys.readouterr().err


def test_traceback_flag_reraises_unexpected_errors(
    versioned_file: Path, buggy_schema_ref: str
) -> None:
    with pytest.raises(KeyError):
        main(["check", str(versioned_file), "--schema", buggy_schema_ref, "--traceback"])


def test_oserror_message_names_the_exception_type(
    tmp_path: Path, schema_ref: str, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(SystemExit):
        main(["check", str(tmp_path / "missing.h5"), "--schema", schema_ref])
    assert "FileNotFoundError" in capsys.readouterr().err
