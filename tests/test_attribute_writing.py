"""Write-side attribute serialization and validation."""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest

import h5t
from h5t._compile import _write_attribute


def _fields(record: type) -> dict[str, Any]:
    return {field.py_name: field for field in record.__h5t_record__.spec.fields}


def test_json_converter_and_serializer_round_trip(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Schema:
        payload: Annotated[
            dict[str, list[int]],
            h5t.Name("stored-payload"),
            h5t.Attr(converter=json.loads, serializer=json.dumps),
        ]

    path = tmp_path / "json.h5"
    with h5py.File(path, "w") as file:
        _write_attribute(_fields(Schema)["payload"], {"numbers": [1, 2]}, file, "/group")
        assert json.loads(file.attrs["stored-payload"]) == {"numbers": [1, 2]}

    loaded = h5t.load(Schema, path)
    assert loaded.payload == {"numbers": [1, 2]}


def test_serializer_failure_is_public_chained_and_path_aware(tmp_path: Path) -> None:
    def broken(value: str) -> str:
        raise RuntimeError(f"cannot encode {value}")

    @h5t.group()
    @dataclasses.dataclass
    class Schema:
        value: Annotated[str, h5t.Name("stored"), h5t.Attr(serializer=broken)]

    with h5py.File(tmp_path / "failure.h5", "w") as file:
        with pytest.raises(h5t.ConversionError, match="cannot encode") as caught:
            _write_attribute(_fields(Schema)["value"], "bad", file, "/group")
    assert caught.value.path == "/group@stored"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_write_validates_before_serializing(tmp_path: Path) -> None:
    calls: list[Any] = []

    @h5t.group()
    @dataclasses.dataclass
    class Schema:
        value: Annotated[int, h5t.Attr(serializer=lambda value: calls.append(value))]

    with h5py.File(tmp_path / "validation.h5", "w") as file:
        with pytest.raises(h5t.ValidationError) as caught:
            _write_attribute(_fields(Schema)["value"], "not-an-int", file, "/group")
    assert caught.value.path == "/group@value"
    assert calls == []


def test_numpy_attributes_enums_and_scalars_are_hdf5_writable(tmp_path: Path) -> None:
    class Mode(Enum):
        FAST = "fast"

    @h5t.group()
    @dataclasses.dataclass
    class Schema:
        array: Annotated[np.ndarray, h5t.Attr()]
        mode: Mode
        count: int
        scalar: np.int64

    fields = _fields(Schema)
    with h5py.File(tmp_path / "native.h5", "w") as file:
        _write_attribute(fields["array"], np.arange(3), file, "/")
        _write_attribute(fields["mode"], Mode.FAST, file, "/")
        _write_attribute(fields["count"], 3, file, "/")
        _write_attribute(fields["scalar"], np.int64(4), file, "/")
        np.testing.assert_array_equal(file.attrs["array"], np.arange(3))
        assert file.attrs["mode"] == "fast"
        assert file.attrs["count"] == 3
        assert file.attrs["scalar"] == 4


def test_converter_is_not_used_as_an_implicit_serializer(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Schema:
        payload: Annotated[dict[str, int], h5t.Attr(converter=json.loads)]

    with h5py.File(tmp_path / "one-way.h5", "w") as file:
        with pytest.raises(h5t.ValidationError, match=r"Attr\(serializer") as caught:
            _write_attribute(_fields(Schema)["payload"], {"value": 1}, file, "/group")
    assert caught.value.path == "/group@payload"


@pytest.mark.parametrize("argument", ["converter", "serializer"])
def test_attr_operations_must_be_callable(argument: str) -> None:
    with pytest.raises(h5t.SchemaError, match=rf"Attr\.{argument} must be callable"):

        @h5t.group()
        @dataclasses.dataclass
        class Invalid:
            value: Annotated[int, h5t.Attr(**{argument: 1})]  # type: ignore[arg-type]


def test_attr_marker_may_appear_only_once_with_serializers() -> None:
    with pytest.raises(h5t.SchemaError, match="Attr may appear only once"):

        @h5t.group()
        @dataclasses.dataclass
        class Invalid:
            value: Annotated[str, h5t.Attr(serializer=str), h5t.Attr(serializer=repr)]
