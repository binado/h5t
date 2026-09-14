from __future__ import annotations

import dataclasses

import h5py

import h5t


@h5t.group()
@dataclasses.dataclass
class Writable:
    required: int
    optional: str | None = None
    revision: int = 1
    generated: int = dataclasses.field(default_factory=lambda: 2)


def test_dump_omits_none_optional_and_materializes_defaults(tmp_path):
    path = tmp_path / "dump.h5"

    h5t.dump(Writable(required=3), path)

    with h5py.File(path) as file:
        assert dict(file.attrs) == {"required": 3, "revision": 1, "generated": 2}


def test_dumped_loaded_record_is_equivalent_not_presence_preserving(tmp_path):
    source = tmp_path / "source.h5"
    dumped = tmp_path / "dumped.h5"
    with h5py.File(source, "w") as file:
        file.attrs["required"] = 3

    original = h5t.load(Writable, source)
    h5t.dump(original, dumped)
    reloaded = h5t.load(Writable, dumped)

    assert reloaded == original
    with h5py.File(dumped) as file:
        assert "optional" not in file.attrs
        assert "revision" in file.attrs
        assert "generated" in file.attrs


def test_dump_rejects_none_or_unreadable_required_field_at_field_path(tmp_path):
    record = Writable(required=3)
    record.required = None  # type: ignore[assignment]
    try:
        h5t.dump(record, tmp_path / "none.h5")
    except h5t.ValidationError as exc:
        assert str(exc).startswith("/@required:")
    else:
        raise AssertionError("expected ValidationError")

    del record.required
    try:
        h5t.dump(record, tmp_path / "missing.h5")
    except h5t.ValidationError as exc:
        assert str(exc).startswith("/@required:")
    else:
        raise AssertionError("expected ValidationError")
