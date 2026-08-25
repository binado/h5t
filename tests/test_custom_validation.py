"""User-defined validation hooks on file, group, and dataset views."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import h5t


def check(schema: type[h5t.File], path: Path) -> h5t.ValidationReport:
    with schema.open(path, validate=False) as view:
        return view.check()


def test_dataset_group_and_file_hooks_run_postorder(tmp_path: Path):
    calls: list[str] = []

    class Values(h5t.Dataset, dtype=h5t.f8):
        def validate(self) -> None:
            calls.append("dataset")

    class Container(h5t.Group):
        values: Values

        def validate(self) -> None:
            calls.append("group")

    class Schema(h5t.File):
        container: Container

        def validate(self) -> None:
            calls.append("file")

    path = tmp_path / "ordered.h5"
    with h5py.File(path, "w") as file:
        file.create_group("container").create_dataset("values", data=np.zeros(3))

    assert check(Schema, path).ok
    assert calls == ["dataset", "group", "file"]


def test_validator_inheritance_uses_normal_super_calls(tmp_path: Path):
    calls: list[str] = []

    class Base(h5t.Group):
        def validate(self) -> None:
            calls.append("base")

    class Derived(Base):
        def validate(self) -> None:
            super().validate()
            calls.append("derived")

    class Schema(h5t.File):
        item: Derived

    path = tmp_path / "inheritance.h5"
    with h5py.File(path, "w") as file:
        file.create_group("item")

    assert check(Schema, path).ok
    assert calls == ["base", "derived"]


def test_invalid_is_attached_to_invoking_node_and_siblings_continue(tmp_path: Path):
    calls: list[str] = []

    class Values(h5t.Dataset, dtype=h5t.f8):
        def validate(self) -> None:
            calls.append(self._h5t_path)
            if self.shape != (3,):
                raise h5t.Invalid("expected three values")

    class Schema(h5t.File):
        left: Values
        right: Values

    path = tmp_path / "siblings.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("left", data=np.zeros(2))
        file.create_dataset("right", data=np.zeros(4))

    report = check(Schema, path)
    assert calls == ["/left", "/right"]
    assert [(problem.path, problem.message) for problem in report.errors] == [
        ("/left", "expected three values"),
        ("/right", "expected three values"),
    ]


def test_group_validates_descendants_through_self(tmp_path: Path):
    class Pair(h5t.Group):
        left: h5t.Dataset[h5t.f8]
        right: h5t.Dataset[h5t.f8]

        def validate(self) -> None:
            if self.left.shape != self.right.shape:
                raise h5t.Invalid("left and right must have the same shape")

    class Schema(h5t.File):
        pair: Pair

    path = tmp_path / "pair.h5"
    with h5py.File(path, "w") as file:
        pair = file.create_group("pair")
        pair.create_dataset("left", data=np.zeros(2))
        pair.create_dataset("right", data=np.zeros(3))

    report = check(Schema, path)
    assert [(problem.path, problem.message) for problem in report.errors] == [
        ("/pair", "left and right must have the same shape")
    ]


def test_dynamic_items_are_validated_independently(tmp_path: Path):
    class Item(h5t.Group):
        values: h5t.Dataset[h5t.f8]
        size: int

        def validate(self) -> None:
            if self.values.shape != (self.size,):
                raise h5t.Invalid("values must match size")

    class Schema(h5t.File):
        items: h5t.Group[Item]

    path = tmp_path / "dynamic.h5"
    with h5py.File(path, "w") as file:
        for name, declared, actual in (("a", 2, 2), ("b", 3, 4)):
            item = file.create_group(f"items/{name}")
            item.attrs["size"] = declared
            item.create_dataset("values", data=np.zeros(actual))

    report = check(Schema, path)
    assert [(problem.path, problem.message) for problem in report.errors] == [
        ("/items/b", "values must match size")
    ]


def test_structurally_unsafe_hook_is_skipped_but_safe_sibling_runs(tmp_path: Path):
    calls: list[str] = []

    class Item(h5t.Group):
        required: int

        def validate(self) -> None:
            calls.append(self._h5t_path)

    class Schema(h5t.File):
        broken: Item
        sound: Item

        def validate(self) -> None:
            calls.append("file")

    path = tmp_path / "unsafe.h5"
    with h5py.File(path, "w") as file:
        file.create_group("broken")
        sound = file.create_group("sound")
        sound.attrs["required"] = 1

    report = check(Schema, path)
    assert not report.ok
    assert calls == ["/sound"]


def test_unexpected_hook_exception_propagates_and_open_closes_file(tmp_path: Path):
    seen: list[h5py.File] = []

    class Schema(h5t.File):
        def validate(self) -> None:
            seen.append(self._h5t_ctx.require())
            raise RuntimeError("validator bug")

    path = tmp_path / "bug.h5"
    with h5py.File(path, "w"):
        pass

    with pytest.raises(RuntimeError, match="validator bug"):
        Schema.open(path)
    assert seen and not seen[0].id.valid


def test_dataset_check_runs_its_hook(tmp_path: Path):
    class Values(h5t.Dataset, dtype=h5t.f8):
        def validate(self) -> None:
            raise h5t.Invalid("dataset failure")

    class Schema(h5t.File):
        values: Values

    path = tmp_path / "dataset.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("values", data=np.zeros(2))

    with Schema.open(path, validate=False) as view:
        report = view.values.check()
    assert [(problem.path, problem.message) for problem in report.errors] == [
        ("/values", "dataset failure")
    ]


def test_validator_may_read_payload(tmp_path: Path):
    class Values(h5t.Dataset, dtype=h5t.f8):
        def validate(self) -> None:
            if np.any(self[:] < 0):
                raise h5t.Invalid("values must be nonnegative")

    class Schema(h5t.File):
        values: Values

    path = tmp_path / "payload.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("values", data=np.array([1.0, -1.0]))

    report = check(Schema, path)
    assert [problem.message for problem in report.errors] == ["values must be nonnegative"]
