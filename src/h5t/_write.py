"""Schema-validated writing of `Group` instances to HDF5, in two phases.

Phase 1 (*plan*) walks the compiled spec over an in-memory `Group` and builds a tree of
what to write, validating everything and reading every lazy `Dataset` payload. It never
touches the target file. Phase 2 (*apply*) opens the file and writes the tree, rolling
back what it wrote if anything fails.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import typing
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

import h5py
import numpy as np
from jaxtyping import jaxtyped
from pydantic_core import PydanticSerializationError

from h5t._array import _DATA_NOT_LOADED, Dataset
from h5t._compile import (
    Group,
    _check_array,
    _compiled_group,
    _filesystem_path,
    _normalize_root,
    _validate_adapter,
)
from h5t._errors import SchemaError, ValidationError
from h5t._spec import (
    ArraySpec,
    AttrsSpec,
    Extras,
    FieldSpec,
    GroupSpec,
    MemberKind,
    attr_path,
    child_path,
)


@dataclasses.dataclass
class _DatasetPlan:
    data: np.ndarray
    attrs: dict[str, Any]


@dataclasses.dataclass
class _GroupPlan:
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    children: dict[str, _GroupPlan | _DatasetPlan] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 1: plan
# ---------------------------------------------------------------------------


def _read_field(owner_type: type, record: Any, name: str) -> Any:
    try:
        return getattr(record, name)
    except AttributeError as exc:
        raise SchemaError(f"{owner_type.__name__}.{name}: model has no such attribute") from exc


def _check_storable(value: Any, path: str) -> None:
    """Reject values h5py would store as an object array, or fail on."""
    if isinstance(value, (str, bytes)):
        return
    try:
        kind = np.asarray(value).dtype.kind
    except Exception as exc:
        raise ValidationError(path, f"cannot store {type(value).__name__}: {exc}") from exc
    if kind == "O":
        raise ValidationError(
            path,
            f"cannot store {type(value).__name__} as an HDF5 attribute; add a PlainSerializer",
        )


def _plan_attrs(
    attrs_spec: AttrsSpec | None,
    attrs_val: Any,
    path: str,
    *,
    forbid_untyped_extras: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if attrs_spec is None:
        if not isinstance(attrs_val, Mapping):
            raise ValidationError(
                path, f"expected Mapping for attrs, got {type(attrs_val).__name__}"
            )
        for raw_name, value in attrs_val.items():
            name = str(raw_name)
            apath = attr_path(path, name)
            if forbid_untyped_extras:
                raise ValidationError(apath, "unexpected attribute")
            if isinstance(value, Enum):
                value = value.value
            _check_storable(value, apath)
            out[name] = value
        return out

    if not isinstance(attrs_val, attrs_spec.model_type):
        raise ValidationError(
            path,
            f"expected {attrs_spec.model_type.__name__} for attrs, got {type(attrs_val).__name__}",
        )

    extra_dict = getattr(attrs_val, "__pydantic_extra__", None) or {}
    if attrs_spec.extras is Extras.FORBID and extra_dict:
        first_extra = str(next(iter(extra_dict)))
        raise ValidationError(attr_path(path, first_extra), "unexpected attribute")

    with jaxtyped("context"):
        for field in attrs_spec.fields:
            apath = attr_path(path, field.h5_name)
            raw_val = _read_field(attrs_spec.model_type, attrs_val, field.py_name)
            if raw_val is None and field.optional:
                continue
            if field.array is not None:
                if not isinstance(raw_val, np.ndarray):
                    raise ValidationError(apath, f"expected ndarray, got {type(raw_val).__name__}")
                _check_array(field.array, raw_val, apath)
                value = raw_val
            else:
                value = _validate_adapter(field.save_adapter, raw_val, apath)
            if value is None:
                continue
            try:
                serialized = field.save_adapter.dump_python(value, mode="python")
            except (PydanticSerializationError, Exception) as exc:
                cause = (
                    exc.__cause__
                    if isinstance(exc, PydanticSerializationError) and exc.__cause__
                    else exc
                )
                raise ValidationError(apath, f"attribute serializer failed: {cause}") from cause
            if isinstance(serialized, Enum):
                serialized = serialized.value
            _check_storable(serialized, apath)
            out[field.h5_name] = serialized

    for raw_name, value in extra_dict.items():
        name = str(raw_name)
        if name in out:
            continue
        apath = attr_path(path, name)
        if isinstance(value, Enum):
            value = value.value
        _check_storable(value, apath)
        out[name] = value

    return out


def _materialize(dataset: Dataset[Any], path: str) -> np.ndarray:
    """Return a dataset's payload without mutating the caller's detached instance."""
    cached = dataset._h5t_data
    if cached is not _DATA_NOT_LOADED:
        return typing.cast(np.ndarray, cached)
    try:
        with dataset.open() as h5ds:
            return np.asarray(h5ds[()])
    except OSError as exc:
        raise ValidationError(path, f"cannot read lazy payload: {exc}") from exc


def _plan_dataset(
    value: Any,
    attrs_spec: AttrsSpec | None,
    array_spec: ArraySpec | None,
    path: str,
) -> _DatasetPlan:
    if not isinstance(value, Dataset):
        raise ValidationError(path, f"expected Dataset, got {type(value).__name__}")
    if array_spec is not None:
        _check_array(array_spec, value, path)
    attrs = _plan_attrs(attrs_spec, value.attrs, path, forbid_untyped_extras=False)
    data = _materialize(value, path)
    return _DatasetPlan(data=data, attrs=attrs)


def _plan_array(value: Any, array_spec: ArraySpec | None, path: str) -> _DatasetPlan:
    if not isinstance(value, np.ndarray):
        raise ValidationError(path, f"expected ndarray, got {type(value).__name__}")
    if array_spec is not None:
        _check_array(array_spec, value, path)
    return _DatasetPlan(data=np.asarray(value), attrs={})


def _plan_member(field: FieldSpec, value: Any, path: str) -> _GroupPlan | _DatasetPlan:
    if field.kind is MemberKind.GROUP:
        assert field.member_type is not None
        group_cls = typing.cast("type[Group[Any]]", field.member_type)
        return _plan_group(_compiled_group(group_cls), value, path)
    if field.kind is MemberKind.DATASET:
        return _plan_dataset(value, field.dataset_attrs, field.array, path)
    return _plan_array(value, field.array, path)


def _plan_collection(field: FieldSpec, value: Mapping[str, Any], path: str) -> _GroupPlan:
    assert field.item is not None
    plan = _GroupPlan()
    for key, item in value.items():
        item_path = child_path(path, str(key))
        if not isinstance(key, str) or not key or "/" in key or key in (".", ".."):
            raise ValidationError(
                item_path, "collection key must be a non-empty str without '/' and not '.' or '..'"
            )
        if field.glob is not None and field.glob.match(key) is None:
            raise ValidationError(item_path, "collection key does not match the field's Glob")
        validated = _validate_adapter(field.item.adapter, item, item_path)
        plan.children[key] = _plan_member(field.item, validated, item_path)
    return plan


def _plan_group(spec: GroupSpec, record: Any, path: str) -> _GroupPlan:
    if not isinstance(record, spec.group_type):
        raise ValidationError(
            path, f"expected {spec.group_type.__name__}, got {type(record).__name__}"
        )
    with jaxtyped("context"):
        plan = _GroupPlan()
        plan.attrs = _plan_attrs(
            spec.attrs,
            _read_field(spec.group_type, record, "attrs"),
            path,
            forbid_untyped_extras=(spec.extras is Extras.FORBID and spec.attrs is None),
        )
        for field in spec.fields:
            member_path = child_path(path, field.h5_name)
            raw_val = _read_field(spec.group_type, record, field.py_name)
            value = _validate_adapter(field.adapter, raw_val, member_path)
            if value is None:
                continue
            if field.kind is MemberKind.COLLECTION:
                plan.children[field.h5_name] = _plan_collection(field, value, member_path)
            else:
                plan.children[field.h5_name] = _plan_member(field, value, member_path)
    return plan


# ---------------------------------------------------------------------------
# Phase 2: apply
# ---------------------------------------------------------------------------


def _write_attrs(node: h5py.Group | h5py.Dataset, attrs: Mapping[str, Any], path: str) -> None:
    for name, value in attrs.items():
        try:
            node.attrs[name] = value
        except (TypeError, ValueError) as exc:
            raise ValidationError(attr_path(path, name), f"cannot write attribute: {exc}") from exc


def _write_plan(group: h5py.Group, plan: _GroupPlan, path: str) -> None:
    _write_attrs(group, plan.attrs, path)
    for name, child in plan.children.items():
        cpath = child_path(path, name)
        if isinstance(child, _GroupPlan):
            _write_plan(group.create_group(name), child, cpath)
            continue
        try:
            dataset = group.create_dataset(name, data=child.data)
        except (TypeError, ValueError) as exc:
            raise ValidationError(cpath, f"cannot write dataset: {exc}") from exc
        _write_attrs(dataset, child.attrs, cpath)


def _clear(node: h5py.Group) -> None:
    for name in list(node.keys()):
        del node[name]
    for name in list(node.attrs.keys()):
        del node.attrs[name]


@dataclasses.dataclass(frozen=True)
class _Undo:
    delete: str | None = None
    clear: str | None = None

    def rollback(self, file: h5py.File) -> None:
        with contextlib.suppress(Exception):
            if self.delete is not None:
                del file[self.delete]
            elif self.clear is not None:
                _clear(typing.cast(h5py.Group, file[self.clear]))


def _first_missing(file: h5py.File, root: str) -> str:
    current = ""
    for part in root.strip("/").split("/"):
        current = f"{current}/{part}"
        if current not in file:
            return current
    return root


def _prepare_target(file: h5py.File, root: str, overwrite: bool) -> _Undo:
    try:
        node = file.get(root)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValidationError(root, f"cannot resolve root: {exc}") from exc
    if node is not None:
        if not isinstance(node, h5py.Group):
            raise ValidationError(root, "expected a group, found a dataset")
        if len(node) or len(node.attrs):
            if not overwrite:
                raise ValidationError(root, "group already exists and is not empty")
            _clear(node)
        return _Undo(clear=root)
    first_missing = _first_missing(file, root)
    try:
        file.require_group(root)
    except (TypeError, ValueError) as exc:
        raise ValidationError(root, f"cannot create root: {exc}") from exc
    return _Undo(delete=first_missing)


def _apply(
    filename: str,
    mode: Literal["w", "w-", "a"],
    root: str,
    plan: _GroupPlan,
    overwrite: bool,
) -> None:
    created_file = mode != "a" or not os.path.exists(filename)
    h5file = h5py.File(filename, mode)
    ok = False
    undo = _Undo()
    try:
        undo = _prepare_target(h5file, root, overwrite)
        _write_plan(typing.cast(h5py.Group, h5file[root]), plan, root)
        ok = True
    finally:
        if not ok and not created_file:
            undo.rollback(h5file)
        h5file.close()
        if not ok and created_file:
            with contextlib.suppress(OSError):
                os.remove(filename)


def _dump_group(
    record: Group[Any],
    path: str | os.PathLike[str],
    root: str = "/",
    mode: Literal["w-", "x", "w", "a"] = "w-",
    overwrite: bool = False,
) -> None:
    if mode not in ("w-", "x", "w", "a"):
        raise ValueError(f"mode must be 'w-', 'x', 'w' or 'a', got {mode!r}")
    schema = type(record)
    if not issubclass(schema, Group):
        raise SchemaError(f"{schema!r} is not an h5t.Group subclass")
    spec = _compiled_group(schema)
    normalized_root = _normalize_root(root)
    filename = _filesystem_path(path)

    plan = _plan_group(spec, record, normalized_root)
    _apply(filename, "w-" if mode == "x" else mode, normalized_root, plan, overwrite)
