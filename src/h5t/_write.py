"""Writing `Group` instances to HDF5.

Pydantic validated the `Group` when it was constructed (field types, jaxtyping shapes and
dtypes, attribute models), and its declarations were checked when the class was built, so
the writer checks neither again. It walks the instance once, creating each node as it goes
and checking only what HDF5 itself imposes: child names and storable attribute values.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Any, Literal

import h5py
import numpy as np
from pydantic import BaseModel
from pydantic_core import PydanticSerializationError

from h5t._array import _DATA_NOT_LOADED, Dataset
from h5t._compile import (
    Group,
    _check_attrs,
    _filesystem_path,
    _h5_name,
    _normalize_root,
    _unwrap,
)
from h5t._errors import ValidationError, attr_path, child_path

_H5PY_ERRORS = (TypeError, ValueError, OSError)


def _children(group: Group[Any]) -> Iterator[tuple[str, Any]]:
    """Yield the on-disk name and value of each child that is to be written."""
    for name, field in type(group).model_fields.items():
        if name == "attrs":
            continue
        value = getattr(group, name)
        if value is None and _unwrap(field.annotation)[1]:
            continue
        yield _h5_name(type(group), name, field), value


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def _attr_values(attrs: Any, path: str) -> Mapping[str, Any]:
    """Return the attributes to store, by their on-disk names."""
    if not isinstance(attrs, BaseModel):
        return attrs
    # The instance may be of a model no group declared (a subclass, or the attributes of
    # an untyped ``Dataset``), which no schema build has checked.
    _check_attrs(type(attrs))
    fields = type(attrs).model_fields
    try:
        # ``include`` limits the dump to declared fields, so computed fields are not stored.
        dumped = attrs.model_dump(mode="python", by_alias=True, include=set(fields))
    except (PydanticSerializationError, TypeError) as exc:
        cause = (
            exc.__cause__ if isinstance(exc, PydanticSerializationError) and exc.__cause__ else exc
        )
        raise ValidationError(path, f"attribute serializer failed: {cause}") from cause
    # A nullable attribute holding ``None`` is simply absent from the file.
    nullable = {
        field.serialization_alias or field.alias or name
        for name, field in fields.items()
        if _unwrap(field.annotation)[1]
    }
    return {
        name: value for name, value in dumped.items() if value is not None or name not in nullable
    }


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


def _write_attrs(node: h5py.Group | h5py.Dataset, attrs: Any) -> None:
    for raw_name, value in _attr_values(attrs, node.name).items():
        name = str(raw_name)
        path = attr_path(node.name, name)
        if isinstance(value, Enum):
            value = value.value
        _check_storable(value, path)
        try:
            node.attrs[name] = value
        except _H5PY_ERRORS as exc:
            raise ValidationError(path, f"cannot write attribute: {exc}") from exc


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _create_group(parent: h5py.Group, name: str, path: str) -> h5py.Group:
    try:
        return parent.create_group(name)
    except _H5PY_ERRORS as exc:
        raise ValidationError(path, f"cannot write group: {exc}") from exc


def _create_dataset(parent: h5py.Group, name: str, path: str, data: np.ndarray) -> h5py.Dataset:
    try:
        return parent.create_dataset(name, data=data)
    except _H5PY_ERRORS as exc:
        raise ValidationError(path, f"cannot write dataset: {exc}") from exc


def _read_lazy(dataset: Dataset[Any], path: str) -> np.ndarray:
    """Return a dataset's array without caching it on the caller's detached instance."""
    if dataset._h5t_data is not _DATA_NOT_LOADED:
        return dataset.data
    try:
        with dataset.open() as h5ds:
            return np.asarray(h5ds[()])
    except (ValidationError, OSError) as exc:
        detail = exc.message if isinstance(exc, ValidationError) else str(exc)
        raise ValidationError(path, f"cannot read dataset data: {detail}") from exc


def _write_member(parent: h5py.Group, name: str, value: Any) -> None:
    path = child_path(parent.name, name)
    if isinstance(value, Group):
        _write_group(_create_group(parent, name, path), value)
    elif isinstance(value, Mapping):
        node = _create_group(parent, name, path)
        for key, item in value.items():
            if not isinstance(key, str) or not key or "/" in key or key in (".", ".."):
                raise ValidationError(
                    child_path(path, str(key)),
                    "collection key must be a non-empty str without '/' and not '.' or '..'",
                )
            _write_member(node, key, item)
    elif isinstance(value, Dataset):
        node = _create_dataset(parent, name, path, _read_lazy(value, path))
        _write_attrs(node, value.attrs)
    else:
        _create_dataset(parent, name, path, np.asarray(value))


def _write_group(node: h5py.Group, group: Group[Any]) -> None:
    _write_attrs(node, group.attrs)
    for name, value in _children(group):
        _write_member(node, name, value)


# ---------------------------------------------------------------------------
# Target file
# ---------------------------------------------------------------------------


def _lazy_datasets(value: Any) -> Iterator[Dataset[Any]]:
    """Yield every `Dataset` reachable from ``value`` whose data is not in memory yet."""
    if isinstance(value, Dataset):
        if value._h5t_data is _DATA_NOT_LOADED:
            yield value
    elif isinstance(value, Group):
        for _, child in _children(value):
            yield from _lazy_datasets(child)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _lazy_datasets(item)


def _prepare_target(file: h5py.File, root: str, overwrite: bool) -> h5py.Group:
    """Return ``root`` ready to write into, clearing it first when ``overwrite`` is set."""
    try:
        node = file.get(root)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValidationError(root, f"cannot resolve root: {exc}") from exc
    if node is not None:
        if not isinstance(node, h5py.Group):
            raise ValidationError(root, "expected a group, found a dataset")
        if (len(node) or len(node.attrs)) and not overwrite:
            raise ValidationError(root, "group already exists and is not empty")
        if overwrite:
            node.clear()
            node.attrs.clear()
        return node
    try:
        return file.require_group(root)
    except _H5PY_ERRORS as exc:
        raise ValidationError(root, f"cannot create root: {exc}") from exc


def _dump_group(
    group: Group[Any],
    path: str | os.PathLike[str],
    root: str = "/",
    mode: Literal["w-", "x", "w", "a"] = "w-",
    overwrite: bool = False,
) -> None:
    if mode not in ("w-", "x", "w", "a"):
        raise ValueError(f"mode must be 'w-', 'x', 'w' or 'a', got {mode!r}")
    normalized_root = _normalize_root(root)
    filename = _filesystem_path(path)

    # A lazy dataset that lives in the target file must be read before that file is opened
    # for writing (and possibly truncated). ``.data`` caches it on the instance.
    for dataset in _lazy_datasets(group):
        if dataset.filename is not None and os.path.realpath(dataset.filename) == os.path.realpath(
            filename
        ):
            _ = dataset.data

    with h5py.File(filename, "w-" if mode == "x" else mode) as h5file:
        _write_group(_prepare_target(h5file, normalized_root, overwrite), group)
