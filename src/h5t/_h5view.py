"""A lazy, read-only ``Mapping`` view of an open ``h5py.Group`` for single-pass loading."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, NoReturn

import h5py
import numpy as np

# Not a legal child name (it contains ``/``), so it can never collide with a member.
ATTRS_KEY = "/attrs"


class _FrozenAttrs(dict[str, Any]):
    """Untyped HDF5 attributes: a ``dict`` that cannot be modified.

    Being a real ``dict`` that owns its items (unlike ``types.MappingProxyType``, a view),
    it can be copied, deep-copied, pickled and dumped by Pydantic.
    """

    __slots__ = ()

    def _read_only(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("untyped attrs are read-only; build a new Group or Dataset instead")

    __setitem__ = __delitem__ = __ior__ = _read_only
    clear = pop = popitem = setdefault = update = _read_only

    def __reduce__(self) -> tuple[type[_FrozenAttrs], tuple[dict[str, Any]]]:
        # The default protocol would refill the copy through ``__setitem__``.
        return type(self), (dict(self),)


def _normalized_attrs(node: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    """Return ``node``'s attributes with numpy scalars and strings turned into Python values."""
    out: dict[str, Any] = {}
    for name in node.attrs:
        value = node.attrs[name]
        if isinstance(value, (bytes, np.bytes_)):
            value = bytes(value)
        elif isinstance(value, np.str_):
            value = str(value)
        elif isinstance(value, np.generic) and not isinstance(value, np.ndarray):
            value = value.item()
        out[name] = value
    return out


def _ghost(node: h5py.Dataset) -> np.ndarray:
    """Return an ndarray with ``node``'s shape and dtype that allocates one scalar only."""
    return np.broadcast_to(np.empty((), node.dtype), node.shape)


class _GroupView(Mapping[str, Any]):
    """The children of an open ``h5py.Group``, presented as a mapping Pydantic can validate.

    Child groups come back as nested views and child datasets as the raw ``h5py.Dataset``
    (nothing is read); the group's attributes live under `ATTRS_KEY`. Each *type* then
    converts what it receives, so one ``model_validate`` call loads the whole tree.
    """

    __slots__ = ("node",)

    def __init__(self, node: h5py.Group) -> None:
        self.node = node

    def __getitem__(self, key: str) -> Any:
        if key == ATTRS_KEY:
            return _normalized_attrs(self.node)
        if "/" in key:
            raise KeyError(key)
        child = self.node[key]  # KeyError when missing
        if isinstance(child, h5py.Group):
            return _GroupView(child)
        return child

    def __iter__(self) -> Iterator[str]:
        return iter(self.node.keys())

    def __len__(self) -> int:
        return len(self.node)

    def __contains__(self, key: object) -> bool:
        if key == ATTRS_KEY:
            return True
        return isinstance(key, str) and "/" not in key and key in self.node
