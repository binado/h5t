"""Lazy runtime views over an open HDF5 file.

A schema instance is a *view*: it retains a path and a shared file handle,
never an in-memory materialisation of the node's datasets. Datasets are
lazy (read on slice), attrs are eager (normalised on access), and every
view shares one :class:`FileContext` whose lifetime the file's context
manager owns.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, KeysView
from typing import Any, cast

import h5py
import numpy as np

from h5t._errors import (
    ClosedFileError,
    SchemaMismatchError,
    ValidationError,
    ValidationReport,
)
from h5t._spec import AttrSpec, DatasetSpec, GroupSpec, NodeSpec, child_path
from h5t._validate import run_validation


class FileContext:
    """Shared ownership of one open ``h5py.File`` handle.

    Every view produced from one ``open()`` call holds a reference to the
    same context; closing it invalidates all of them at once.

    Parameters
    ----------
    h5file : h5py.File
        The open (read-only) handle.
    filename : str
        Display name used in error messages and reports.
    """

    def __init__(self, h5file: h5py.File, filename: str) -> None:
        self._h5file: h5py.File | None = h5file
        self.filename = filename

    @property
    def closed(self) -> bool:
        """Whether the handle has been closed."""
        return self._h5file is None

    def close(self) -> None:
        """Close the handle; idempotent."""
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def require(self) -> h5py.File:
        """Return the live handle.

        Returns
        -------
        h5py.File
            The open handle.

        Raises
        ------
        ClosedFileError
            If the owning context manager has already exited.
        """
        if self._h5file is None:
            raise ClosedFileError(
                f"the file '{self.filename}' is closed; views do not outlive"
                " their owning context manager"
            )
        return self._h5file


class ViewBase:
    """Plumbing common to group and dataset views.

    Views are constructed internally via :func:`make_view`; user code never
    instantiates schema classes directly in v0.1 (``open()`` is read-only).
    """

    _h5t_spec: NodeSpec
    _h5t_ctx: FileContext
    _h5t_path: str

    def _h5t_node(self) -> h5py.Group | h5py.Dataset:
        """Return the underlying h5py object for this view's path."""
        h5file = self._h5t_ctx.require()
        if self._h5t_path == "/":
            return h5file
        node = h5file.get(self._h5t_path)
        if node is None:
            raise SchemaMismatchError(self._h5t_path, "node no longer exists in the file")
        if not isinstance(node, (h5py.Group, h5py.Dataset)):
            raise SchemaMismatchError(
                self._h5t_path, f"unsupported HDF5 object of type {type(node).__name__}"
            )
        return node

    def __repr__(self) -> str:
        state = "closed" if self._h5t_ctx.closed else "open"
        return (
            f"<{type(self).__name__} view of '{self._h5t_path}'"
            f" in '{self._h5t_ctx.filename}' ({state})>"
        )


def make_view(spec: NodeSpec, ctx: FileContext, path: str) -> Any:
    """Build a typed view instance for a spec node at a file location.

    Parameters
    ----------
    spec : GroupSpec or DatasetSpec
        The compiled spec of the node; its ``view_type`` names the class to
        instantiate.
    ctx : FileContext
        Shared file context.
    path : str
        Absolute HDF5 path of the node.

    Returns
    -------
    ViewBase
        An instance of ``spec.view_type`` bound to ``(ctx, path)``.
    """
    view_type = spec.view_type
    assert view_type is not None, "compiled specs always carry a view type"
    view: Any = object.__new__(view_type)
    view._h5t_spec = spec
    view._h5t_ctx = ctx
    view._h5t_path = path
    return view


class GroupViewOps(ViewBase):
    """Mapping semantics and validation entry points for group views.

    The mapping API preserves HDF5's two namespaces: ``g["name"]``
    addresses a child and ``g.attrs["name"]`` addresses an attr. Bracket
    access is deliberately untyped except on dynamic ``Group[T]``
    collections, where matching keys return typed item views.
    """

    def _h5t_group(self) -> h5py.Group:
        """Return the underlying h5py group, checking the node kind."""
        node = self._h5t_node()
        if not isinstance(node, h5py.Group):
            raise SchemaMismatchError(self._h5t_path, "expected a group, found a dataset")
        return node

    @property
    def attrs(self) -> h5py.AttributeManager:
        """Raw attribute namespace of this group (h5py semantics)."""
        return self._h5t_group().attrs

    def keys(self) -> KeysView[str]:
        """Names of all children of this group."""
        return cast(KeysView[str], self._h5t_group().keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._h5t_group())

    def __contains__(self, key: str) -> bool:
        return key in self._h5t_group()

    def _h5t_getitem(self, key: str) -> Any:
        """Untyped child access; typed item views on dynamic collections."""
        group = self._h5t_group()
        spec = self._h5t_spec
        assert isinstance(spec, GroupSpec)
        dynamic = spec.dynamic
        if dynamic is not None and _dynamic_match(dynamic.pattern, key):
            if key not in group:
                raise SchemaMismatchError(
                    child_path(self._h5t_path, key),
                    f"no such child in '{self._h5t_path}'",
                )
            return make_view(dynamic.item, self._h5t_ctx, child_path(self._h5t_path, key))
        return group[key]

    def validate(self) -> None:
        """Validate the file subtree rooted at this view; raise on problems.

        Raises
        ------
        ValidationError
            Batching every problem found in the walk. The file handle is
            left open (it is owned by the enclosing context manager).
        """
        report = self.check()
        if not report.ok:
            raise ValidationError(report, file_closed=False)

    def check(self) -> ValidationReport:
        """Run the same walk as :meth:`validate` but return the report.

        Returns
        -------
        ValidationReport
            All findings; empty when the subtree conforms.
        """
        spec = self._h5t_spec
        assert isinstance(spec, GroupSpec)
        return run_validation(
            spec,
            self._h5t_group(),
            filename=self._h5t_ctx.filename,
            schema_name=type(self).__name__,
            base_path=self._h5t_path,
        )


def _dynamic_match(pattern: str | None, key: str) -> bool:
    """Whether a child key is selected by a dynamic group's Keys pattern."""
    return pattern is None or re.fullmatch(pattern, key) is not None


class DatasetViewOps(ViewBase):
    """Lazy dataset view: metadata is exposed, payload reads happen on slice."""

    def _h5t_dataset(self) -> h5py.Dataset:
        """Return the underlying h5py dataset, checking the node kind."""
        node = self._h5t_node()
        if not isinstance(node, h5py.Dataset):
            raise SchemaMismatchError(self._h5t_path, "expected a dataset, found a group")
        return node

    @property
    def shape(self) -> tuple[int, ...]:
        """Stored shape, from metadata only."""
        return cast("tuple[int, ...]", self._h5t_dataset().shape)

    @property
    def dtype(self) -> np.dtype:
        """Stored dtype, from metadata only."""
        return cast(np.dtype, self._h5t_dataset().dtype)

    @property
    def ndim(self) -> int:
        """Stored rank, from metadata only."""
        return cast(int, self._h5t_dataset().ndim)

    @property
    def attrs(self) -> h5py.AttributeManager:
        """Raw attribute namespace of this dataset (h5py semantics)."""
        return self._h5t_dataset().attrs

    def __len__(self) -> int:
        return len(self._h5t_dataset())

    def __getitem__(self, key: Any) -> Any:
        """Read data; this is the only payload-reading operation on a view."""
        return self._h5t_dataset()[key]


class _MemberDescriptor:
    """Base for the non-data descriptors installed per annotated member.

    The first access resolves and caches the value in the instance
    ``__dict__``; because these descriptors define no ``__set__``,
    subsequent lookups hit the cache directly.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        value = self._resolve(obj)
        obj.__dict__[self._name] = value
        return value

    def _resolve(self, obj: ViewBase) -> Any:
        raise NotImplementedError


class AttrDescriptor(_MemberDescriptor):
    """Eager, normalised access to one declared attribute.

    Parameters
    ----------
    spec : AttrSpec
        The compiled attr member.
    """

    def __init__(self, spec: AttrSpec) -> None:
        super().__init__(spec.py_name)
        self._spec = spec

    def _resolve(self, obj: ViewBase) -> Any:
        node = obj._h5t_node()
        spec = self._spec
        if spec.h5_name not in node.attrs:
            if spec.optional:
                return None
            message = f"required attr '{spec.h5_name}' missing"
            if isinstance(node, h5py.Group) and spec.h5_name in node:
                kind = "dataset" if isinstance(node[spec.h5_name], h5py.Dataset) else "group"
                message += (
                    f" \u2014 but a {kind} '{child_path(obj._h5t_path, spec.h5_name)}' exists here."
                )
            raise SchemaMismatchError(obj._h5t_path, message)
        raw = node.attrs[spec.h5_name]
        ok, value, msg = spec.type.check(raw)
        if not ok:
            raise SchemaMismatchError(obj._h5t_path, f"attr '{spec.h5_name}': {msg}")
        return value


class NodeDescriptor(_MemberDescriptor):
    """Lazy access to one declared child node (dataset or subgroup).

    Parameters
    ----------
    spec : GroupSpec or DatasetSpec
        The compiled child member.
    """

    def __init__(self, spec: NodeSpec) -> None:
        super().__init__(spec.py_name)
        self._spec = spec

    def _resolve(self, obj: ViewBase) -> Any:
        spec = self._spec
        parent = obj._h5t_node()
        if not isinstance(parent, h5py.Group):
            raise SchemaMismatchError(obj._h5t_path, "expected a group, found a dataset")
        cpath = child_path(obj._h5t_path, spec.h5_name)
        node = parent.get(spec.h5_name)
        expected = "dataset" if isinstance(spec, DatasetSpec) else "group"
        if node is None:
            if spec.optional:
                return None
            message = f"required {expected} '{spec.h5_name}' missing"
            if spec.h5_name in parent.attrs:
                message += f" \u2014 but an attr '{spec.h5_name}' exists here."
            raise SchemaMismatchError(obj._h5t_path, message)
        if isinstance(node, h5py.Dataset):
            actual = "dataset"
        elif isinstance(node, h5py.Group):
            actual = "group"
        else:
            actual = type(node).__name__
        if actual != expected:
            raise SchemaMismatchError(cpath, f"expected a {expected}, found a {actual}")
        return make_view(spec, obj._h5t_ctx, cpath)
