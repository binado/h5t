"""The structural spec tree compiled from schema-class annotations.

Built-in checks and view construction operate on these nodes. Each node
also retains its view type so validation can invoke user-defined hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from h5t._dtypes import DType, normalize_str

# --------------------------------------------------------------------------
# Public annotation markers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Name:
    """Map a Python field name to its canonical on-disk HDF5 name.

    Parameters
    ----------
    name : str
        The HDF5 name, which may be invalid or reserved as a Python
        identifier (e.g. ``"run-id"`` or ``"keys"``).
    """

    name: str


@dataclass(frozen=True)
class Keys:
    """Select which children of a dynamic group are validated as the item type.

    Parameters
    ----------
    pattern : str
        A regular expression; children whose names fully match are
        validated as the item schema, nonmatching children follow the
        group's extras policy.
    """

    pattern: str


class Extras(Enum):
    """Policy for undeclared children and attrs."""

    IGNORE = "ignore"
    WARN = "warn"
    FORBID = "forbid"


# --------------------------------------------------------------------------
# Attr types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AttrType:
    """The checked type of an HDF5 attribute member.

    Exactly one of :attr:`base` or :attr:`literals` is set.

    Attributes
    ----------
    base : type or None
        One of ``str``, ``int``, ``float``, ``bool``, or ``numpy.ndarray``.
    literals : tuple or None
        Closed set of allowed values (from ``Literal[...]``); values are
        normalised before membership is checked.
    """

    base: type | None = None
    literals: tuple[Any, ...] | None = None

    def describe(self) -> str:
        """Return a short human-readable description of the type."""
        if self.literals is not None:
            return "Literal[" + ", ".join(repr(v) for v in self.literals) + "]"
        assert self.base is not None
        return self.base.__name__

    def check(self, raw: object) -> tuple[bool, Any, str | None]:
        """Check and normalise a raw attribute value against this type.

        Parameters
        ----------
        raw : object
            The value as returned by h5py.

        Returns
        -------
        ok : bool
            Whether the value conforms.
        normalized : object
            The normalised Python value when ``ok``; otherwise ``None``.
        message : str or None
            A human-readable failure description when not ``ok``.
        """
        if self.literals is not None:
            expected_type = type(self.literals[0])
            ok, value, msg = AttrType(base=expected_type).check(raw)
            if not ok:
                return False, None, msg
            if value not in self.literals:
                return False, None, f"value {value!r} not in {self.describe()}"
            return True, value, None
        assert self.base is not None
        if self.base is str:
            try:
                return True, normalize_str(raw), None
            except ValueError as exc:
                return False, None, str(exc)
        if self.base is bool:
            if isinstance(raw, (bool, np.bool_)):
                return True, bool(raw), None
            return False, None, f"expected bool, got {type(raw).__name__}"
        if self.base is int:
            if isinstance(raw, (bool, np.bool_)):
                return False, None, "expected int, got bool"
            if isinstance(raw, (int, np.integer)):
                return True, int(raw), None
            return False, None, f"expected int, got {type(raw).__name__}"
        if self.base is float:
            if isinstance(raw, (bool, np.bool_)):
                return False, None, "expected float, got bool"
            if isinstance(raw, (int, float, np.integer, np.floating)):
                return True, float(raw), None
            return False, None, f"expected float, got {type(raw).__name__}"
        if self.base is np.ndarray:
            if isinstance(raw, np.ndarray):
                return True, raw, None
            return False, None, f"expected an array-valued attr, got {type(raw).__name__}"
        raise AssertionError(f"unreachable attr base type {self.base!r}")


# --------------------------------------------------------------------------
# Spec nodes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AttrSpec:
    """Spec of one HDF5 attribute member.

    Attributes
    ----------
    py_name : str
        Python field name.
    h5_name : str
        On-disk attribute name (differs from ``py_name`` under ``Name``).
    type : AttrType
        The checked attribute type.
    optional : bool
        Whether the attribute may be absent (``| None`` in the annotation).
    default : object or None
        Write-time default. Never applied on read: a missing required attr
        is a validation error even when a default exists.
    """

    py_name: str
    h5_name: str
    type: AttrType
    optional: bool = False
    default: Any | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Spec of one dataset member (or a dataset class template).

    Attributes
    ----------
    py_name : str
        Python field name; empty for a class-level template.
    h5_name : str
        On-disk dataset name; empty for a class-level template.
    dtype : type of DType or None
        Declared element dtype token; ``None`` only in incomplete
        templates, never in a compiled member.
    attrs : tuple of AttrSpec
        Attributes declared on the dataset itself.
    optional : bool
        Whether the dataset may be absent.
    view_type : type
        Class used to build the lazy view for this member.
    """

    py_name: str
    h5_name: str
    dtype: type[DType] | None
    attrs: tuple[AttrSpec, ...] = ()
    optional: bool = False
    view_type: type | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class DynamicSpec:
    """Spec of the dynamically named children of a ``Group[T]`` collection.

    Attributes
    ----------
    item : GroupSpec or DatasetSpec
        Schema every selected child must satisfy.
    pattern : str or None
        ``Keys`` regular expression selecting children; ``None`` selects
        every child.
    """

    item: GroupSpec | DatasetSpec
    pattern: str | None = None


@dataclass(frozen=True)
class GroupSpec:
    """Spec of one group member, a group class, or the file root.

    Attributes
    ----------
    py_name : str
        Python field name; empty for a class-level spec or the file root.
    h5_name : str
        On-disk group name; empty for the file root.
    children : tuple of GroupSpec or DatasetSpec
        Statically declared child nodes, in declaration order.
    attrs : tuple of AttrSpec
        Attributes declared on the group.
    extras : Extras
        Policy for undeclared children and attrs.
    dynamic : DynamicSpec or None
        Present when the group's dynamically named children must satisfy an
        item schema (``Group[T]``).
    optional : bool
        Whether the group may be absent.
    view_type : type
        Class used to build the lazy view for this member.
    """

    py_name: str
    h5_name: str
    children: tuple[GroupSpec | DatasetSpec, ...] = ()
    attrs: tuple[AttrSpec, ...] = ()
    extras: Extras = Extras.IGNORE
    dynamic: DynamicSpec | None = None
    optional: bool = False
    view_type: type | None = field(default=None, compare=False, repr=False)


MemberSpec = GroupSpec | DatasetSpec | AttrSpec
"""Any compiled class-body member."""

NodeSpec = GroupSpec | DatasetSpec
"""Any spec node that corresponds to an HDF5 object (group or dataset)."""


def child_path(parent: str, h5_name: str) -> str:
    """Join an HDF5 parent path and a child name.

    Parameters
    ----------
    parent : str
        Absolute parent path (``"/"`` for the root).
    h5_name : str
        Child name.

    Returns
    -------
    str
        The absolute child path.
    """
    if parent.endswith("/"):
        return parent + h5_name
    return f"{parent}/{h5_name}"
