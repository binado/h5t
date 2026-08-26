"""Compiled schema records and public annotation markers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import TypeAdapter


@dataclass(frozen=True)
class Name:
    """Use ``name`` for this member in the HDF5 file."""

    name: str


@dataclass(frozen=True)
class Attr:
    """Declare a field as an HDF5 attribute.

    ``converter`` is applied to the value returned by h5py before Pydantic
    validation. It is useful for serialized attributes such as JSON strings.
    """

    converter: Callable[[Any], Any] | None = None


@dataclass(frozen=True)
class Eager:
    """Load a detached dataset's complete payload during ``from_file``."""


class Extras(Enum):
    """Policy for undeclared immediate HDF5 members."""

    IGNORE = "ignore"
    FORBID = "forbid"


class MemberKind(Enum):
    """How a declared field is represented in HDF5."""

    ATTRIBUTE = "attribute"
    ARRAY = "array"
    DATASET = "dataset"
    GROUP = "group"


_NO_DEFAULT = object()


@dataclass(frozen=True)
class FieldSpec:
    """The compiled loading instructions for one annotated field."""

    py_name: str
    h5_name: str
    kind: MemberKind
    annotation: Any
    adapter: TypeAdapter[Any] = field(compare=False, repr=False)
    optional: bool = False
    default: Any = _NO_DEFAULT
    converter: Callable[[Any], Any] | None = None
    eager: bool = False
    member_type: type | None = None

    @property
    def has_default(self) -> bool:
        """Whether the class body supplied a default."""
        return self.default is not _NO_DEFAULT


@dataclass(frozen=True)
class ClassSpec:
    """The flattened schema compiled for a ``Group`` or ``Dataset`` class."""

    fields: tuple[FieldSpec, ...] = ()
    extras: Extras = Extras.IGNORE


def child_path(parent: str, name: str) -> str:
    """Join an absolute group path and one immediate child name."""
    return f"/{name}" if parent == "/" else f"{parent.rstrip('/')}/{name}"


def attr_path(parent: str, name: str) -> str:
    """Return a display path for an HDF5 attribute."""
    return f"{parent.rstrip('/') or '/'}@{name}"
