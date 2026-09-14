"""Compiled schema records and public annotation markers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import TypeAdapter

from h5t._array import LazyArray


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
    """Read a dataset field's complete payload during ``h5t.load`` instead of lazily."""


@dataclass(frozen=True)
class Payload:
    """Load a child dataset into a record type that carries no ``@h5t.dataset`` marker.

    ``data`` names the field of the record type holding the payload; it must be
    annotated ``np.ndarray`` (materialized eagerly) or ``LazyArray`` (read on first
    access). ``attrs``, if given, names a ``Mapping``-annotated field that receives the
    dataset's attrs snapshot -- validated values for declared names, raw for undeclared.
    ``extras`` governs the dataset's undeclared attributes. All three live at the use
    site rather than on the record type, which is the point: ``Payload`` is for a
    third-party type you cannot decorate. Prefer ``@h5t.dataset`` when you can.
    """

    data: str
    attrs: str | None = None
    extras: Literal["ignore", "forbid"] = "ignore"


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


class RecordKind(Enum):
    """Whether a decorated record describes an HDF5 dataset or a group."""

    DATASET = "dataset"
    GROUP = "group"


_NO_DEFAULT = object()


@dataclass(frozen=True)
class ClassSpec:
    """The fields and extras policy compiled for one record type."""

    fields: tuple[FieldSpec, ...] = ()
    extras: Extras = Extras.IGNORE


@dataclass(frozen=True)
class ForeignSpec:
    """Compiled loading instructions for one record type, dataset- or group-shaped."""

    record_type: type
    spec: ClassSpec
    kind: RecordKind
    data: str | None
    attrs: str | None
    lazy_type: type[LazyArray] | None
    signature: inspect.Signature | None


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
    foreign: ForeignSpec | None = None
    default_factory: Callable[[], Any] | None = None

    @property
    def has_default(self) -> bool:
        """Whether the class body supplied a default."""
        return self.default is not _NO_DEFAULT


def child_path(parent: str, name: str) -> str:
    """Join an absolute group path and one immediate child name."""
    return f"/{name}" if parent == "/" else f"{parent.rstrip('/')}/{name}"


def attr_path(parent: str, name: str) -> str:
    """Return a display path for an HDF5 attribute."""
    return f"{parent.rstrip('/') or '/'}@{name}"
