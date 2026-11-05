"""Compiled schema records and public annotation markers."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import TypeAdapter

from h5t._array import LazyArray


@dataclass(frozen=True)
class Name:
    """Store a member under a different name in the HDF5 file.

    Use it inside ``Annotated`` when the name in the file differs from the field name,
    for example because it isn't a valid Python identifier.

    Parameters
    ----------
    name : str
        The name of the attribute, dataset or group in the file.

    Examples
    --------
    >>> from typing import Annotated
    >>> import dataclasses, h5t
    >>> @h5t.group()
    ... @dataclasses.dataclass
    ... class Run:
    ...     title: Annotated[str, h5t.Name("run title")]
    """

    name: str


@dataclass(frozen=True)
class Attr:
    """Declare a field as an HDF5 attribute, with optional conversion hooks.

    Plain fields such as ``int`` or ``str`` are already read as attributes. Use ``Attr``
    when the stored value needs converting (e.g. a JSON string), or when an array should
    come from an attribute rather than a dataset.

    Parameters
    ----------
    converter : callable, optional
        Applied to the value read from the file, before validation. If it fails you get a
        `ConversionError`.
    serializer : callable, optional
        The reverse of ``converter``, used by `save` just before writing, e.g.
        ``json.dumps``.

    Examples
    --------
    >>> import json
    >>> from typing import Annotated, Any
    >>> import dataclasses, h5t
    >>> @h5t.group()
    ... @dataclasses.dataclass
    ... class Run:
    ...     config: Annotated[
    ...         dict[str, Any], h5t.Attr(converter=json.loads, serializer=json.dumps)
    ...     ]
    """

    converter: Callable[[Any], Any] | None = None
    serializer: Callable[[Any], Any] | None = None


@dataclass(frozen=True)
class Eager:
    """Read a dataset field's complete payload during `load` instead of lazily.

    Use it on a dataset field typed `LazyArray` when you want the data ready straight
    away. (An ``np.ndarray`` field is always read at load time.)

    Examples
    --------
    >>> from typing import Annotated
    >>> import dataclasses, h5t
    >>> @h5t.group()
    ... @dataclasses.dataclass
    ... class Run:
    ...     samples: Annotated[h5t.LazyArray, h5t.Eager()]
    """


@dataclass(frozen=True)
class Payload:
    """Load a child dataset into a record type that carries no ``@h5t.dataset`` marker.

    Use it for a class you can't decorate, such as one from another library. Prefer
    `dataset` when the class is yours.

    Parameters
    ----------
    data : str
        The field that receives the data. Annotate it ``np.ndarray`` (read now) or
        `LazyArray` (read later).
    attrs : str, optional
        A ``Mapping``-annotated field that receives all of the dataset's attributes.
    extras : {"ignore", "forbid"}
        Whether to ignore (default) or reject attributes the class doesn't declare.

    Examples
    --------
    >>> from typing import Annotated
    >>> import dataclasses, numpy as np, h5t
    >>> @dataclasses.dataclass
    ... class Measurement:
    ...     unit: str
    ...     data: np.ndarray
    >>> @h5t.group()
    ... @dataclasses.dataclass
    ... class Result:
    ...     measurement: Annotated[Measurement, h5t.Payload("data")]
    """

    data: str
    attrs: str | None = None
    extras: Literal["ignore", "forbid"] = "ignore"


@dataclass(frozen=True)
class Glob:
    """Filter which children of a collection field's group are loaded.

    Valid only on a ``dict[str, T]`` / ``Mapping[str, T]`` field.

    Parameters
    ----------
    pattern : str
        A shell-style pattern (``fnmatch.fnmatchcase``: ``*``, ``?``, ``[seq]``),
        matched case-sensitively against each child's name. It may not be empty or
        contain ``/``. Children that do not match are skipped.

    Examples
    --------
    >>> from typing import Annotated
    >>> import dataclasses, h5t
    >>> @h5t.group()
    ... @dataclasses.dataclass
    ... class Run:
    ...     chains: Annotated[dict[str, h5t.LazyArray], h5t.Glob("chain_?")]
    """

    pattern: str = "*"


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
    COLLECTION = "collection"


class RecordKind(Enum):
    """Whether a decorated record describes an HDF5 dataset or a group."""

    DATASET = "dataset"
    GROUP = "group"


_NO_DEFAULT = object()


@dataclass(frozen=True)
class ArraySpec:
    """A jaxtyping array annotation, kept whole so jaxtyping does the checking.

    ``hint`` is the jaxtyping class itself (``Float[np.ndarray, "n n"]``); it is excluded
    from comparison because jaxtyping builds a fresh class per subscript, so two
    identical annotations would otherwise never compare equal. ``family`` (``Float``) and
    ``dim_str`` carry the identity.
    """

    hint: type = field(compare=False, repr=False)
    family: type
    dim_str: str


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
    payload_array: ArraySpec | None = None


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
    serializer: Callable[[Any], Any] | None = None
    eager: bool = False
    member_type: type | None = None
    foreign: ForeignSpec | None = None
    default_factory: Callable[[], Any] | None = None
    array: ArraySpec | None = None
    item: FieldSpec | None = None
    glob: re.Pattern[str] | None = field(default=None, compare=False)

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
