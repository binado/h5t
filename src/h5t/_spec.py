"""Compiled schema specifications and public annotation markers."""

from __future__ import annotations

import re
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, TypeAdapter


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
    >>> import h5t
    >>> class Run(h5t.Group):
    ...     chains: Annotated[dict[str, h5t.Dataset], h5t.Glob("chain_?")]
    """

    pattern: str = "*"


class Extras(Enum):
    """Policy for undeclared immediate HDF5 members or attributes."""

    IGNORE = "ignore"
    FORBID = "forbid"


class MemberKind(Enum):
    """How a declared child field on a `Group` is represented in HDF5."""

    ARRAY = "array"
    DATASET = "dataset"
    GROUP = "group"
    COLLECTION = "collection"


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
class AttrFieldSpec:
    """Compiled loading instructions for one field of a `BaseModel` attributes schema."""

    py_name: str
    h5_name: str
    adapter: TypeAdapter[Any] = field(compare=False, repr=False)
    save_adapter: TypeAdapter[Any] = field(compare=False, repr=False)
    optional: bool = False
    default: Any = _NO_DEFAULT
    default_factory: Callable[[], Any] | None = None
    array: ArraySpec | None = None

    @property
    def has_default(self) -> bool:
        """Whether the field declared a default value."""
        return self.default is not _NO_DEFAULT


@dataclass(frozen=True)
class AttrsSpec:
    """Compiled loading instructions for a `BaseModel` attributes schema ``T``."""

    model_type_ref: weakref.ReferenceType[type[BaseModel]] = field(compare=False, repr=False)
    fields: tuple[AttrFieldSpec, ...] = ()
    extras: Extras = Extras.IGNORE

    @property
    def model_type(self) -> type[BaseModel]:
        """The `BaseModel` subclass for this attributes spec."""
        cls = self.model_type_ref()
        assert cls is not None
        return cls


@dataclass(frozen=True)
class FieldSpec:
    """The compiled loading instructions for one child field of a `Group`."""

    py_name: str
    h5_name: str
    kind: MemberKind
    annotation: Any
    adapter: TypeAdapter[Any] = field(compare=False, repr=False)
    optional: bool = False
    default: Any = _NO_DEFAULT
    default_factory: Callable[[], Any] | None = None
    member_type: type | None = None
    dataset_attrs: AttrsSpec | None = None
    array: ArraySpec | None = None
    item: FieldSpec | None = None
    glob: re.Pattern[str] | None = field(default=None, compare=False)

    @property
    def has_default(self) -> bool:
        """Whether the class body supplied a default."""
        return self.default is not _NO_DEFAULT


@dataclass(frozen=True)
class GroupSpec:
    """Compiled loading instructions for a `Group` subclass."""

    group_type: type[BaseModel]
    attrs: AttrsSpec | None
    fields: tuple[FieldSpec, ...] = ()
    extras: Extras = Extras.IGNORE


def child_path(parent: str, name: str) -> str:
    """Join an absolute group path and one immediate child name."""
    return f"/{name}" if parent == "/" else f"{parent.rstrip('/')}/{name}"


def attr_path(parent: str, name: str) -> str:
    """Return a display path for an HDF5 attribute."""
    return f"{parent.rstrip('/') or '/'}@{name}"
