"""The minimal shape grammar (PLAN.md section 4).

::

    dimension := NAME | INTEGER | _
    shape     := "" | dimension (" " dimension)*

There are no variadic axes and no arithmetic expressions. ``""`` denotes an
HDF5 scalar dataspace; ``_`` accepts any axis length without binding a name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import assert_never

from h5t._errors import SchemaError

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INT_RE = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class NamedDim:
    """A named dimension variable; equal names must unify within a scope.

    Attributes
    ----------
    name : str
        The dimension variable name, a valid Python identifier.
    """

    name: str


@dataclass(frozen=True)
class FixedDim:
    """An axis constrained to one exact integer length.

    Attributes
    ----------
    size : int
        The required axis length.
    """

    size: int


@dataclass(frozen=True)
class AnyDim:
    """An anonymous axis (``_``): any length, no binding."""


Dim = NamedDim | FixedDim | AnyDim
Shape = tuple[Dim, ...]


def parse_shape(text: str) -> Shape:
    """Parse a shape string into a tuple of dimension terms.

    Parameters
    ----------
    text : str
        A shape string in the grammar above, e.g. ``"n_samples 3"`` or
        ``""`` for a scalar dataspace.

    Returns
    -------
    Shape
        One :class:`NamedDim`, :class:`FixedDim`, or :class:`AnyDim` per
        axis; the empty tuple for a scalar.

    Raises
    ------
    SchemaError
        If ``text`` is not a string or contains a token outside the
        grammar (including empty tokens from stray spaces).
    """
    if not isinstance(text, str):
        raise SchemaError(f"shape must be a string, got {text!r}")
    if text == "":
        return ()
    dims: list[Dim] = []
    for token in text.split(" "):
        if token == "_":
            dims.append(AnyDim())
        elif _INT_RE.match(token):
            dims.append(FixedDim(int(token)))
        elif _NAME_RE.match(token):
            dims.append(NamedDim(token))
        elif token == "":
            raise SchemaError(
                f"malformed shape string {text!r}: dimensions are separated by single spaces"
            )
        else:
            raise SchemaError(
                f"malformed shape string {text!r}: {token!r} is not a NAME, INTEGER, or '_'"
            )
    return tuple(dims)


def format_shape(shape: Shape) -> str:
    """Render a parsed shape back to its string form.

    Parameters
    ----------
    shape : Shape
        A parsed shape tuple.

    Returns
    -------
    str
        The canonical shape string, e.g. ``"n_samples 3"``.
    """
    parts: list[str] = []
    for dim in shape:
        match dim:
            case NamedDim(name=name):
                parts.append(name)
            case FixedDim(size=size):
                parts.append(str(size))
            case AnyDim():
                parts.append("_")
            case _:
                assert_never(dim)
    return " ".join(parts)
