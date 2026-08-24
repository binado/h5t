"""Dtype tokens and the numeric compatibility policy (PLAN.md section 6).

Numeric dtypes match on *kind and itemsize*; byte order is ignored because
h5py converts endianness transparently on read. There is no width
tolerance: an ``f4`` dataset fails an ``f8`` schema.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class DType:
    """Base class for h5t dtype tokens.

    Tokens are used as types, never instantiated: ``h5t.Dataset[h5t.f8]``.

    Attributes
    ----------
    kind : str
        NumPy dtype kind character (``"f"``, ``"i"``, ``"u"``, ``"c"``).
    itemsize : int
        Width in bytes.
    """

    kind: ClassVar[str]
    itemsize: ClassVar[int]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__()
        if "kind" not in cls.__dict__ or "itemsize" not in cls.__dict__:
            raise TypeError("DType subclasses must define 'kind' and 'itemsize'")


class f4(DType):
    """32-bit IEEE float."""

    kind = "f"
    itemsize = 4


class f8(DType):
    """64-bit IEEE float."""

    kind = "f"
    itemsize = 8


class i1(DType):
    """8-bit signed integer."""

    kind = "i"
    itemsize = 1


class i2(DType):
    """16-bit signed integer."""

    kind = "i"
    itemsize = 2


class i4(DType):
    """32-bit signed integer."""

    kind = "i"
    itemsize = 4


class i8(DType):
    """64-bit signed integer."""

    kind = "i"
    itemsize = 8


class u1(DType):
    """8-bit unsigned integer."""

    kind = "u"
    itemsize = 1


class u2(DType):
    """16-bit unsigned integer."""

    kind = "u"
    itemsize = 2


class u4(DType):
    """32-bit unsigned integer."""

    kind = "u"
    itemsize = 4


class u8(DType):
    """64-bit unsigned integer."""

    kind = "u"
    itemsize = 8


class c8(DType):
    """64-bit complex (two 32-bit floats)."""

    kind = "c"
    itemsize = 8


class c16(DType):
    """128-bit complex (two 64-bit floats)."""

    kind = "c"
    itemsize = 16


def dtype_matches(token: type[DType], actual: np.dtype) -> bool:
    """Check a stored dtype against a schema dtype token.

    Parameters
    ----------
    token : type of DType
        The schema-declared dtype token.
    actual : numpy.dtype
        The dtype reported by the file's metadata.

    Returns
    -------
    bool
        True when kind and itemsize both match. Byte order is ignored; no
        width tolerance is applied.
    """
    return actual.kind == token.kind and actual.itemsize == token.itemsize


def normalize_str(value: object) -> str:
    """Normalise any HDF5 string flavour to a Python ``str``.

    Parameters
    ----------
    value : object
        A value read from an HDF5 attribute: ``str``, ``bytes``,
        ``numpy.str_``, or ``numpy.bytes_``.

    Returns
    -------
    str
        The decoded string.

    Raises
    ------
    ValueError
        If ``value`` is not string-like, or is bytes that do not decode as
        UTF-8. An undecodable attr is an error, never a silent ``bytes``
        passthrough.
    """
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"attr bytes are not valid UTF-8: {value!r}") from exc
    raise ValueError(f"expected a string value, got {type(value).__name__}")
