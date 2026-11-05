"""Public exception types for schema compilation and detached loading."""

from __future__ import annotations


class H5TError(Exception):
    """Base class for every error raised by h5t."""


class SchemaError(H5TError):
    """A schema class declaration is incoherent or unsupported.

    Usually raised right where the class is decorated.
    """


class ValidationError(H5TError):
    """An HDF5 node (or a record being saved) does not conform to its schema.

    The message starts with where the problem is in the file.

    Parameters
    ----------
    path : str
        HDF5 path of the offending node, e.g. ``"/fiducials"`` or ``"/@version"``.
    message : str
        What is wrong there.

    Attributes
    ----------
    path : str
        HDF5 path of the offending node.
    message : str
        The message without the path prefix.
    """

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


class ConversionError(ValidationError):
    """An ``Attr`` converter or serializer raised.

    A subclass of `ValidationError`, so it carries the same ``path`` and ``message``.
    """
