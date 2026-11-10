"""Public exception types and HDF5 path helpers."""

from __future__ import annotations


def child_path(parent: str, name: str) -> str:
    """Join an absolute group path and one immediate child name."""
    return f"/{name}" if parent == "/" else f"{parent.rstrip('/')}/{name}"


def attr_path(parent: str, name: str) -> str:
    """Return a display path for an HDF5 attribute."""
    return f"{parent.rstrip('/') or '/'}@{name}"


class H5TError(Exception):
    """Base class for every error raised by h5t."""


class SchemaError(H5TError):
    """A schema class declaration is incoherent or unsupported.

    Raised during HDF5 operations, before opening the file. Type-owned schema hooks
    may also raise this while Pydantic builds a model. Native Pydantic model-building
    and unresolved-annotation errors propagate unchanged.
    """


class ValidationError(H5TError):
    """An HDF5 node (or a group being dumped) does not conform to its schema.

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
