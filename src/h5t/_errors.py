"""Public exception types and HDF5 path helpers."""

from __future__ import annotations

from typing import Any

from h5t._h5view import ATTRS_KEY


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

    Raised while Pydantic builds the model: when the class is defined, or at the
    ``model_rebuild`` (or first use) that resolves its forward references. Native Pydantic
    model-building and unresolved-annotation errors propagate unchanged.
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


def _located_error(exc: Any, path: str, *, in_attrs: bool = False) -> ValidationError:
    """Turn the first error of a Pydantic ``ValidationError`` into one located under ``path``.

    ``path`` is the HDF5 node that was being validated. Pydantic reports locations by alias,
    which is the HDF5 name, so the parts of ``loc`` extend ``path`` directly: child names
    for a group, or one attribute name followed by a location inside that attribute's value.
    """
    first = exc.errors(include_url=False, include_context=False, include_input=False)[0]
    loc = [str(part) for part in first["loc"]]
    message = str(first["msg"])
    if loc and loc[0] == ATTRS_KEY:
        in_attrs = True
        del loc[0]
    if not in_attrs:
        for part in loc:
            path = child_path(path, part)
    elif loc:
        path = attr_path(path, loc[0])
        if loc[1:]:
            message = f"{'.'.join(loc[1:])}: {message}"
    return ValidationError(path, message)
