"""Public exception types for schema compilation and detached loading."""

from __future__ import annotations


class H5TError(Exception):
    """Base class for h5t errors."""


class SchemaError(H5TError):
    """A schema class declaration is incoherent or unsupported."""


class ValidationError(H5TError):
    """An HDF5 node does not conform to a compiled schema."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


class ConversionError(ValidationError):
    """An explicit ``Attr`` converter failed."""
