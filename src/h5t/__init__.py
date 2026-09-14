"""Detached, typed records for read-only HDF5 access."""

from h5t._array import LazyArray
from h5t._compile import dataset, group, load
from h5t._errors import ConversionError, H5TError, SchemaError, ValidationError
from h5t._spec import Attr, Eager, Name, Payload

__version__ = "0.2.0"

__all__ = [
    "Attr",
    "ConversionError",
    "Eager",
    "H5TError",
    "LazyArray",
    "Name",
    "Payload",
    "SchemaError",
    "ValidationError",
    "__version__",
    "dataset",
    "group",
    "load",
]
