"""Detached, typed records for reading and writing HDF5 groups."""

from h5t._array import LazyArray
from h5t._compile import dataset, group, load
from h5t._errors import ConversionError, H5TError, SchemaError, ValidationError
from h5t._spec import Attr, Eager, Glob, Name, Payload
from h5t._write import save

__version__ = "0.3.0"

__all__ = [
    "Attr",
    "ConversionError",
    "Eager",
    "Glob",
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
    "save",
]
