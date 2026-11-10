"""Detached, typed Pydantic models for reading and writing HDF5 groups."""

from h5t._array import Dataset
from h5t._compile import Group
from h5t._errors import H5TError, SchemaError, ValidationError

__version__ = "0.4.0"

__all__ = [
    "Dataset",
    "Group",
    "H5TError",
    "SchemaError",
    "ValidationError",
    "__version__",
]
