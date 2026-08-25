"""h5t: a schema layer for HDF5.

Declare your file format as a Python class, validate its structure and
domain invariants, and share the class so downstream users load files
correctly.

Examples
--------
>>> import h5t
>>> class Posterior(h5t.Group):
...     mass_1: h5t.Dataset[h5t.f8]
...     approximant: str
>>> class PEResult(h5t.File):
...     runs: h5t.Group[Posterior]
"""

from h5t._compile import Dataset, File, Group
from h5t._dtypes import (
    DType,
    c8,
    c16,
    f4,
    f8,
    i1,
    i2,
    i4,
    i8,
    u1,
    u2,
    u4,
    u8,
)
from h5t._errors import (
    ClosedFileError,
    H5TError,
    Invalid,
    Problem,
    SchemaError,
    SchemaMismatchError,
    Severity,
    ValidationError,
    ValidationReport,
)
from h5t._spec import Keys, Name

__version__ = "0.1.0"

__all__ = [
    "ClosedFileError",
    "DType",
    "Dataset",
    "File",
    "Group",
    "H5TError",
    "Invalid",
    "Keys",
    "Name",
    "Problem",
    "SchemaError",
    "SchemaMismatchError",
    "Severity",
    "ValidationError",
    "ValidationReport",
    "__version__",
    "c8",
    "c16",
    "f4",
    "f8",
    "i1",
    "i2",
    "i4",
    "i8",
    "u1",
    "u2",
    "u4",
    "u8",
]
