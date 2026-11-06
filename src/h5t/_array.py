"""``Dataset[T]``, the detached, lazily-read HDF5 dataset with typed attributes."""

from __future__ import annotations

import typing
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, ClassVar, Generic, Self

import h5py
import numpy as np
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import PydanticCustomError, core_schema
from typing_extensions import TypeVar

from h5t._errors import SchemaError, ValidationError

_DATA_NOT_LOADED = object()

T = TypeVar("T", bound=BaseModel | Mapping[str, Any], default=Mapping[str, Any])

_SPECIALIZED_CACHE: weakref.WeakKeyDictionary[
    type[BaseModel], weakref.WeakValueDictionary[type, type]
] = weakref.WeakKeyDictionary()


def _is_mapping_type(tp: Any) -> bool:
    if tp in (dict, Mapping):
        return True
    origin = typing.get_origin(tp)
    if origin in (dict, Mapping):
        args = typing.get_args(tp)
        return len(args) == 2 and args[0] is str
    return False


class Dataset(Generic[T]):
    """A detached HDF5 dataset: metadata and typed attributes, with lazy array slicing.

    No file is kept open: a ``Dataset`` loaded via `Group.model_validate_hdf5` remembers
    where the dataset lives, along with its shape, dtype, and validated ``attrs``, and
    reads array slices or the full payload only when indexed (`ds[()]`, `ds[:10]`) or
    accessed via `data`. You can also construct a ``Dataset`` in memory from a NumPy
    array to write out with `Group.model_dump_hdf5`.

    Parameters
    ----------
    data : array_like
        In-memory array payload.
    attrs : BaseModel or Mapping, optional
        Dataset attributes. Defaults to an empty mapping (or default-constructed
        ``T`` model when specialized as ``Dataset[T]``).

    Attributes
    ----------
    attrs : T
        Validated dataset attributes (`BaseModel` subclass or read-only `Mapping`).
    filename : str or None
        File the dataset lives in, or ``None`` if constructed in memory.
    path : str or None
        Absolute HDF5 path of the dataset, or ``None`` if constructed in memory.
    shape : tuple of int
        Dataset shape.
    dtype : numpy.dtype
        Dataset data type.

    Examples
    --------
    >>> import numpy as np
    >>> from pydantic import BaseModel
    >>> import h5t
    >>> class Attrs(BaseModel):
    ...     unit: str
    >>> ds = h5t.Dataset(np.arange(4.0), Attrs(unit="m"))
    >>> ds.shape
    (4,)
    >>> ds.attrs.unit
    'm'
    >>> ds[:2].tolist()
    [0.0, 1.0]
    """

    _h5t_attrs_type: ClassVar[type[BaseModel] | None] = None

    filename: str | None
    path: str | None
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    attrs: T

    def __class_getitem__(cls, item: Any) -> Any:
        if isinstance(item, TypeVar):
            return super().__class_getitem__(item)
        if _is_mapping_type(item):
            return cls
        if isinstance(item, type) and issubclass(item, BaseModel):
            per_base = _SPECIALIZED_CACHE.get(item)
            if per_base is None:
                per_base = weakref.WeakValueDictionary()
                _SPECIALIZED_CACHE[item] = per_base
            cached = per_base.get(cls)
            if cached is not None:
                return cached
            specialized = type(
                f"{cls.__name__}[{item.__name__}]",
                (cls,),
                {
                    "_h5t_attrs_type": item,
                    "__module__": cls.__module__,
                },
            )
            per_base[cls] = specialized
            return specialized
        raise SchemaError(
            "Dataset[...] type argument must be a BaseModel subclass or Mapping[str, Any], "
            f"got {item!r}"
        )

    def __init__(
        self,
        data: Any,
        attrs: T | Mapping[str, Any] | None = None,
    ) -> None:
        arr = np.asarray(data)
        self.filename = None
        self.path = None
        self.shape = arr.shape
        self.dtype = arr.dtype
        self._h5t_data: object = arr
        attrs_type = self._h5t_attrs_type
        if attrs_type is not None:
            if isinstance(attrs, attrs_type):
                self.attrs = typing.cast(T, attrs)
            else:
                raw = {} if attrs is None else attrs
                self.attrs = typing.cast(T, attrs_type.model_validate(raw))
        elif isinstance(attrs, BaseModel):
            self.attrs = typing.cast(T, attrs)
        else:
            raw_map = {} if attrs is None else dict(attrs)
            self.attrs = typing.cast(T, MappingProxyType(raw_map))

    @classmethod
    def _from_hdf5(
        cls,
        filename: str,
        path: str,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        attrs: Any,
        *,
        data: np.ndarray | None = None,
    ) -> Self:
        instance = object.__new__(cls)
        instance.filename = filename
        instance.path = path
        instance.shape = shape
        instance.dtype = dtype
        instance.attrs = attrs
        instance._h5t_data = _DATA_NOT_LOADED if data is None else data
        return instance

    @property
    def ndim(self) -> int:
        """Number of dimensions in the dataset shape."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements in the dataset."""
        return int(np.prod(self.shape, dtype=np.intp))

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the elements of the dataset."""
        return self.size * self.dtype.itemsize

    @property
    def data(self) -> np.ndarray:
        """Read and cache the complete payload as a NumPy array.

        The first access on a detached dataset reads the payload from the HDF5 file and
        caches it; later accesses reuse the cached array.

        Returns
        -------
        numpy.ndarray
            The complete payload.

        Raises
        ------
        ValidationError
            The dataset no longer exists in the source file, or is not a dataset.
        """
        if self._h5t_data is _DATA_NOT_LOADED:
            with self.open() as dataset:
                value = np.asarray(dataset[()])
            self._h5t_data = value
        return typing.cast(np.ndarray, self._h5t_data)

    def read(self) -> np.ndarray:
        """Return the complete payload; an alias of `data`.

        Returns
        -------
        numpy.ndarray
            The same cached array as `data`.
        """
        return self.data

    def __getitem__(self, key: Any) -> np.ndarray:
        if key is Ellipsis or key == () or key == slice(None):
            return self.data
        if self._h5t_data is not _DATA_NOT_LOADED:
            return np.asarray(typing.cast(np.ndarray, self._h5t_data)[key])
        with self.open() as dataset:
            return np.asarray(dataset[key])

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        arr = self.data
        if copy is None:
            return np.asarray(arr, dtype=dtype)
        return np.array(arr, dtype=dtype, copy=copy)

    @contextmanager
    def open(self) -> Iterator[h5py.Dataset]:
        """Open the source file and yield the live ``h5py.Dataset`` handle.

        Always looks at the file as it is now, and closes it when the ``with`` block
        ends.

        Yields
        ------
        h5py.Dataset
            The open dataset, valid only inside the ``with`` block.

        Raises
        ------
        ValidationError
            The dataset is in-memory only, no longer exists, or the path names a group.
        """
        if self.filename is None or self.path is None:
            raise ValidationError("<in-memory>", "dataset is not backed by an HDF5 file")
        with h5py.File(self.filename, mode="r") as h5file:
            node = h5file.get(self.path)
            if node is None:
                raise ValidationError(self.path, "dataset no longer exists")
            if not isinstance(node, h5py.Dataset):
                raise ValidationError(self.path, "expected a dataset, found a group")
            yield node

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.attrs == other.attrs and np.array_equal(self.data, other.data)

    def __repr__(self) -> str:
        loc = f"path={self.path!r}, " if self.path is not None else ""
        return (
            f"{type(self).__name__}({loc}shape={self.shape}, "
            f"dtype={self.dtype}, attrs={self.attrs!r})"
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        attrs_type = cls._h5t_attrs_type
        if attrs_type is None:
            args = typing.get_args(source_type)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                attrs_type = args[0]

        def validate_dataset(value: Any) -> Dataset[Any]:
            if not isinstance(value, Dataset):
                raise PydanticCustomError(
                    "dataset_type",
                    "Input should be an instance of Dataset, got {actual}",
                    {"actual": type(value).__name__},
                )
            if cls._h5t_attrs_type is None and attrs_type is None:
                if not isinstance(value, cls):
                    raise PydanticCustomError(
                        "dataset_type",
                        "Input should be an instance of {expected}, got {actual}",
                        {"expected": cls.__name__, "actual": type(value).__name__},
                    )
                if not isinstance(value.attrs, Mapping):
                    raise PydanticCustomError(
                        "dataset_attrs",
                        "Expected Mapping for dataset attrs, got {actual}",
                        {"actual": type(value.attrs).__name__},
                    )
                return value

            assert attrs_type is not None
            target_cls = cls if cls._h5t_attrs_type is not None else cls[attrs_type]
            if isinstance(value.attrs, attrs_type):
                if isinstance(value, target_cls):
                    return value
                clone = object.__new__(target_cls)
                clone.filename = value.filename
                clone.path = value.path
                clone.shape = value.shape
                clone.dtype = value.dtype
                clone.attrs = value.attrs
                clone._h5t_data = value._h5t_data
                return clone
            if isinstance(value.attrs, Mapping):
                validated_attrs = attrs_type.model_validate(dict(value.attrs))
                clone = object.__new__(target_cls)
                clone.filename = value.filename
                clone.path = value.path
                clone.shape = value.shape
                clone.dtype = value.dtype
                clone.attrs = typing.cast(Any, validated_attrs)
                clone._h5t_data = value._h5t_data
                return clone
            raise PydanticCustomError(
                "dataset_attrs",
                "Expected {expected} for dataset attrs, got {actual}",
                {"expected": attrs_type.__name__, "actual": type(value.attrs).__name__},
            )

        return core_schema.no_info_plain_validator_function(validate_dataset)
