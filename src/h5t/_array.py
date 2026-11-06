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
from jaxtyping import jaxtyped
from pydantic import BaseModel, GetCoreSchemaHandler, ValidatorFunctionWrapHandler
from pydantic_core import PydanticCustomError, core_schema
from typing_extensions import TypeVar

from h5t._errors import SchemaError, ValidationError
from h5t._h5view import _GroupView, _normalized_attrs, _raw_attrs

_DATA_NOT_LOADED = object()

T = TypeVar("T", bound=BaseModel | Mapping[str, Any], default=Mapping[str, Any])

_SPECIALIZED_CACHE: weakref.WeakKeyDictionary[
    type[BaseModel], weakref.WeakValueDictionary[type, type]
] = weakref.WeakKeyDictionary()


def _is_mapping_type(tp: Any) -> bool:
    # Identity, not ``in``: PEP 649 forward-reference stand-ins overload ``__eq__``.
    if tp is dict or tp is Mapping:
        return True
    origin = typing.get_origin(tp)
    if origin in (dict, Mapping):
        args = typing.get_args(tp)
        return len(args) == 2 and args[0] is str
    return False


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return a read-only view, leaving the caller's own array untouched."""
    view = array.view()
    view.flags.writeable = False
    return view


def _reject_not_a_dataset(value: Any, expected: type) -> None:
    """Raise unless ``value`` is a `Dataset` (and an ``expected`` instance)."""
    if isinstance(value, _GroupView):
        raise PydanticCustomError("dataset_type", "expected a dataset, found a group")
    if not isinstance(value, Dataset):
        raise PydanticCustomError(
            "dataset_type",
            "Input should be an instance of Dataset, got {actual}",
            {"actual": type(value).__name__},
        )
    if not isinstance(value, expected):
        raise PydanticCustomError(
            "dataset_type",
            "Input should be an instance of {expected}, got {actual}",
            {"expected": expected.__name__, "actual": type(value).__name__},
        )


class Dataset(Generic[T]):
    """A detached HDF5 dataset: metadata and typed attributes, with lazy array slicing.

    No file is kept open: a ``Dataset`` loaded via `Group.model_validate_hdf5` remembers
    where the dataset lives, along with its shape, dtype, and validated ``attrs``, and
    reads array slices or the full array only when indexed (``ds[()]``, ``ds[:10]``) or
    accessed via `data`. You can also construct a ``Dataset`` in memory from a NumPy
    array to write out with `Group.model_dump_hdf5`.

    Parameters
    ----------
    data : array_like
        In-memory array data.
    attrs : BaseModel or Mapping, optional
        Dataset attributes. Defaults to an empty mapping. For ``Dataset[T]`` the empty
        mapping is validated as ``T``, so every field of ``T`` then needs a default.

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

    _h5t_filename: str | None
    _h5t_path: str | None
    _h5t_shape: tuple[int, ...]
    _h5t_dtype: np.dtype[Any]
    _h5t_attrs: T

    def __class_getitem__(cls, item: Any) -> Any:
        if isinstance(item, TypeVar):
            return super().__class_getitem__(item)
        if _is_mapping_type(item):
            args = typing.get_args(item)
            if args and args[1] is not Any:
                raise SchemaError(
                    "Dataset[...] accepts Mapping[str, Any] (or a bare mapping) for untyped "
                    f"attributes; got {item!r}. Use a BaseModel subclass to type them."
                )
            return cls
        if isinstance(item, (str, typing.ForwardRef)) or type(item).__name__ == "_Stringifier":
            # A not-yet-defined name (PEP 649 evaluates annotations with stand-ins). Raising
            # NameError makes Pydantic treat the field as an undefined forward reference
            # and re-evaluate it later instead of freezing it as an untyped ``Dataset``.
            raise NameError(f"name {getattr(item, '__forward_arg__', item)!s} is not defined")
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
        self._h5t_filename = None
        self._h5t_path = None
        self._h5t_shape = arr.shape
        self._h5t_dtype = arr.dtype
        self._h5t_data: object = _readonly(arr)
        attrs_type = self._h5t_attrs_type
        if attrs_type is not None:
            if isinstance(attrs, attrs_type):
                self._h5t_attrs = typing.cast(T, attrs)
            else:
                raw = {} if attrs is None else attrs
                self._h5t_attrs = typing.cast(T, attrs_type.model_validate(raw))
        elif isinstance(attrs, BaseModel):
            self._h5t_attrs = typing.cast(T, attrs)
        else:
            raw_map = {} if attrs is None else dict(attrs)
            self._h5t_attrs = typing.cast(T, MappingProxyType(raw_map))

    @classmethod
    def _build(
        cls,
        filename: str | None,
        path: str | None,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        attrs: Any,
        cached: object = _DATA_NOT_LOADED,
    ) -> Self:
        """Create an instance directly from its state (the one place that does so)."""
        instance = object.__new__(cls)
        instance._h5t_filename = filename
        instance._h5t_path = path
        instance._h5t_shape = shape
        instance._h5t_dtype = dtype
        instance._h5t_attrs = attrs
        instance._h5t_data = cached
        return instance

    def _clone_with(self, target_cls: type[Dataset[Any]], attrs: Any) -> Dataset[Any]:
        """Copy this dataset's location and cached data into ``target_cls`` with ``attrs``."""
        return target_cls._build(
            self.filename, self.path, self.shape, self.dtype, attrs, self._h5t_data
        )

    @property
    def filename(self) -> str | None:
        """The source HDF5 file, or ``None`` for an in-memory dataset."""
        return self._h5t_filename

    @property
    def path(self) -> str | None:
        """The dataset's absolute HDF5 path, or ``None`` for an in-memory dataset."""
        return self._h5t_path

    @property
    def shape(self) -> tuple[int, ...]:
        """The dataset's shape."""
        return self._h5t_shape

    @property
    def dtype(self) -> np.dtype[Any]:
        """The dataset's NumPy dtype."""
        return self._h5t_dtype

    @property
    def attrs(self) -> T:
        """The dataset's validated HDF5 attributes."""
        return self._h5t_attrs

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
        """Read and cache the complete array as a NumPy array.

        The first access on a detached dataset reads the array from the HDF5 file and
        caches it; later accesses reuse the cached array.

        Returns
        -------
        numpy.ndarray
            The complete array, read-only.

        Raises
        ------
        ValidationError
            The source file cannot be opened, the dataset no longer exists or is not a
            dataset, or its shape or dtype changed since it was loaded.
        """
        if self._h5t_data is _DATA_NOT_LOADED:
            with self.open() as dataset:
                value = _readonly(np.asarray(dataset[()]))
            self._h5t_data = value
        return typing.cast(np.ndarray, self._h5t_data)

    def read(self) -> np.ndarray:
        """Return the complete array; an alias of `data`.

        Returns
        -------
        numpy.ndarray
            The same cached array as `data`.
        """
        return self.data

    def __getitem__(self, key: Any) -> np.ndarray:
        """Read ``key`` with ``h5py``-style indexing and return a NumPy array.

        ``ds[()]``, ``ds[...]`` and ``ds[:]`` return the complete cached array (reading it
        once). Any other key reads just that selection from the file, without caching,
        unless the full array is already cached, in which case it indexes the cache.

        Parameters
        ----------
        key : int, slice, tuple, Ellipsis, array_like
            Anything ``h5py.Dataset`` or ``numpy.ndarray`` accepts as an index.

        Returns
        -------
        numpy.ndarray
            The selected data. Full reads are read-only.

        Raises
        ------
        ValidationError
            The dataset cannot be read from its source file; see `open`.
        """
        # Compare by type first: ``ndarray == ()`` is elementwise and would be ambiguous.
        if (
            key is Ellipsis
            or (isinstance(key, tuple) and key == ())
            or (isinstance(key, slice) and key == slice(None))
        ):
            return self.data
        if self._h5t_data is not _DATA_NOT_LOADED:
            return np.asarray(typing.cast(np.ndarray, self._h5t_data)[key])
        with self.open() as dataset:
            return np.asarray(dataset[key])

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Support ``np.asarray(ds)`` by reading and caching the complete array."""
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
            The dataset is in-memory only, its source file cannot be opened, the dataset
            no longer exists, the path names a group, or the shape or dtype differ from the
            values recorded when it was loaded.
        """
        if self.filename is None or self.path is None:
            raise ValidationError("<in-memory>", "dataset is not backed by an HDF5 file")
        try:
            h5file = h5py.File(self.filename, mode="r")
        except OSError as exc:
            raise ValidationError(
                self.path, f"cannot open the source file ({type(exc).__name__}): {exc}"
            ) from exc
        with h5file:
            node = h5file.get(self.path)
            if node is None:
                raise ValidationError(self.path, "dataset no longer exists")
            if not isinstance(node, h5py.Dataset):
                raise ValidationError(self.path, "expected a dataset, found a group")
            if node.shape != self.shape or node.dtype != self.dtype:
                raise ValidationError(
                    self.path,
                    f"dataset changed since it was loaded: recorded shape {self.shape} dtype "
                    f"{self.dtype}, file has shape {node.shape} dtype {node.dtype}",
                )
            yield node

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Dataset):
            return NotImplemented
        # Cheap metadata first: only equal-looking datasets pay for reading both arrays.
        if self.shape != other.shape or self.dtype != other.dtype or self.attrs != other.attrs:
            return False
        return bool(np.array_equal(self.data, other.data))

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

        target_cls: type[Dataset[Any]] = (
            cls if attrs_type is None or cls._h5t_attrs_type is not None else cls[attrs_type]
        )

        if attrs_type is None:

            def validate_untyped(value: Any) -> Dataset[Any]:
                if isinstance(value, h5py.Dataset):
                    return target_cls._build(
                        value.file.filename,
                        value.name,
                        tuple(value.shape),
                        np.dtype(value.dtype),
                        MappingProxyType(_raw_attrs(value)),
                    )
                _reject_not_a_dataset(value, cls)
                if not isinstance(value.attrs, Mapping):
                    raise PydanticCustomError(
                        "dataset_attrs",
                        "Expected Mapping for dataset attrs, got {actual}",
                        {"actual": type(value.attrs).__name__},
                    )
                return value

            return core_schema.no_info_plain_validator_function(validate_untyped)

        attrs_schema = handler.generate_schema(attrs_type)

        def validate_typed(
            value: Any, validate_attrs: ValidatorFunctionWrapHandler
        ) -> Dataset[Any]:
            if isinstance(value, h5py.Dataset):
                with jaxtyped("context"):
                    attrs = validate_attrs(_normalized_attrs(value))
                return target_cls._build(
                    value.file.filename,
                    value.name,
                    tuple(value.shape),
                    np.dtype(value.dtype),
                    attrs,
                )
            # The typed branch deliberately skips the ``expected`` check: any Dataset
            # is accepted and cloned into ``target_cls`` with validated attrs.
            _reject_not_a_dataset(value, Dataset)
            if isinstance(value.attrs, attrs_type):
                if isinstance(value, target_cls):
                    return value
                return value._clone_with(target_cls, value.attrs)
            if isinstance(value.attrs, Mapping):
                with jaxtyped("context"):
                    validated_attrs = validate_attrs(dict(value.attrs))
                return value._clone_with(target_cls, validated_attrs)
            raise PydanticCustomError(
                "dataset_attrs",
                "Expected {expected} for dataset attrs, got {actual}",
                {"expected": attrs_type.__name__, "actual": type(value.attrs).__name__},
            )

        return core_schema.no_info_wrap_validator_function(validate_typed, attrs_schema)
