"""``LazyArray``, the detached, lazily-read dataset payload."""

from __future__ import annotations

import typing
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import h5py
import numpy as np

from h5t._errors import ValidationError

_DATA_NOT_LOADED = object()


class LazyArray:
    """A detached HDF5 dataset payload: metadata snapshot, data read on first access.

    No file handle is kept open. Only the filename and dataset path are retained, so
    the payload can be read later. Use it as a record's `Payload` field or directly as
    a member annotation. See :doc:`/user_guide/arrays` for snapshot semantics.

    Parameters
    ----------
    filename : str
        File the dataset lives in.
    path : str
        Absolute HDF5 path of the dataset.
    shape : tuple of int
        Shape captured when the file was loaded.
    dtype : numpy.dtype
        Dtype captured when the file was loaded.
    data : numpy.ndarray, optional
        An already-materialized payload; when given, nothing is read lazily.

    Attributes
    ----------
    filename : str
        File the dataset lives in.
    path : str
        Absolute HDF5 path of the dataset.
    shape : tuple of int
        Shape captured at load time.
    dtype : numpy.dtype
        Dtype captured at load time.

    Examples
    --------
    >>> import numpy as np
    >>> from h5t import LazyArray
    >>> lazy = LazyArray("run.h5", "/samples", (3,), np.dtype("f8"), data=np.zeros(3))
    >>> lazy.ndim
    1
    >>> lazy.data.shape
    (3,)
    """

    def __init__(
        self,
        filename: str,
        path: str,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        data: np.ndarray | None = None,
    ) -> None:
        self.filename = filename
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self._h5t_data: object = _DATA_NOT_LOADED if data is None else data

    @property
    def ndim(self) -> int:
        """Number of dimensions in the captured shape."""
        return len(self.shape)

    @property
    def data(self) -> np.ndarray:
        """Read and cache the complete payload as a NumPy array.

        The first access reads the dataset from the source file and caches the result
        permanently; later accesses return the cached array.

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

        Raises
        ------
        ValidationError
            The dataset no longer exists in the source file, or is not a dataset.
        """
        return self.data

    @contextmanager
    def open(self) -> Iterator[h5py.Dataset]:
        """Open the source file and yield the dataset for live access.

        Always opens the *current* file and dataset, so it may observe replacements.
        The file is closed on normal and exceptional exit.

        Yields
        ------
        h5py.Dataset
            The open dataset, valid only inside the ``with`` block. Useful for reading
            slices without loading the whole payload.

        Raises
        ------
        ValidationError
            The dataset no longer exists, or the path names a group.
        """
        with h5py.File(self.filename, mode="r") as h5file:
            node = h5file.get(self.path)
            if node is None:
                raise ValidationError(self.path, "dataset no longer exists")
            if not isinstance(node, h5py.Dataset):
                raise ValidationError(self.path, "expected a dataset, found a group")
            yield node
