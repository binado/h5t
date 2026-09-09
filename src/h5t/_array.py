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
    """A detached HDF5 dataset payload: metadata snapshot, data read on first access."""

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
        """Read and cache the complete current payload as a NumPy array."""
        if self._h5t_data is _DATA_NOT_LOADED:
            with self.open() as dataset:
                value = np.asarray(dataset[()])
            self._h5t_data = value
        return typing.cast(np.ndarray, self._h5t_data)

    def read(self) -> np.ndarray:
        """Return the same cached complete payload as ``data``."""
        return self.data

    @contextmanager
    def open(self) -> Iterator[h5py.Dataset]:
        """Open the current source file and yield this dataset for live access."""
        with h5py.File(self.filename, mode="r") as h5file:
            node = h5file.get(self.path)
            if node is None:
                raise ValidationError(self.path, "dataset no longer exists")
            if not isinstance(node, h5py.Dataset):
                raise ValidationError(self.path, "expected a dataset, found a group")
            yield node
