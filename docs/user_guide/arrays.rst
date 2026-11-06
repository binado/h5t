Arrays and lazy datasets
========================

Two ways to declare a dataset field
-----------------------------------

``np.ndarray``
    The complete payload is read eagerly into an ``ndarray`` while the file is open, and any
    attributes on the dataset are ignored.

:class:`h5t.Dataset` / ``h5t.Dataset[T]``
    A detached snapshot of the dataset's metadata (``filename``, ``path``, ``shape``,
    ``dtype``) and validated ``attrs: T``. The array payload is read on demand when indexed
    or accessed via ``.data``. Custom ``Dataset`` subclasses are preserved too.

Using a ``Dataset``
-------------------

.. code-block:: python

   import h5py
   import numpy as np
   from pydantic import BaseModel

   import h5t

   with h5py.File("arrays.h5", "w") as f:
       ds = f.create_dataset("samples", data=np.arange(1000.0))
       ds.attrs["unit"] = "V"


   class SampleAttrs(BaseModel):
       unit: str


   class Run(h5t.Group):
       samples: h5t.Dataset[SampleAttrs]


   samples = Run.model_validate_hdf5("arrays.h5").samples
   print(samples.filename, samples.path, samples.shape, samples.dtype, samples.attrs.unit)

   first_hundred = samples[:100]  # partial slice reads only those elements from disk
   complete = samples[()]         # full read via [()], [:], or .data reads and caches
   assert samples.data is complete
   assert samples.read() is complete

   with samples.open() as live:   # explicit live h5py.Dataset handle; closed on exit
       tail = live[-10:]

:meth:`h5t.Group.model_validate_hdf5` closes every file handle on normal and exceptional
exits, and so does :meth:`h5t.Dataset.open`.

``NDArray`` aliases
-------------------

``numpy.typing.NDArray[...]`` is accepted wherever ``np.ndarray`` is. With numpy 2.5 and
later ``NDArray`` is a PEP 695 type alias; h5t expands such aliases until it reaches the
real array type. The dtype parameter is not validated: use :doc:`jaxtyping` for that.

Snapshot consistency
--------------------

A loaded group is a snapshot, with deliberate lazy-read semantics for :class:`h5t.Dataset`:

- Group and dataset ``attrs``, dataset ``shape`` / ``dtype`` / ``path``, and plain
  ``np.ndarray`` fields reflect the file during :meth:`h5t.Group.model_validate_hdf5`.
- A ``Dataset``'s partial slice (`ds[:10]`) reads from the file on demand without caching the
  full array.
- A ``Dataset``'s first full read (`ds[()]`, `ds[:]`, ``.data``, or ``.read()``) reads the
  complete array from the file at that moment and caches it permanently on the instance.
- ``.open()`` always opens the current file and yields the live ``h5py.Dataset``.
- :meth:`h5t.Group.model_dump_hdf5` copies lazy payloads: an uncached ``Dataset`` is read from
  its source file at dump time without caching it on your instance.
