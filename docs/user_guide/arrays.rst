Arrays and lazy payloads
========================

Three ways to declare a dataset field
-------------------------------------

``np.ndarray``
    The complete payload is read into an ndarray while the file is open.

:class:`h5t.LazyArray`
    A detached snapshot of the dataset's metadata (``filename``, ``path``, ``shape``,
    ``dtype``). The payload is read on first access. A ``LazyArray`` subclass is accepted
    too.

:class:`h5t.Eager`
    ``Annotated[LazyArray, Eager()]`` is still a ``LazyArray``, but its payload is cached
    during ``h5t.load``. Combined with :class:`h5t.Payload` it is allowed only when the
    payload field is a ``LazyArray``; an ``np.ndarray`` payload is already eager.

``LazyArray`` can be a member annotation directly (a child dataset with no attributes),
or the payload field of a dataset record.

Using a ``LazyArray``
---------------------

.. code-block:: python

   import dataclasses

   import h5py
   import numpy as np

   import h5t

   with h5py.File("arrays.h5", "w") as f:
       f["samples"] = np.arange(1000.0)


   @h5t.group()
   @dataclasses.dataclass
   class Run:
       samples: h5t.LazyArray


   lazy = h5t.load(Run, "arrays.h5").samples
   print(lazy.filename, lazy.path, lazy.shape, lazy.dtype, lazy.ndim)

   complete = lazy.data           # first access reads and caches an ndarray
   assert lazy.read() is complete

   with lazy.open() as live:      # fresh file view; closed on exit
       first_hundred = live[:100]

``h5t.load`` closes every handle on normal and exceptional exits, and so does
:meth:`h5t.LazyArray.open`.

``NDArray`` aliases
-------------------

``numpy.typing.NDArray[...]`` is accepted wherever ``np.ndarray`` is. With numpy 2.5 and
later ``NDArray`` is a PEP 695 type alias; h5t expands such aliases until it reaches the
real array type. The dtype parameter is not validated: use :doc:`jaxtyping` for that.

Snapshot consistency
--------------------

A loaded record is a snapshot, with one deliberate exception:

- Group and dataset attributes, dataset shape/dtype/path, eager datasets and plain
  arrays reflect the file during loading (``h5t.load()``).
- A lazy dataset's first ``.data`` / ``.read()`` observes the file at that later moment
  and caches the resulting array permanently.
- ``.open()`` always opens the current file and current dataset. It may therefore observe
  replacements or fail if the source was changed or deleted.
- ``h5t.save`` copies lazy payloads: an uncached one is read from its source file at
  save time, without caching it on your record.
