Checking dtype and shape with jaxtyping
=======================================

A `jaxtyping <https://docs.kidger.site/jaxtyping/>`_ annotation over ``np.ndarray`` or
:class:`h5t.LazyArray` has its dtype and shape checked by jaxtyping itself, from the
dataset's HDF5 metadata, without reading the data or casting it. h5t never parses a
shape string.

- ``Float`` accepts any float width and ``Float64`` only float64 (a file's byte order is
  ignored). ``Shaped`` accepts any dtype, and a custom ``AbstractDtype`` subclass works too.
- The whole shape grammar is jaxtyping's (``...``, ``*batch``, ``#n``, ``n+1``, ...).

Dimension binding scopes
------------------------

A named dimension binds on first use and must agree across every array field of the same
record. A nested record starts fresh, and a ``@h5t.dataset`` record is its own scope.
Bare array items of a collection share the owner record's bindings (see
:doc:`collections`). :func:`h5t.save` applies the same checks.

.. code-block:: python

   import dataclasses

   import h5py
   import numpy as np
   from jaxtyping import Float

   import h5t

   with h5py.File("shapes.h5", "w") as f:
       f["fisher"] = np.zeros((3, 3))
       f["fiducials"] = np.zeros(4)


   @h5t.group()
   @dataclasses.dataclass
   class Result:
       fisher: Float[np.ndarray, "n n"]
       fiducials: Float[np.ndarray, "n"]


   try:
       h5t.load(Result, "shapes.h5")
   except h5t.ValidationError as exc:
       print(exc)

The error message names the dimension and both sizes::

   /fiducials: the size of dimension n is 4 which does not equal the existing value of 3
