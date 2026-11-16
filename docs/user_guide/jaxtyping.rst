Checking dtype and shape with jaxtyping
=======================================

A `jaxtyping <https://docs.kidger.site/jaxtyping/>`_ annotation over ``np.ndarray`` or
:class:`h5t.Dataset` (including parameterized ``h5t.Dataset[T]``) has its dtype and shape
checked by jaxtyping itself, from the dataset's HDF5 metadata, without reading lazy disk
chunks or casting data. ``h5t`` never parses a shape string.

- ``Float`` accepts any float width and ``Float64`` only float64 (a file's byte order is
  ignored). ``Shaped`` accepts any dtype, and a custom ``AbstractDtype`` subclass works too.
- The whole shape grammar is jaxtyping's (``...``, ``*batch``, ``#n``, ``n+1``, ...).

Dimension binding scopes
------------------------

A named dimension binds on first use and must agree across every array and ``Dataset`` field
of the same group. A nested :class:`h5t.Group` starts a fresh scope. Collection items that are
bare arrays or datasets share the owner group's bindings, while collection items that are
``Group`` subclasses bind independently (see :doc:`collections`).
:meth:`h5t.Group.model_dump_hdf5` and in-memory ``Group(...)`` construction apply the same
checks.

.. code-block:: python

   import h5py
   import numpy as np
   from jaxtyping import Float
   from pydantic import BaseModel

   import h5t

   with h5py.File("shapes.h5", "w") as f:
       f["fisher"] = np.zeros((3, 3))
       ds = f.create_dataset("fiducials", data=np.zeros(4))
       ds.attrs["unit"] = "m"


   class FiducialAttrs(BaseModel):
       unit: str


   class Result(h5t.Group):
       fisher: Float[np.ndarray, "n n"]
       fiducials: Float[h5t.Dataset[FiducialAttrs], "n"]


   try:
       Result.model_validate_hdf5("shapes.h5")
   except h5t.ValidationError as exc:
       print(exc)

The error message names the dimension and both sizes::

   /fiducials: the size of dimension n is 4 which does not equal the existing value of 3
