Quickstart
==========

A *schema* is a subclass of :class:`h5t.Group` (which is itself a :class:`pydantic.BaseModel`).
Parameterize ``h5t.Group[Attrs]`` with a :class:`pydantic.BaseModel` subclass to type its HDF5
attributes on ``.attrs``, and declare immediate HDF5 child datasets and subgroups as fields on
the group class.

Define, save and load
---------------------

.. code-block:: python

   import numpy as np
   from pydantic import BaseModel

   import h5t


   class MeasurementAttrs(BaseModel):
       unit: str


   class ResultAttrs(BaseModel):
       version: int


   class Result(h5t.Group[ResultAttrs]):
       values: np.ndarray                           # a child dataset, loaded eagerly
       measurement: h5t.Dataset[MeasurementAttrs]   # a lazy child dataset with typed attributes


   group = Result(
       attrs=ResultAttrs(version=1),
       values=np.arange(3.0),
       measurement=h5t.Dataset(np.linspace(0, 1, 4), MeasurementAttrs(unit="m")),
   )

   group.model_dump_hdf5("result.h5", mode="w")
   result = Result.model_validate_hdf5("result.h5")
   print(result.attrs.version, result.values, result.measurement.attrs.unit)

When :meth:`h5t.Group.model_validate_hdf5` returns, the HDF5 file is already closed, even if
something went wrong.

Lazy datasets
-------------

An :class:`h5t.Dataset` field remembers a dataset's shape, dtype, and validated ``attrs``,
but doesn't read the array data until you index it or access ``.data``:

.. code-block:: python

   import h5py
   import numpy as np

   import h5t

   with h5py.File("lazy.h5", "w") as f:
       f["samples"] = np.arange(1000.0)


   class Run(h5t.Group):
       samples: h5t.Dataset


   samples = Run.model_validate_hdf5("lazy.h5").samples
   print(samples.path, samples.shape, samples.dtype, samples.ndim)

   first_hundred = samples[:100]     # reads only the requested slice from disk
   complete = samples[()]            # reads and caches the full array
   assert samples.data is complete

See :doc:`/user_guide/arrays` for more.
