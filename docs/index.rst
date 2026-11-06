h5t
===

``h5t`` turns HDF5 groups into typed, detached Pydantic objects, and writes them back using
the same description. You describe the layout of your file once as an :class:`h5t.Group`
subclass; ``h5t`` reads it, checks shapes and dtypes, and closes the file for you. Child
:class:`h5t.Dataset` fields are loaded lazily, so you only pay for the slices you actually
read.

.. code-block:: python

   import numpy as np
   from pydantic import BaseModel

   import h5t


   class ResultAttrs(BaseModel):
       version: int


   class Result(h5t.Group[ResultAttrs]):
       values: np.ndarray


   record = Result(attrs=ResultAttrs(version=1), values=np.arange(3.0))
   record.model_dump_hdf5("result.h5", mode="w")
   result = Result.model_validate_hdf5("result.h5")
   print(result.attrs.version, result.values)

Where to start
--------------

- :doc:`getting_started/index`: install ``h5t`` and save and load your first file.
- :doc:`user_guide/index`: how to describe your files, work with datasets and collections,
  write files, and check them from the command line.
- :doc:`reference/index`: every function and class.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   user_guide/index
   reference/index
