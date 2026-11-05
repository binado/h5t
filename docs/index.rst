h5t
===

``h5t`` turns HDF5 groups into plain, typed Python objects, and writes them back using
the same description. You describe the layout of your file once as a class; h5t reads it,
checks it, and closes the file for you. Large arrays can be loaded lazily, so you only
pay for the data you actually use.

.. code-block:: python

   import dataclasses

   import numpy as np

   import h5t


   @h5t.group()
   @dataclasses.dataclass
   class Result:
       version: int
       values: np.ndarray


   h5t.save(Result(version=1, values=np.arange(3.0)), "result.h5", mode="w")
   result = h5t.load(Result, "result.h5")
   print(result.version, result.values)

Where to start
--------------

- :doc:`getting_started/index`: install ``h5t`` and save and load your first file.
- :doc:`user_guide/index`: how to describe your files, work with arrays and collections,
  write files, and check them from the command line.
- :doc:`reference/index`: every function and class.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   user_guide/index
   reference/index
