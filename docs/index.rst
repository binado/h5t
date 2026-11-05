h5t
===

``h5t`` loads HDF5 groups into detached, typed Python records, and writes them back
with the same schema. Group attributes and ordinary NumPy-array fields are materialized
while the file is open. Typed datasets keep snapshot metadata and can read their payload
lazily, without retaining an open file descriptor.

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

- :doc:`getting_started/index`: install ``h5t``, then define, save and load a first schema.
- :doc:`user_guide/index`: every rule of the schema language, arrays, collections,
  writing, and the CLI.
- :doc:`reference/index`: the public API.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   user_guide/index
   reference/index
