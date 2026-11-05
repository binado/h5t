Quickstart
==========

A *schema* is just a class, usually a dataclass, that describes part of your file. Mark a
class with :func:`h5t.group` if it describes an HDF5 group (the root counts too), or with
:func:`h5t.dataset` if it describes a dataset. Your class stays an ordinary class; h5t
adds nothing to it.

Define, save and load
---------------------

.. code-block:: python

   import dataclasses

   import numpy as np

   import h5t


   @h5t.dataset(data="data")
   @dataclasses.dataclass
   class Measurement:
       unit: str
       data: h5t.LazyArray


   @h5t.group()
   @dataclasses.dataclass
   class Result:
       version: int                  # an HDF5 attribute
       values: np.ndarray            # a child dataset, loaded eagerly
       measurement: Measurement      # a child dataset with attributes


   lazy = h5t.LazyArray("", "", (4,), np.dtype("f8"), data=np.linspace(0, 1, 4))
   record = Result(version=1, values=np.arange(3.0), measurement=Measurement("m", lazy))

   h5t.save(record, "result.h5", mode="w")
   result = h5t.load(Result, "result.h5")
   print(result.version, result.values, result.measurement.unit)

When ``h5t.load`` returns, the file is already closed, even if something went wrong. The
result is a normal Python object that doesn't depend on the file.

Lazy arrays
-----------

A :class:`h5t.LazyArray` field remembers a dataset's shape and type, but doesn't read the
numbers until you ask for them:

.. code-block:: python

   import dataclasses

   import h5py
   import numpy as np

   import h5t

   with h5py.File("lazy.h5", "w") as f:
       f["samples"] = np.arange(1000.0)


   @h5t.group()
   @dataclasses.dataclass
   class Run:
       samples: h5t.LazyArray


   lazy = h5t.load(Run, "lazy.h5").samples
   print(lazy.path, lazy.shape, lazy.dtype, lazy.ndim)

   complete = lazy.data          # first access reads the data and keeps it
   assert lazy.read() is complete

   with lazy.open() as live:     # a fresh file view, useful for slices
       first_hundred = live[:100]

See :doc:`/user_guide/arrays` for more.
