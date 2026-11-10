API reference
=============

.. module:: h5t

Groups and datasets
-------------------

.. autoclass:: h5t.Group
   :members: model_validate_hdf5, model_dump_hdf5

.. autoclass:: h5t.Dataset
   :members: ndim, size, nbytes, data, open, __getitem__, __array__

Exceptions
----------

.. autoexception:: h5t.H5TError

.. autoexception:: h5t.SchemaError

.. autoexception:: h5t.ValidationError
