API reference
=============

.. module:: h5t

Loading and saving
------------------

.. autofunction:: h5t.load

.. autofunction:: h5t.save

Record decorators
-----------------

.. autofunction:: h5t.dataset

.. autofunction:: h5t.group

Markers
-------

.. autoclass:: h5t.Name

.. autoclass:: h5t.Attr

.. autoclass:: h5t.Eager

.. autoclass:: h5t.Payload

.. autoclass:: h5t.Glob

Arrays
------

.. autoclass:: h5t.LazyArray
   :members: ndim, data, read, open

Exceptions
----------

.. autoexception:: h5t.H5TError

.. autoexception:: h5t.SchemaError

.. autoexception:: h5t.ValidationError

.. autoexception:: h5t.ConversionError
