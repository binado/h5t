Foreign record types (``Payload``)
==================================

:func:`h5t.dataset` and :func:`h5t.group` are themselves sugar over :class:`h5t.Payload`,
h5t's lower-level mechanism for loading a child dataset into a plain record type without
decorating it, useful for a third-party type you cannot add a decorator to.

.. code-block:: python

   from dataclasses import dataclass
   from typing import Annotated

   import h5py
   import numpy as np

   import h5t

   with h5py.File("payload.h5", "w") as f:
       d = f.create_dataset("measurement", data=np.arange(3.0))
       d.attrs["unit"] = "m"


   @dataclass
   class Measurement:
       unit: str
       data: np.ndarray                  # eager payload


   @h5t.group()
   @dataclass
   class Result:
       measurement: Annotated[Measurement, h5t.Payload("data")]


   print(h5t.load(Result, "payload.h5").measurement)

``Payload("data")`` names the field holding the payload: annotate it ``np.ndarray`` for
an eager array, or :class:`h5t.LazyArray` for h5t's usual detached, lazily-read payload.
Every other field of the record type is loaded as a dataset attribute: scalars,
``Literal[...]``, ``Attr(converter=...)``, defaults, and ``T | None`` all behave as they
do on a group record.

``extras=``
    ``Payload``'s own ``extras=`` (default ``"ignore"``) governs the dataset's
    attributes, since a type you did not decorate cannot carry its own policy.

Constructor
    h5t calls the record type's real constructor. ``__post_init__`` runs, and an
    invariant it raises surfaces as a :class:`h5t.ValidationError`.

``attrs=``
    Pass ``attrs="field"`` to bind a ``Mapping``-annotated field to the dataset's
    attribute snapshot (validated values for declared names, raw for undeclared). It lists
    every declared name, including ones absent from the file that fell back to ``None`` or
    a default.

.. code-block:: python

   from collections.abc import Mapping
   from dataclasses import dataclass
   from typing import Annotated, Any

   import h5py
   import numpy as np

   import h5t

   with h5py.File("payload_attrs.h5", "w") as f:
       d = f.create_dataset("measurement", data=np.arange(3.0))
       d.attrs["unit"] = "m"
       d.attrs["extra"] = 1


   @dataclass
   class Measurement:
       unit: str
       data: np.ndarray
       attrs: Mapping[str, Any]


   @h5t.group()
   @dataclass
   class Result:
       measurement: Annotated[Measurement, h5t.Payload("data", attrs="attrs")]


   print(dict(h5t.load(Result, "payload_attrs.h5").measurement.attrs))

A foreign record has no ``path`` / ``shape`` / ``dtype`` of its own unless its ``Payload``
field is a ``LazyArray`` (whose own ``.path`` / ``.shape`` / ``.dtype`` you read through
it). A ``Payload`` field can also name a ``@h5t.group``-decorated type directly.

Resolution scope
----------------

An ``Annotated[T, Payload(...)]`` use site resolves ``T``'s annotations against its module
globals and class dict only. ``@h5t.dataset`` / ``@h5t.group`` additionally capture their
own call site's frame, so a function-local record's annotations resolve against function
locals too. See :doc:`compilation`.
