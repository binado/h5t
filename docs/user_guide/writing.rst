Writing with ``model_dump_hdf5``
================================

:meth:`h5t.Group.model_dump_hdf5` validates a :class:`h5t.Group` instance against its own
schema and writes it, so the code that produces files is checked by the same declarations
that read them.

.. code-block:: python

   import numpy as np
   from pydantic import BaseModel

   import h5t


   class ResultAttrs(BaseModel):
       version: int


   class Result(h5t.Group[ResultAttrs]):
       values: np.ndarray


   result = Result(attrs=ResultAttrs(version=1), values=np.arange(3.0))
   result.model_dump_hdf5("out.h5", mode="w")                    # "w-" (default) fails if it exists
   result.model_dump_hdf5("out.h5", root="/run2", mode="a")      # add another group to the file
   print(Result.model_validate_hdf5("out.h5", root="/run2"))

Options
-------

``mode``
    ``"w-"`` / ``"x"`` (default) fail if the file exists, ``"w"`` truncates, ``"a"``
    writes into an existing file or creates one.

``root`` and ``overwrite``
    Missing intermediate groups are created. A ``root`` group that already has members or
    attributes raises :class:`h5t.ValidationError` unless ``overwrite=True``, which
    deletes its contents first and writes over them, as ``del group[name]`` does in h5py.

Attribute serialization
-----------------------

- ``PlainSerializer(fn)`` on an attribute field serializes the Python value before writing
  (e.g. ``PlainSerializer(json.dumps)`` paired with ``BeforeValidator(json.loads)``).
- An ``Enum`` attribute is stored as its ``.value``.
- An attribute h5py cannot store (such as a raw ``dict`` without a serializer) raises a
  :class:`h5t.ValidationError`.
- A ``None`` optional attribute is omitted from the file.
- An untyped ``Group`` or ``Dataset`` writes its ``attrs`` mapping entries directly as raw
  HDF5 attributes.
- Collection keys must be non-empty, contain no ``/``, and not be ``"."`` or ``".."``.

Two-phase planning
------------------

The whole group tree is planned and every :class:`h5t.Dataset` data is read *before* the
file is opened, so a failed check never touches the target file (and saving into the file a
lazy ``Dataset`` came from is safe).

Lazy datasets are copied: an uncached ``Dataset`` is read from its source file at save time,
without caching it on your instance.

Runtime subclasses
------------------

Each nested group is written using its runtime class's ``model_fields``, so additional
subclass fields are included even when the parent annotation names a base class. This
also applies to collection items. Attribute models use their runtime declared fields
and serializers; computed fields are excluded. HDF5 declaration checks happen during
planning, before opening the destination, without revalidating instance values.
