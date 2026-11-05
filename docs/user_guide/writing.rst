Writing with ``h5t.save``
=========================

:func:`h5t.save` validates a ``@h5t.group`` record against its own schema and writes it,
so the code that produces files is checked by the same declarations that read them.

.. code-block:: python

   import dataclasses

   import numpy as np

   import h5t


   @h5t.group()
   @dataclasses.dataclass
   class Result:
       version: int
       values: np.ndarray


   result = Result(version=1, values=np.arange(3.0))
   h5t.save(result, "out.h5", mode="w")                    # "w-" (default) fails if it exists
   h5t.save(result, "out.h5", root="/run2", mode="a")      # add another group to the file
   print(h5t.load(Result, "out.h5", root="/run2"))

Options
-------

``mode``
    ``"w-"`` / ``"x"`` (default) fail if the file exists, ``"w"`` truncates, ``"a"``
    writes into an existing file or creates one.

``root`` and ``overwrite``
    Missing intermediate groups are created. A ``root`` group that already has members or
    attributes raises :class:`h5t.ValidationError` unless ``overwrite=True``, which
    replaces its contents (destructive, and not undone if the write then fails).

Attribute serialization
-----------------------

- ``Attr(serializer=...)`` is the inverse of ``converter`` (e.g. ``json.dumps``).
- An ``Enum`` attribute is stored as its ``.value``.
- An attribute h5py cannot store (such as a ``dict``) is a ``ValidationError`` telling you
  to add a serializer.
- A ``None`` optional is omitted from the file.
- A field bound with ``attrs=`` writes its undeclared entries back as raw attributes, so
  load → save keeps them. Declared names always come from their typed fields.
- Collection keys must be non-empty, contain no ``/`` and match the field's ``Glob``.

Validation first, rollback on failure
-------------------------------------

The whole record is validated, jaxtyping shapes are checked, and every ``LazyArray``
payload is read *before* the file is opened, so a bad record never touches the target
(and saving into the file a ``LazyArray`` came from is safe). If the write itself fails,
the new file is removed; with ``mode="a"`` the group that ``save`` created is removed on
a best-effort basis.

Lazy payloads are copied: an uncached one is read from its source file at save time,
without caching it on your record.
