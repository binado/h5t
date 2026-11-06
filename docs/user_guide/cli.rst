Command line: ``h5t check``
===========================

``h5t check`` imports a :class:`h5t.Group` schema, loads a file with it, and reports whether
the file conforms.

.. code-block:: bash

   h5t check result.h5 --schema mypackage.schemas:Result --root /results/latest

``--schema``
    ``pkg.module:ClassName`` of an :class:`h5t.Group` subclass.

``--root``
    Absolute path of the group to load (default ``/``).

Exit status
-----------

==== ===========================================================================
Code Meaning
==== ===========================================================================
0    the file conforms to the schema
1    the first file/schema mismatch (:class:`h5t.ValidationError`)
2    import, schema declaration, usage, or I/O errors
==== ===========================================================================
