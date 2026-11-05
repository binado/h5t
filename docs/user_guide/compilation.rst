Compilation and forward references
==================================

Eager compilation
-----------------

The decorators compile the schema at the decorator call. Annotations resolve against the
module globals, the decorating function's locals (read once, never stored), the explicit
``localns=`` argument, and the class's own name (so self-reference works). A bad
declaration therefore raises :class:`h5t.SchemaError` at the decorator, right where the
mistake is.

``@h5t.dataset`` records never defer: they declare attributes only, so they cannot
forward-reference another schema type.

Deferral for group records
--------------------------

A ``@h5t.group`` record that names a record defined later in its module, or references
itself, compiles on first use (when it is loaded, saved, or used as a field of another
record).

.. code-block:: python

   import dataclasses

   import h5py

   import h5t


   @h5t.group()
   @dataclasses.dataclass
   class Node:
       name: str
       child: "Child | None" = None      # defined below: compilation is deferred


   @h5t.group()
   @dataclasses.dataclass
   class Child:
       label: str


   with h5py.File("tree.h5", "w") as f:
       f.attrs["name"] = "root"
       f.create_group("child").attrs["label"] = "leaf"

   print(h5t.load(Node, "tree.h5"))

Function-local records and ``localns``
--------------------------------------

A record declared inside a function sees that function's locals when the decorator runs,
so ordinary function-local records and self-references just work. A deferred
``@h5t.group`` record is retried later *without* them: if it both forward-references a
name and uses a function-local type, pass the type explicitly with
``@h5t.group(localns={"Child": Child})``. ``@h5t.dataset`` accepts ``localns=`` too.

Notes
-----

- A ``NameError`` raised by a *function an annotation calls* is a bug in that function,
  not a forward reference, and is reported as a :class:`h5t.SchemaError`.
- A still-unresolved name at load time becomes a :class:`h5t.SchemaError`.
- A group-kind member's record is resolved at load time, which is what makes
  ``child: Node | None`` possible without infinite recursion.
