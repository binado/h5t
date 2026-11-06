Compilation and forward references
==================================

Eager compilation
-----------------

Subclassing :class:`h5t.Group` compiles the schema at class creation. Annotations resolve
against the module globals, the defining function's locals (read once, never stored), the
explicit ``localns=`` class keyword argument, and the class's own name (so self-reference
works). A bad declaration therefore raises :class:`h5t.SchemaError` at class definition,
right where the mistake is.

Deferral for forward references
-------------------------------

A :class:`h5t.Group` subclass that names another ``Group`` defined later in its module, or
references itself, compiles at class definition when it can. A forward reference that is
not yet defined defers compilation to first use (when the group is loaded or dumped);
a self-reference resolves immediately.

.. code-block:: python

   import h5py
   from pydantic import BaseModel

   import h5t


   class NodeAttrs(BaseModel):
       name: str


   class ChildAttrs(BaseModel):
       label: str


   class Node(h5t.Group[NodeAttrs]):
       child: "Child | None" = None      # defined below: compilation is deferred


   class Child(h5t.Group[ChildAttrs]):
       pass


   with h5py.File("tree.h5", "w") as f:
       f.attrs["name"] = "root"
       f.create_group("child").attrs["label"] = "leaf"

   print(Node.model_validate_hdf5("tree.h5"))

Function-local groups and ``localns``
-------------------------------------

A group declared inside a function sees that function's locals when the class statement runs,
so ordinary function-local groups and self-references just work. A deferred :class:`h5t.Group`
is retried later *without* them: if it both forward-references a name and uses a
function-local type, pass the namespace explicitly as a class keyword argument:
``class Node(h5t.Group, localns={"Child": Child}):``.

Notes
-----

- A still-unresolved name at load time becomes a :class:`h5t.SchemaError` naming it.
- Pydantic cannot tell a forward reference from a ``NameError`` raised inside a function an
  annotation calls, so the latter is deferred too and reported the same way on first use.
- A group-kind member's target schema is resolved at load time, which is what makes
  ``child: Node | None`` possible without infinite recursion.
