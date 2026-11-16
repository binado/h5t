Model lifecycle and forward references
======================================

Pydantic owns model construction
--------------------------------

Subclassing :class:`h5t.Group` builds an ordinary Pydantic model. Pydantic resolves
annotations and provides ``model_fields``; h5t has no separate compiled schema or
annotation namespace. Ordinary in-memory construction uses Pydantic validation.

HDF5 declaration checks run while Pydantic builds a group's schema, so an invalid
declaration fails where the class is defined, with no file involved. They reject
unsupported child annotations, duplicate HDF5 names, invalid aliases, and unsupported
attribute and extra-field policies. Each class checks its own fields; nested groups and
collection items are checked when their own class is built.

An invalid HDF5 declaration raises :class:`h5t.SchemaError`. Pydantic model-building
errors propagate unchanged, including ``PydanticUndefinedAnnotation`` for unresolved
names. A class with unresolved forward references is checked later, when
``model_rebuild()`` (or its first use) resolves them.

Forward references
------------------

Before loading into an incomplete model, h5t calls its ``model_rebuild()``. Module-level
forward references and self-references follow Pydantic's normal rules:

.. code-block:: python

   import h5t

   class Node(h5t.Group):
       child: "Node | None" = None

For unresolved function-local dependencies, rebuild explicitly once the types exist:

.. code-block:: python

   import numpy as np
   import h5t

   def make_schema():
       class Parent(h5t.Group):
           child: "Child"

       class Child(h5t.Group):
           values: np.ndarray

       Parent.model_rebuild(_types_namespace={"Child": Child})
       return Parent

The former ``localns=`` class keyword is removed without a compatibility shim. Replace
it with ``Model.model_rebuild(_types_namespace={...})``. h5t does not inspect caller
frames, retain custom namespaces, or clear Pydantic's parent namespace.

Writing runtime subclasses
--------------------------

Writing inspects each group's runtime class. A child declared as a base ``Group`` but
holding a subclass writes the subclass's additional fields too, including collection
items. Typed attributes likewise use their runtime model's declared fields and
serializers. Computed fields are excluded, and nullable attributes holding ``None``
are omitted. Values are not revalidated during writing. A runtime attributes model that
no group declares is checked as it is written.
