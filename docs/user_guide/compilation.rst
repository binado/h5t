Model lifecycle and forward references
======================================

Pydantic owns model construction
--------------------------------

Subclassing :class:`h5t.Group` builds an ordinary Pydantic model. Pydantic resolves
annotations and provides ``model_fields``; h5t has no separate compiled schema or
annotation namespace. Ordinary in-memory construction uses Pydantic validation.

HDF5 declaration checks run when loading or writing, before the destination or source
file is opened. They reject unsupported child annotations, duplicate HDF5 names,
invalid aliases, and unsupported attribute and extra-field policies.
Loading checks all reachable group declarations, including optional children and
collection items. Recursive schemas are checked once per operation.

An invalid HDF5 declaration raises :class:`h5t.SchemaError`. Pydantic model-building
errors propagate unchanged, including ``PydanticUndefinedAnnotation`` for unresolved
names. Type-owned schema hooks (such as those for datasets and jaxtyping arrays) can
still reject unsupported types while Pydantic builds the model.

Forward references
------------------

Before inspecting an incomplete model, h5t calls its ``model_rebuild()``. Module-level
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
are omitted. Values are not revalidated during writing.
