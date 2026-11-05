Schemas
=======

A schema is a plain class decorated with :func:`h5t.dataset` or :func:`h5t.group`.
The two decorators define the two kinds of record:

- A **dataset** record names one payload field and may declare only attributes besides.
- A **group** record has no payload and admits attributes, nested groups, datasets and
  arrays. It may reference other schema types, including itself.

Field rules
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Annotation
     - HDF5 representation
     - Loading behavior
   * - ``@h5t.group()``-decorated class
     - child group (or the root)
     - recursively materialized via ``record_type(**fields)``
   * - ``@h5t.dataset(data=...)``-decorated class
     - child dataset
     - attributes loaded, ``record_type(**fields)`` constructed
   * - ``Annotated[T, Payload("field")]``
     - child dataset
     - attributes loaded, ``T(**fields)`` constructed
   * - ``LazyArray`` (or a subclass)
     - child dataset
     - metadata snapshot, payload read on first access
   * - ``Annotated[<any dataset field>, Eager()]``
     - child dataset
     - complete payload cached during loading
   * - ``np.ndarray``
     - child dataset
     - complete payload loaded as an ndarray
   * - ``Float[np.ndarray, "n n"]``, ``Float[LazyArray, "n _"]`` (jaxtyping)
     - child dataset
     - dtype and shape checked against the file's metadata before any read; otherwise
       as ``np.ndarray`` / ``LazyArray``
   * - ``dict[str, T]`` / ``Mapping[str, T]``
     - child group named by the field
     - one item per child of that group, as a plain ``dict``; see :doc:`collections`
   * - ``Annotated[T, Attr(...)]``
     - attribute
     - converter, then Pydantic validation
   * - scalar or ``Literal[...]``
     - attribute
     - Pydantic validation

:class:`h5t.Name` renames any field kind. :class:`h5t.Attr` is valid only for attributes
and never combines with :class:`h5t.Payload`. :class:`h5t.Eager` is valid only for dataset
fields. A parameterized alias such as ``numpy.typing.NDArray[np.floating]`` is accepted
wherever ``np.ndarray`` is; for ``NDArray`` the dtype parameter is not validated (use
:doc:`jaxtyping` to check it). Unsupported collection-shaped child annotations raise
:class:`h5t.SchemaError`.

.. code-block:: python

   import dataclasses
   import json
   from typing import Annotated, Any

   import numpy as np

   import h5t


   @h5t.group()
   @dataclasses.dataclass
   class Nested:
       label: str


   @h5t.group(extras="ignore")
   @dataclasses.dataclass
   class Result:
       version: int                                   # implicit HDF5 attribute
       title: Annotated[str, h5t.Name("name")]        # renamed attribute
       config: Annotated[dict[str, Any], h5t.Attr(converter=json.loads, serializer=json.dumps)]
       array_attr: Annotated[np.ndarray, h5t.Attr()]
       values: np.ndarray                             # eager dataset payload
       nested: Nested                                 # recursively loaded group
       note: str | None                               # absent becomes None
       revision: int = 1                              # absent uses a validated default


   record = Result(
       version=2,
       title="demo",
       config={"a": 1},
       array_attr=np.arange(3),
       values=np.zeros(4),
       nested=Nested("n"),
       note=None,
   )
   h5t.save(record, "schema.h5", mode="w")
   print(h5t.load(Result, "schema.h5"))

Missing members, defaults and optionals
---------------------------------------

A member absent from the file follows these rules, in order:

1. a declared default is used (and validated); a ``default_factory`` is called first;
2. a ``T | None`` annotation gives ``None``;
3. otherwise :class:`h5t.ValidationError` is raised.

For dataclasses, defaults are read from ``dataclasses.fields()``; for other types, from
the declaring class and then the constructor signature.

Attribute conversion and validation
-----------------------------------

An attribute's raw h5py value goes through ``Attr(converter=...)`` first (a failure is a
:class:`h5t.ConversionError`), then through a Pydantic ``TypeAdapter`` (a failure is a
:class:`h5t.ValidationError`). Pydantic is used only for validation, never as a
``BaseModel``.

Extras policy
-------------

Every decorated record accepts ``extras="ignore"`` (the default) or ``extras="forbid"``.
A group policy applies to its immediate child and attribute names. A dataset policy
applies to its attributes. Nested schemas keep their own policy, while a plain
``np.ndarray`` field never checks the dataset's attributes.

Attribute snapshots (``attrs=``)
--------------------------------

Pass ``attrs="field"`` to the decorator to bind a ``Mapping``-annotated field to the
node's full attribute snapshot: an immutable mapping keyed by on-disk HDF5 name, where
declared attributes hold their Pydantic-processed values and undeclared ones (retained
under ``extras="ignore"``) hold the raw h5py values. It lists every declared name,
including ones absent from the file that fell back to ``None`` or to a default.

.. code-block:: python

   import dataclasses
   from collections.abc import Mapping
   from typing import Any

   import h5py

   import h5t

   with h5py.File("attrs.h5", "w") as f:
       f.attrs["version"] = 3
       f.attrs["extra"] = "kept"


   @h5t.group(attrs="attrs")
   @dataclasses.dataclass
   class Result:
       version: int
       attrs: Mapping[str, Any]


   print(dict(h5t.load(Result, "attrs.h5").attrs))

No reserved field names
-----------------------

A record type has no reserved field names: ``data``, ``path``, ``shape`` and ``attrs``
are all free, because h5t contributes no base class for a field name to shadow. Only
repeated *HDF5* names within a record are rejected. Annotations starting with ``_`` are
not treated as fields.

Errors
------

All errors derive from :class:`h5t.H5TError`:

- :class:`h5t.SchemaError`: a declaration is incoherent. Usually raised at the decorator.
- :class:`h5t.ValidationError`: the file does not conform. It is fail-fast and
  path-aware, e.g. ``/fiducials: the size of dimension n is 4 which does not equal the
  existing value of 3``.
- :class:`h5t.ConversionError`: an ``Attr`` converter or serializer failed; a subclass of
  ``ValidationError``.

Records are built with their real constructor, so ``__post_init__`` runs. An
invariant it raises surfaces as a :class:`h5t.ValidationError`; only a failure to bind
constructor arguments is a :class:`h5t.SchemaError`.
