Schemas
=======

A schema is a subclass of :class:`h5t.Group` (a :class:`pydantic.BaseModel` subclass).
``h5t`` provides two symmetrical generic primitives that mirror ``h5py.Group`` and
``h5py.Dataset``:

- :class:`h5t.Group` / ``h5t.Group[T]``: stores group attributes on ``.attrs: T`` (defaulting
  to ``Mapping[str, Any]`` or a :class:`pydantic.BaseModel` subclass ``T``) and treats all
  other declared fields as immediate HDF5 child nodes (subgroups, datasets, or collections).
- :class:`h5t.Dataset` / ``h5t.Dataset[T]``: the same ``.attrs: T`` convention for a
  detached, lazily-indexed HDF5 dataset; see :doc:`arrays`.

Child field rules on ``Group``
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 20 48

   * - Annotation on ``Group``
     - HDF5 representation
     - Loading behavior
   * - :class:`h5t.Group` subclass
     - child group (or the root)
     - recursively materialized
   * - :class:`h5t.Dataset` or ``h5t.Dataset[T]``
     - child dataset
     - metadata and ``attrs`` loaded; array data read lazily when indexed
   * - ``np.ndarray``
     - child dataset
     - complete array loaded eagerly as an ``ndarray``
   * - ``Float[np.ndarray, "n n"]``, ``Float[h5t.Dataset[T], "n _"]`` (jaxtyping)
     - child dataset
     - dtype and shape checked against the file's metadata before any read; otherwise
       as ``np.ndarray`` / ``Dataset[T]``
   * - ``dict[str, T]`` / ``Mapping[str, T]``
     - child group named by the field
     - one item per child of that group, as a plain ``dict``; see :doc:`collections`

Use Pydantic's ``Field(alias="...")`` to map any child field or attribute field to a different
on-disk HDF5 name. A parameterized alias such as ``numpy.typing.NDArray[np.floating]`` is
accepted wherever ``np.ndarray`` is; for ``NDArray`` the dtype parameter is not validated (use
:doc:`jaxtyping` to check it).

.. code-block:: python

   import json
   from typing import Annotated, Any

   import numpy as np
   from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

   import h5t


   class NestedAttrs(BaseModel):
       label: str


   class Nested(h5t.Group[NestedAttrs]):
       pass


   class ResultAttrs(BaseModel):
       version: int
       title: Annotated[str, Field(alias="name")]
       config: Annotated[
           dict[str, Any],
           BeforeValidator(json.loads),
           PlainSerializer(json.dumps),
       ]
       array_attr: np.ndarray
       note: str | None = None
       revision: int = 1


   class Result(h5t.Group[ResultAttrs]):
       model_config = ConfigDict(extra="ignore")

       values: np.ndarray
       nested: Nested


   group = Result(
       attrs=ResultAttrs(
           version=2,
           title="demo",
           config={"a": 1},
           array_attr=np.arange(3),
       ),
       values=np.zeros(4),
       nested=Nested(attrs=NestedAttrs(label="n")),
   )
   group.model_dump_hdf5("schema.h5", mode="w")
   print(Result.model_validate_hdf5("schema.h5"))

Missing members, defaults and optionals
---------------------------------------

A child node or attribute absent from the file follows Pydantic's rules:

1. a declared default is used as written (it is validated only with
   ``Field(validate_default=True)``); a ``Field(default_factory=...)`` is called instead;
2. otherwise the field is required and :class:`h5t.ValidationError` is raised, even for
   ``T | None``: write ``T | None = None`` for an optional member.

Attribute conversion and validation
-----------------------------------

Declare typed attributes on a :class:`pydantic.BaseModel` subclass ``T`` and pass it to
``h5t.Group[T]`` or ``h5t.Dataset[T]``. Standard Pydantic validators (``BeforeValidator``,
``AfterValidator``), serializers (``PlainSerializer``), and ``Field(...)`` constraints work
directly on attribute fields.

If ``T`` is omitted (bare ``h5t.Group`` or ``h5t.Dataset``), ``.attrs`` is a read-only
``Mapping[str, Any]`` containing the raw ``h5py`` attribute values.

Extras policy
-------------

Configure ``model_config = ConfigDict(extra="ignore")`` (the default) or
``ConfigDict(extra="forbid")``:

- On a :class:`h5t.Group` subclass, ``extra="forbid"`` rejects undeclared HDF5 child nodes
  (and any attributes if the group does not declare a typed ``Group[T]`` model).
- On an attributes :class:`pydantic.BaseModel` ``T`` in ``Group[T]`` or ``Dataset[T]``,
  ``extra="forbid"`` rejects undeclared HDF5 attributes on that group or dataset.

Errors
------

All errors derive from :class:`h5t.H5TError`:

- :class:`h5t.SchemaError`: a schema class declaration is invalid. Raised at class creation.
- :class:`h5t.ValidationError`: the file or model does not conform. Fail-fast and
  path-aware, e.g. ``/fiducials: the size of dimension n is 4 which does not equal the
  existing value of 3``. Messages from Pydantic keep its wording with the location turned
  into an HDF5 path: ``/values: Field required``, ``/@count: Input should be a valid integer``.

A ``mode="before"`` model validator on a ``Group`` sees a read-only mapping over the HDF5
group (children by name, attributes under ``"/attrs"``), not a ``dict``.
