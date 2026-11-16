Dynamic collections
===================

Real files often hold one group or dataset per sample, with names a schema cannot list.
A ``dict[str, T]`` or ``Mapping[str, T]`` field on a :class:`h5t.Group` mirrors the HDF5
tree: it names **one child group** (Pydantic's ``Field(alias=...)`` is honoured), and its
value is a plain ``dict`` with one item per child of that group, in HDF5 iteration order.
Attributes on the collection's own group are ignored.

``T`` is any :class:`h5t.Group` subclass, :class:`h5t.Dataset` (or ``Dataset[Attrs]``),
``np.ndarray``, or jaxtyping array.

.. code-block:: python

   from typing import Annotated

   import h5py
   import numpy as np
   from pydantic import BaseModel, Field

   import h5t

   with h5py.File("runs.h5", "w") as f:
       for name in ("chain_1", "chain_2"):
           f.create_dataset(f"chains/{name}", data=np.arange(3.0))
       f.create_group("snaps/s1").attrs["step"] = 1
       f.create_group("snaps/s2").attrs["step"] = 2


   class SnapshotAttrs(BaseModel):
       step: int


   class Snapshot(h5t.Group[SnapshotAttrs]):
       pass


   class Run(h5t.Group):
       snaps: dict[str, Snapshot]
       chains: dict[str, h5t.Dataset]
       raw: Annotated[dict[str, Snapshot], Field(alias="Raw Data")] = Field(
           default_factory=dict
       )


   run = Run.model_validate_hdf5("runs.h5")
   print(list(run.snaps), list(run.chains), run.raw)

Rules and limits
----------------

- A missing group follows the usual rules (a default or ``default_factory``, else a
  :class:`h5t.ValidationError`; ``| None`` alone does not make it optional).
- Keys must be ``str``. Items cannot be scalars, optional, nested collections, or carry
  ``Field(alias=...)``.
- Array and ``Dataset`` items share the owner group's jaxtyping dimension bindings; nested
  ``Group`` items bind independently.
- On :meth:`h5t.Group.model_dump_hdf5`, collection keys must be non-empty, contain no ``/``,
  and not be ``"."`` or ``".."``; otherwise the item would not load back.
