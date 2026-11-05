Dynamic collections
===================

Real files often hold one group or dataset per sample, with names a schema cannot list.
A ``dict[str, T]`` or ``Mapping[str, T]`` field (no :class:`h5t.Attr`) mirrors the HDF5
tree: it names **one child group** (:class:`h5t.Name` is honoured), and its value is a
plain ``dict`` with one item per child of that group, in HDF5 iteration order.
Attributes on the collection's own group are ignored.

``T`` is any group/dataset record, ``Payload`` type, ``LazyArray``, ``np.ndarray`` or
jaxtyping array.

Filtering with ``Glob``
-----------------------

``Annotated[..., h5t.Glob("chain_?")]`` keeps only children whose name matches the
shell-style pattern (case-sensitive; it may not be empty or contain ``/``).

.. code-block:: python

   import dataclasses
   from typing import Annotated

   import h5py
   import numpy as np

   import h5t

   with h5py.File("runs.h5", "w") as f:
       for name in ("chain_1", "chain_2", "other"):
           f.create_dataset(f"chains/{name}", data=np.arange(3.0))
       f.create_group("snaps/s1").attrs["step"] = 1
       f.create_group("snaps/s2").attrs["step"] = 2


   @h5t.group()
   @dataclasses.dataclass
   class Snapshot:
       step: int


   @h5t.group()
   @dataclasses.dataclass
   class Run:
       snaps: dict[str, Snapshot]
       chains: Annotated[dict[str, h5t.LazyArray], h5t.Glob("chain_?")]
       raw: Annotated[dict[str, Snapshot], h5t.Name("Raw Data")] = dataclasses.field(
           default_factory=dict
       )


   run = h5t.load(Run, "runs.h5")
   print(list(run.snaps), list(run.chains), run.raw)

Rules and limits
----------------

- A missing group follows the usual rules (``| None``, a default or ``default_factory``,
  else a :class:`h5t.ValidationError`).
- Keys must be ``str``. Items cannot be scalars, optional, nested collections, or carry
  ``Name`` / ``Attr`` / ``Glob``.
- Bare array items share the owner record's jaxtyping dimension bindings; record items
  bind independently.
- Layouts with per-sample groups at the file *root* are not covered yet.
- On :func:`h5t.save`, collection keys must be non-empty, contain no ``/`` and match the
  field's ``Glob``; otherwise the item would not load back.
