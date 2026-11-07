User guide
==========

- :doc:`schemas`: ``Group[T]``, ``Dataset[T]``, field rules, defaults, extras policy and errors.
- :doc:`arrays`: ``Dataset[T]`` versus ``np.ndarray``, slicing, caching and snapshot semantics.
- :doc:`jaxtyping`: dtype and shape checks from file metadata.
- :doc:`collections`: ``dict[str, T]`` fields and ``Glob``.
- :doc:`writing`: ``Group.model_dump_hdf5``.
- :doc:`compilation`: Pydantic model lifecycle, forward references and explicit rebuilding.
- :doc:`cli`: the ``h5t check`` command.

.. toctree::
   :hidden:
   :maxdepth: 1

   schemas
   arrays
   jaxtyping
   collections
   writing
   compilation
   cli
