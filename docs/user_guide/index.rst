User guide
==========

- :doc:`schemas`: records, field rules, defaults, extras policy and errors.
- :doc:`arrays`: ``np.ndarray`` versus ``LazyArray`` versus ``Eager``, and snapshot semantics.
- :doc:`jaxtyping`: dtype and shape checks from file metadata.
- :doc:`collections`: ``dict[str, T]`` fields and ``Glob``.
- :doc:`payload`: loading undecoratable types with ``Payload``.
- :doc:`writing`: ``h5t.save``.
- :doc:`compilation`: when schemas compile, forward references and ``localns``.
- :doc:`cli`: the ``h5t check`` command.

.. toctree::
   :hidden:
   :maxdepth: 1

   schemas
   arrays
   jaxtyping
   collections
   payload
   writing
   compilation
   cli
