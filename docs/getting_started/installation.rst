Installation
============

.. code-block:: bash

   pip install h5t     # or: uv add h5t

``h5t`` requires Python 3.11 or newer. ``h5py``, ``numpy``, ``jaxtyping`` and Pydantic v2
are installed automatically.

Building these docs
-------------------

.. code-block:: bash

   uv sync --group docs
   just docs build     # warnings are errors
   just docs open      # build, then open in a browser
