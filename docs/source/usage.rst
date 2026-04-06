Usage
=====

This page covers the common loading patterns used across the package.
For an exhaustive per-model list, see :doc:`model_loading`.

Common loading pattern
----------------------

Most `rshf` classes are loaded through Hugging Face Hub:

.. code-block:: python

   from rshf.satmae import SatMAE

   model = SatMAE.from_pretrained("MVRL/satmae-vitlarge-fmow-pretrain-800")

Finding available repositories
------------------------------

Use :func:`rshf.list_models` with a keyword:

.. code-block:: python

   from rshf import list_models

   list_models("geoclip")
   list_models("satmae")

Building from architecture config only
--------------------------------------

Use :func:`rshf.from_config` when you want random initialization with a known
model architecture:

.. code-block:: python

   from rshf import from_config
   from rshf.satmae import SatMAE

   model = from_config(SatMAE, "MVRL/satmae-vitlarge-fmow-pretrain-800")
