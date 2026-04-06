Installation
============

Install from PyPI
-----------------

.. code-block:: bash

   pip install rshf

Basic import check
------------------

.. code-block:: python

   from rshf import list_models
   from rshf.satmae import SatMAE

   # Prints matching repositories from the curated collection.
   list_models("satmae")

   # Load a pretrained checkpoint.
   model = SatMAE.from_pretrained("MVRL/satmae-vitlarge-fmow-pretrain-800")
