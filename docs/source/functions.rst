Functions and helpers
=====================

This page documents top-level helper functions and module-level utility
functions exposed by ``rshf``.

Top-level package functions
---------------------------

``rshf.list_models(model_name: str) -> None``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Print model repositories in the curated collection whose IDs contain
``model_name``.

.. code-block:: python

   from rshf import list_models
   list_models("geoclip")

``rshf.from_config(model_class, repo_id, revision=None, **kwargs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download ``config.json`` from a repository and instantiate ``model_class``
using that architecture configuration with random weights.

.. code-block:: python

   from rshf import from_config
   from rshf.satmae import SatMAE

   model = from_config(SatMAE, "MVRL/satmae-vitlarge-fmow-pretrain-800")

``rshf.utils.help(model) -> None``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Print the target model class docstring.

.. code-block:: python

   from rshf.utils import help as print_help
   from rshf.sinr import SINR

   print_help(SINR)

SINR helpers
------------

``rshf.sinr.SINRConfig``
~~~~~~~~~~~~~~~~~~~~~~~~

Configuration class for constructing ``SINR`` from scratch.

``rshf.sinr.preprocess_locs(locs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert lon/lat inputs into the sinusoidal feature representation expected by
``SINR``.

.. code-block:: python

   import torch
   from rshf.sinr import SINR, preprocess_locs

   model = SINR.from_pretrained("<hf-repo-id>")
   locs = torch.tensor([[-80.0, 40.0]], dtype=torch.float32)
   features = preprocess_locs(locs)
   embeddings = model(features)

GeoCLIP helpers
---------------

``rshf.geoclip.GeoCLIPConfig``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration class for constructing ``GeoCLIP`` from scratch.

.. code-block:: python

   from rshf.geoclip import GeoCLIP, GeoCLIPConfig

   config = GeoCLIPConfig(sigma=[2, 4], input_size=2, encoded_size=256, dim=512)
   model = GeoCLIP(config)

TaxaBind helper methods
-----------------------

``TaxaBind`` is initialized with configuration and provides modular getter
functions:

- ``get_image_text_encoder()``
- ``get_tokenizer()``
- ``get_image_processor()``
- ``get_audio_encoder()``
- ``get_audio_processor()``
- ``process_audio(track, sr)``
- ``get_location_encoder()``
- ``get_env_encoder()``
- ``get_sat_encoder()``
- ``get_sat_processor()``
