Model loading guide
===================

This page documents how to load each model exported by ``rshf``.

Discover repository IDs
-----------------------

When a model uses ``from_pretrained``, pass a Hugging Face repository ID.
You can find matching IDs in the curated collection with:

.. code-block:: python

   from rshf import list_models
   list_models("satmae")

The underlying collection is:
``MVRL/remote-sensing-foundation-models-664e8fcd67d8ca8c03f42d00``.

Per-model loading reference
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Model
     - Import
     - Load
   * - BioCLIP
     - ``from rshf.bioclip import BioCLIP``
     - ``model = BioCLIP()``
   * - Climplicit
     - ``from rshf.climplicit import Climplicit``
     - ``model = Climplicit.from_pretrained("<hf-repo-id>")``
   * - CLIP
     - ``from rshf.clip import CLIPModel``
     - ``model = CLIPModel.from_pretrained("<hf-repo-id>")``
   * - CROMA
     - ``from rshf.croma import CROMA``
     - ``model = CROMA.from_pretrained("<hf-repo-id>")``
   * - DinoV3 Sat
     - ``from rshf.dinov3_sat import Dinov3_Sat``
     - ``model = Dinov3_Sat.from_pretrained("<hf-repo-id>")``
   * - GeoCLAP
     - ``from rshf.geoclap import GeoCLAP``
     - ``model = GeoCLAP.from_pretrained("<hf-repo-id>")``
   * - GeoCLIP
     - ``from rshf.geoclip import GeoCLIP``
     - ``model = GeoCLIP.from_pretrained("<hf-repo-id>")``
   * - Presto
     - ``from rshf.presto import Presto``
     - ``model = Presto.from_pretrained("<hf-repo-id>")``
   * - Prithvi
     - ``from rshf.prithvi import Prithvi``
     - ``model = Prithvi.from_pretrained("<hf-repo-id>")``
   * - ProM3E
     - ``from rshf.prom3e import ProM3E``
     - ``model = ProM3E.from_pretrained("<hf-repo-id>")``
   * - RCME
     - ``from rshf.rcme import RCME``
     - ``model = RCME()``
   * - RemoteCLIP
     - ``from rshf.remoteclip import RemoteCLIP``
     - ``model = RemoteCLIP("ViT-B-32")``
   * - RVSA
     - ``from rshf.rvsa import RVSA``
     - ``model = RVSA.from_pretrained("<hf-repo-id>")``
   * - Sat2Cap
     - ``from rshf.sat2cap import Sat2Cap``
     - ``model = Sat2Cap.from_pretrained("<hf-repo-id>")``
   * - SatClip
     - ``from rshf.satclip import SatClip``
     - ``model = SatClip.from_pretrained("<hf-repo-id>")``
   * - SatMAE
     - ``from rshf.satmae import SatMAE``
     - ``model = SatMAE.from_pretrained("MVRL/satmae-vitlarge-fmow-pretrain-800")``
   * - SatMAE multi-spectral pretrain
     - ``from rshf.satmae import SatMAE_Pre_MS``
     - ``model = SatMAE_Pre_MS.from_pretrained("<hf-repo-id>")``
   * - SatMAE multi-spectral finetune
     - ``from rshf.satmae import SatMAE_Fine_MS``
     - ``model = SatMAE_Fine_MS.from_pretrained("<hf-repo-id>")``
   * - SatMAE++
     - ``from rshf.satmaepp import SatMAEPP``
     - ``model = SatMAEPP.from_pretrained("<hf-repo-id>")``
   * - ScaleMAE
     - ``from rshf.scalemae import ScaleMAE``
     - ``model = ScaleMAE.from_pretrained("<hf-repo-id>")``
   * - SenCLIP
     - ``from rshf.senclip import SenCLIP``
     - ``model = SenCLIP.from_pretrained("<hf-repo-id>")``
   * - SINR
     - ``from rshf.sinr import SINR``
     - ``model = SINR.from_pretrained("<hf-repo-id>")``
   * - StreetCLIP
     - ``from rshf.streetclip import StreetCLIP``
     - ``model = StreetCLIP.from_pretrained("<hf-repo-id>")``
   * - TaxaBind
     - ``from rshf.taxabind import TaxaBind``
     - ``model = TaxaBind(config)``

Additional CLIP exports
-----------------------

The ``rshf.clip`` module also exports ``CLIPProcessor``, ``CLIPTextModel``,
and ``CLIPVisionModel``:

.. code-block:: python

   from rshf.clip import CLIPProcessor, CLIPTextModel, CLIPVisionModel

   processor = CLIPProcessor.from_pretrained("<hf-repo-id>")
   text_encoder = CLIPTextModel.from_pretrained("<hf-repo-id>")
   vision_encoder = CLIPVisionModel.from_pretrained("<hf-repo-id>")
