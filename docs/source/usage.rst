Usage
=====

Minimal example:

.. code-block:: python

   import torch
   from rshf.satmae import SatMAE

   model = SatMAE.from_pretrained("MVRL/satmae-vitlarge-fmow-pretrain-800")
   image = torch.randint(0, 256, (224, 224, 3)).float().numpy()
   batch = model.transform(image, 224).unsqueeze(0)
   embedding = model.forward_encoder(batch, mask_ratio=0.0)[0]
   print(embedding.shape)
