# rshf
### Remote sensing pretrained models easy loading using huggingface -- PyTorch (for fast benchmarking)

### Installation:
```bash
pip install rshf
```

### Example:
```python
from rshf.satmae import SatMAE
model = SatMAE.from_pretrained("MVRL/satmae-vitlarge-fmow-pretrain-800")
print(model.forward_encoder(torch.randn(1, 3, 224, 224), mask_ratio=0.0)[0].shape)
```

### List of models available here: [Link](https://huggingface.co/collections/MVRL/remote-sensing-foundation-models-664e8fcd67d8ca8c03f42d00)