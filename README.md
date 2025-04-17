# rshf
[![PyPI - Version](https://img.shields.io/pypi/v/rshf)](https://pypi.org/project/rshf/)
[![PyPI Downloads](https://static.pepy.tech/badge/rshf)](https://pypistats.org/packages/rshf)
[![PyPI Docs](https://img.shields.io/readthedocs/rshf)](https://rshf-docs.readthedocs.io/en/latest/)


### Remote sensing pretrained models easy loading using huggingface -- PyTorch (for fast benchmarking)

### Installation:
```bash
pip install rshf
```

### Example:
```python
from rshf.satmae import SatMAE
model = SatMAE.from_pretrained("MVRL/satmae-vitlarge-fmow-pretrain-800")
input = model.transform(torch.randint(0, 256, (224, 224, 3)).float().numpy(), 224).unsqueeze(0)
print(model.forward_encoder(input, mask_ratio=0.0)[0].shape)
```

### TODO:
- [ ] Add transforms for each model
- [ ] Add Documentation (https://rshf-docs.readthedocs.io/en/latest/)
- [x] Add initial set of models

### Citations

|Model Type|Venue|Citation|
|----------|-----|--------|
|BioCLIP|CVPR'24|[link](./rshf/bioclip/README.md)
|Climplicit|ICLRW'25|[link](./rshf/climplicit/README.md)
|CLIP|ICML'21|[link](./rshf/clip/README.md)
|CROMA|NeurIPS'23|[link](./rshf/croma/README.md)
|GeoCLAP|BMVC'23|[link](./rshf/geoclap/README.md)
|GeoCLIP|NeurIPS'23|[link](./rshf/geoclip/README.md)
|Presto||[link](./rshf/presto/README.md)
|Prithvi||[link](./rshf/prithvi/README.md)
|RemoteCLIP|TGRS'23|[link](./rshf/remoteclip/README.md)
|RVSA|TGRS'22|[link](./rshf/rvsa/README.md)
|Sat2Cap|EarthVision'24|[link](./rshf/sat2cap/README.md)
|SatClip|AAAI'25|[link](./rshf/satclip/README.md)
|SatMAE|NeurIPS'22|[link](./rshf/satmae/README.md)
|SatMAE++|CVPR'24|[link](./rshf/satmaepp/README.md)
|ScaleMAE|ICCV'23|[link](./rshf/scalemae/README.md)
|SenCLIP|WACV'25|[link](./rshf/senclip/README.md)
|SINR|ICML'23|[link](./rshf/sinr/README.md)
|StreetCLIP||[link](./rshf/streetclip/README.md)
|TaxaBind|WACV'25|[link](./rshf/taxabind/README.md)
### List of models available here: [Link](https://huggingface.co/collections/MVRL/remote-sensing-foundation-models-664e8fcd67d8ca8c03f42d00)
