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

### Citations

|Model Type|Venue|Citation|
|----------|-----|--------|
|BioCLIP|CVPR'24|[link](./rshf/bioclip/README.md)
|CLIP|ICML'21|[link](./rshf/clip/README.md)
|CROMA|NeurIPS'23|[link](./rshf/croma/README.md)
|GeoCLAP|BMVC'23|[link](./rshf/geoclap/README.md)
|GeoCLIP|NeurIPS'23|[link](./rshf/geoclip/README.md)
|Presto||[link](./rshf/presto/README.md)
|Prithvi||[link](./rshf/prithvi/README.md)
|RemoteCLIP|TGRS'23|[link](./rshf/remoteclip/README.md)
|RVSA|TGRS'22|[link](./rshf/rvsa/README.md)
|SatClip||[link](./rshf/satclip/README.md)
|SatMAE|NeurIPS'22|[link](./rshf/satmae/README.md)
|SatMAE++|CVPR'24|[link](./rshf/satmaepp/README.md)
|ScaleMAE|ICCV'23|[link](./rshf/scalemae/README.md)
|SINR|ICML'23|[link](./rshf/sinr/README.md)
### List of models available here: [Link](https://huggingface.co/collections/MVRL/remote-sensing-foundation-models-664e8fcd67d8ca8c03f42d00)