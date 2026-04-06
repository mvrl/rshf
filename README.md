# rshf

[![PyPI - Version](https://img.shields.io/pypi/v/rshf)](https://pypi.org/project/rshf/)
[![PyPI Downloads](https://static.pepy.tech/badge/rshf)](https://pypistats.org/packages/rshf)
[![PyPI Docs](https://img.shields.io/readthedocs/rshf)](https://rshf-docs.readthedocs.io/en/latest/)

`rshf` provides a single Python package for loading remote sensing foundation
models from Hugging Face and related backends.

This documentation focuses on:

1. How to load every model exposed by `rshf`.
2. What utility functions are available in the package.

## Installation

```bash
pip install rshf
```

## Quick start

```python
from rshf import list_models
from rshf.satmae import SatMAE

# Print matching model repositories in the curated collection.
list_models("satmae")

# Load pretrained weights.
model = SatMAE.from_pretrained("MVRL/satmae-vitlarge-fmow-pretrain-800")
```

## Model loading reference

Most classes in `rshf` are loaded with:

```python
model = ModelClass.from_pretrained("<hf-repo-id>")
```

Use `list_models("<keyword>")` to discover matching repositories in the
collection:
https://huggingface.co/collections/MVRL/remote-sensing-foundation-models-664e8fcd67d8ca8c03f42d00

### Per-model imports and loaders

| Model | Import | Load |
|---|---|---|
| BioCLIP | `from rshf.bioclip import BioCLIP` | `model = BioCLIP()` |
| Climplicit | `from rshf.climplicit import Climplicit` | `model = Climplicit.from_pretrained("<hf-repo-id>")` |
| CLIP (generic) | `from rshf.clip import CLIPModel` | `model = CLIPModel.from_pretrained("<hf-repo-id>")` |
| CROMA | `from rshf.croma import CROMA` | `model = CROMA.from_pretrained("<hf-repo-id>")` |
| DinoV3 Sat | `from rshf.dinov3_sat import Dinov3_Sat` | `model = Dinov3_Sat.from_pretrained("<hf-repo-id>")` |
| GeoCLAP | `from rshf.geoclap import GeoCLAP` | `model = GeoCLAP.from_pretrained("<hf-repo-id>")` |
| GeoCLIP | `from rshf.geoclip import GeoCLIP` | `model = GeoCLIP.from_pretrained("<hf-repo-id>")` |
| Presto | `from rshf.presto import Presto` | `model = Presto.from_pretrained("<hf-repo-id>")` |
| Prithvi | `from rshf.prithvi import Prithvi` | `model = Prithvi.from_pretrained("<hf-repo-id>")` |
| ProM3E | `from rshf.prom3e import ProM3E` | `model = ProM3E.from_pretrained("<hf-repo-id>")` |
| RCME | `from rshf.rcme import RCME` | `model = RCME()` |
| RemoteCLIP | `from rshf.remoteclip import RemoteCLIP` | `model = RemoteCLIP("ViT-B-32")` |
| RVSA | `from rshf.rvsa import RVSA` | `model = RVSA.from_pretrained("<hf-repo-id>")` |
| Sat2Cap (vision encoder) | `from rshf.sat2cap import Sat2Cap` | `model = Sat2Cap.from_pretrained("<hf-repo-id>")` |
| SatClip | `from rshf.satclip import SatClip` | `model = SatClip.from_pretrained("<hf-repo-id>")` |
| SatMAE | `from rshf.satmae import SatMAE` | `model = SatMAE.from_pretrained("MVRL/satmae-vitlarge-fmow-pretrain-800")` |
| SatMAE multi-spectral pretrain | `from rshf.satmae import SatMAE_Pre_MS` | `model = SatMAE_Pre_MS.from_pretrained("<hf-repo-id>")` |
| SatMAE multi-spectral finetune | `from rshf.satmae import SatMAE_Fine_MS` | `model = SatMAE_Fine_MS.from_pretrained("<hf-repo-id>")` |
| SatMAE++ | `from rshf.satmaepp import SatMAEPP` | `model = SatMAEPP.from_pretrained("<hf-repo-id>")` |
| ScaleMAE | `from rshf.scalemae import ScaleMAE` | `model = ScaleMAE.from_pretrained("<hf-repo-id>")` |
| SenCLIP | `from rshf.senclip import SenCLIP` | `model = SenCLIP.from_pretrained("<hf-repo-id>")` |
| SINR | `from rshf.sinr import SINR` | `model = SINR.from_pretrained("<hf-repo-id>")` |
| StreetCLIP | `from rshf.streetclip import StreetCLIP` | `model = StreetCLIP.from_pretrained("<hf-repo-id>")` |
| TaxaBind | `from rshf.taxabind import TaxaBind` | `model = TaxaBind(config)` |

### Additional CLIP components

```python
from rshf.clip import CLIPProcessor, CLIPTextModel, CLIPVisionModel

processor = CLIPProcessor.from_pretrained("<hf-repo-id>")
text_encoder = CLIPTextModel.from_pretrained("<hf-repo-id>")
vision_encoder = CLIPVisionModel.from_pretrained("<hf-repo-id>")
```

## Package functions and helpers

### `rshf.list_models(model_name: str) -> None`

Print repositories from the curated collection whose IDs contain `model_name`.

```python
from rshf import list_models

list_models("geoclip")
```

### `rshf.from_config(model_class, repo_id, revision=None, **kwargs)`

Download `config.json` from a model repo and instantiate a model with random
weights (architecture only, no pretrained checkpoint weights).

```python
from rshf import from_config
from rshf.satmae import SatMAE

model = from_config(SatMAE, "MVRL/satmae-vitlarge-fmow-pretrain-800")
```

### `rshf.utils.help(model) -> None`

Print a model class docstring.

```python
from rshf.utils import help as print_help
from rshf.sinr import SINR

print_help(SINR)
```

### SINR utilities

```python
from rshf.sinr import SINR, SINRConfig, preprocess_locs

model = SINR.from_pretrained("<hf-repo-id>")
locs = preprocess_locs(...)
embeddings = model(locs)
```

### GeoCLIP config utility

```python
from rshf.geoclip import GeoCLIP, GeoCLIPConfig

config = GeoCLIPConfig(sigma=[2, 4], input_size=2, encoded_size=256, dim=512)
model = GeoCLIP(config)
```

### TaxaBind loader functions

`TaxaBind` is config-driven and exposes dedicated getters:

- `get_image_text_encoder()`
- `get_tokenizer()`
- `get_image_processor()`
- `get_audio_encoder()`
- `get_audio_processor()`
- `get_location_encoder()`
- `get_env_encoder()`
- `get_sat_encoder()`
- `get_sat_processor()`
- `process_audio(track, sr)`

## Citations

Model-specific references are available in each model folder:

- `rshf/bioclip/README.md`
- `rshf/climplicit/README.md`
- `rshf/clip/README.md`
- `rshf/croma/README.md`
- `rshf/dinov3_sat/README.md`
- `rshf/geoclap/README.md`
- `rshf/geoclip/README.md`
- `rshf/presto/README.md`
- `rshf/prithvi/README.md`
- `rshf/prom3e/README.md`
- `rshf/rcme/README.md`
- `rshf/remoteclip/README.md`
- `rshf/rvsa/README.md`
- `rshf/sat2cap/README.md`
- `rshf/satclip/README.md`
- `rshf/satmae/README.md`
- `rshf/satmaepp/README.md`
- `rshf/scalemae/README.MD`
- `rshf/senclip/README.md`
- `rshf/sinr/README.md`
- `rshf/streetclip/README.md`
- `rshf/taxabind/README.md`
