## RANGE

RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings

### Overview

RANGE is a retrieval-augmented framework for embedding geographic coordinates that directly estimates visual features for given locations. Unlike traditional contrastive methods (like SatCLIP or GeoCLIP), RANGE estimates visual features by combining features from multiple similar-looking locations in a database, allowing representations to capture high-resolution information at multiple scales.

### Key Features

- **Multi-resolution embeddings**: Generate geo-embeddings at desired frequency levels by manipulating spatial smoothness constraints
- **Retrieval-augmented**: Combines visual features from similar locations for richer representations
- **State-of-the-art performance**: Outperforms SatCLIP, GeoCLIP, and CSP by up to 13.1% on classification tasks

### Models

- **RANGE**: Semantic retrieval-based geo-embeddings
- **RANGE+**: Enhanced version combining semantic and geographic similarity

### Usage

```python
from rshf.range import RANGE, RANGEConfig

# RANGE+ with custom parameters
config = RANGEConfig(
    model_type="RANGE+",
    beta=0.5,              # Interpolation between semantic and geographic
    temp=12.0,             # Semantic similarity temperature
    geo_temp=40.0,         # Geographic similarity temperature
    db_size="large"        # Database size: 'large' or 'med'
)
model = RANGE(config)

# Generate embeddings for locations (lon, lat in degrees)
locs = torch.tensor([[40.0, -80.0], [45.0, -90.0]]).double()
embeddings = model(locs)
print(embeddings.shape)  # (2, 1280)

# Or use standard RANGE
config = RANGEConfig(model_type="RANGE")
model = RANGE(config)
```

### Database

RANGE requires a precomputed database of location embeddings. The database is automatically downloaded from HuggingFace when needed. Two database sizes are available:

- `range_db_large.npz`: Full database for best performance
- `range_db_med.npz`: Medium-sized database for faster inference

You can also provide a custom database path:

```python
embeddings = model(locs, db_path="/path/to/range_db_large.npz")
```

### References

**Paper**: https://arxiv.org/abs/2502.19781 \
**GitHub**: https://github.com/mvrl/RANGE \
**HuggingFace**: https://huggingface.co/collections/MVRL/range-67e99fa1dfc6c86a3b872c09

### Citation

```bibtex
@inproceedings{dhakal2025range,
  title={RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings},
  author={Dhakal, Aayush and Sastry, Srikumar and Khanal, Subash and Ahmad, Adeel and Xing, Eric and Jacobs, Nathan},
  booktitle={Computer Vision and Pattern Recognition},
  year={2025},
  organization={IEEE/CVF}
}
```
