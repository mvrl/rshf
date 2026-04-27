## Clay

Clay is an open-source foundation model for Earth observation that processes satellite imagery to generate semantic embeddings. It uses a Vision Transformer architecture adapted to understand geospatial and temporal relations in satellite data, trained via self-supervised learning using a Masked Autoencoder (MAE) method.

### Key Features

- **Sensor-Agnostic**: Works with multiple satellite instruments including Sentinel-2, Landsat, NAIP, Sentinel-1 SAR, and MODIS
- **Temporal & Spatial Understanding**: Incorporates location and time information into embeddings
- **Multiple Use Cases**: Generate embeddings, fine-tune for classification/regression, or use as a backbone for other models
- **Pre-trained Weights**: Available on HuggingFace at [made-with-clay/Clay](https://huggingface.co/made-with-clay/Clay)

### Installation

Install RSHF with Clay dependencies:

```bash
pip install rshf
```

The Clay model package is automatically installed as a dependency.

### Usage

```python
from rshf.clay import Clay
import torch

# Load pretrained model from HuggingFace Hub (automatically downloads checkpoint)
model = Clay.from_pretrained("MVRL/clay-v1.5")
model.eval()

# Prepare input data
# Sentinel-2 L2A example with 10 bands
chips = torch.randn(1, 10, 256, 256)  # [batch, bands, height, width]

# Wavelengths in nanometers for Sentinel-2 bands
# [Blue, Green, Red, RedEdge1, RedEdge2, RedEdge3, NIR, RedEdge4, SWIR1, SWIR2]
wavelengths = torch.tensor([[485, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]], 
                           dtype=torch.float32)

# Temporal and spatial metadata: [week, hour, lat, lon]
timestamps = torch.zeros(1, 4)

# Generate embeddings
with torch.no_grad():
    embeddings = model.encoder(chips, timestamps, wavelengths)
    
print(f"Embeddings shape: {embeddings.shape}")  # [1, 1024]
```

### Advanced Usage

#### Reconstruction

```python
# Reconstruct masked patches
with torch.no_grad():
    outputs = model.reconstruct(chips, timestamps, wavelengths)
```

#### Custom Sensors

For custom sensors, provide wavelength information for each band:

```python
# Example: Custom 6-band sensor
custom_wavelengths = torch.tensor([[450, 550, 650, 800, 1600, 2200]], 
                                   dtype=torch.float32)
embeddings = model.encoder(chips, timestamps, custom_wavelengths)
```

### References

**Documentation**: https://clay-foundation.github.io/model/ \
**GitHub**: https://github.com/Clay-foundation/model \
**HuggingFace**: https://huggingface.co/made-with-clay/Clay \
**License**: Apache-2.0

### Citation

```bibtex
@software{clay_foundation_model_2024,
  author = {Clay Foundation},
  title = {Clay Foundation Model: An open source AI model for Earth},
  year = {2024},
  url = {https://github.com/Clay-foundation/model},
  version = {1.5}
}
```

### Supported Sensors

| Sensor | Bands | Resolution | Description |
|--------|-------|------------|-------------|
| Sentinel-2 L2A | 10 | 10m | Optical multispectral |
| Landsat C2 L1/L2 | 6 | 30m | Optical multispectral |
| NAIP | 4 | 1m | Aerial RGB + NIR |
| LINZ | 3 | 0.5m | Aerial RGB |
| Sentinel-1 | 2 | 10m | SAR (VV, VH) |
| MODIS | 7 | 500m | Global surface reflectance |

For custom sensors, you can provide a metadata configuration file. See the [Clay documentation](https://clay-foundation.github.io/model/getting-started/quickstart.html#supported-sensors) for details.
