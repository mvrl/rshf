#!/usr/bin/env python3
"""
Upload Clay model checkpoint to HuggingFace Hub.
"""

import os
from claymodel.module import ClayMAEModule
from huggingface_hub import HfApi, create_repo

# Load the model from checkpoint
print("Loading Clay model from checkpoint...")
model = ClayMAEModule.load_from_checkpoint(
    "clay-v1.5.ckpt",
    metadata_path="configs/metadata.yaml",
    strict=False
)
model.eval()

print("Model loaded successfully!")

# Repository details
repo_id = "MVRL/clay-v1.5"
print(f"\nPreparing to upload to {repo_id}...")

# Check if HF_TOKEN is set
if not os.environ.get("HF_TOKEN"):
    print("\n⚠️  HF_TOKEN environment variable not set!")
    print("Please set it in Cursor Dashboard (Cloud Agents > Secrets)")
    print("\nTo upload manually, run:")
    print(f"  huggingface-cli login")
    print(f"  huggingface-cli upload {repo_id} clay-v1.5-upload/clay-v1.5.ckpt")
    exit(1)

# Create repository if it doesn't exist
print(f"Creating repository {repo_id} (if it doesn't exist)...")
try:
    api = HfApi()
    create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"✓ Repository {repo_id} ready")
except Exception as e:
    print(f"✗ Error creating repository: {e}")
    exit(1)

# Upload the checkpoint file
print(f"\nUploading checkpoint to {repo_id}...")
try:
    api.upload_file(
        path_or_fileobj="clay-v1.5-upload/clay-v1.5.ckpt",
        path_in_repo="clay-v1.5.ckpt",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"✓ Checkpoint uploaded successfully!")
except Exception as e:
    print(f"✗ Error uploading checkpoint: {e}")
    exit(1)

# Create and upload README
readme_content = """---
license: apache-2.0
tags:
- remote-sensing
- earth-observation
- foundation-model
- geospatial
- satellite-imagery
- vision-transformer
library_name: rshf
---

# Clay Foundation Model v1.5

Clay is an open-source AI foundation model for Earth observation. It processes satellite imagery to generate semantic embeddings using a Vision Transformer architecture adapted for geospatial and temporal relations.

## Model Details

- **Model Size**: Large
- **Architecture**: Vision Transformer (ViT) with Masked Autoencoder (MAE)
- **Training**: Self-supervised learning on Earth observation data
- **Version**: 1.5
- **Parameters**: ~300M

## Supported Sensors

| Sensor | Bands | Resolution | Description |
|--------|-------|------------|-------------|
| Sentinel-2 L2A | 10 | 10m | Optical multispectral |
| Landsat C2 L1/L2 | 6 | 30m | Optical multispectral |
| NAIP | 4 | 1m | Aerial RGB + NIR |
| LINZ | 3 | 0.5m | Aerial RGB |
| Sentinel-1 | 2 | 10m | SAR (VV, VH) |
| MODIS | 7 | 500m | Global surface reflectance |

## Usage

### Installation

First, install the required dependencies:

```bash
pip install rshf
pip install git+https://github.com/Clay-foundation/model.git
```

### Loading the Model

```python
from rshf.clay import Clay
import torch

# Load pretrained model from HuggingFace Hub
model = Clay.from_pretrained("MVRL/clay-v1.5")
model.eval()

# Prepare input data (Sentinel-2 L2A example)
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

## Use Cases

- **Generate embeddings** for feature detection (mines, aquaculture, farms, etc.)
- **Fine-tune** for classification, regression, and change detection
- **Detect changes** like deforestation, wildfires, urban development
- **Transfer learning** for downstream Earth observation tasks

## Original Model

This model is from the Clay Foundation Model available at:
- **Documentation**: https://clay-foundation.github.io/model/
- **GitHub**: https://github.com/Clay-foundation/model
- **Original Weights**: https://huggingface.co/made-with-clay/Clay

## Citation

```bibtex
@software{clay_foundation_model_2024,
  author = {Clay Foundation},
  title = {Clay Foundation Model: An open source AI model for Earth},
  year = {2024},
  url = {https://github.com/Clay-foundation/model},
  version = {1.5}
}
```

## License

Apache-2.0

## Acknowledgments

Clay is a program of Renaissance Philanthropy, a 501(c)(3) organization. The model was developed with support from the Earth observation community.
"""

print("\nCreating README.md...")
try:
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"✓ README.md uploaded successfully!")
except Exception as e:
    print(f"✗ Error uploading README: {e}")

print(f"\n✅ Upload complete! Model available at: https://huggingface.co/{repo_id}")
