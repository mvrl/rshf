"""
Clay Foundation Model

References:
- Documentation: https://clay-foundation.github.io/model/
- GitHub: https://github.com/Clay-foundation/model
- HuggingFace: https://huggingface.co/made-with-clay/Clay
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from huggingface_hub import PyTorchModelHubMixin
import warnings

try:
    from claymodel.module import ClayMAEModule
    CLAY_AVAILABLE = True
except ImportError:
    CLAY_AVAILABLE = False
    warnings.warn(
        "Clay model is not installed. Please install it with: "
        "pip install git+https://github.com/Clay-foundation/model.git"
    )


class ClayConfig(PretrainedConfig):
    """
    Configuration class to store the configuration of a `Clay` model.
    
    Clay is a foundation model for Earth observation that uses a Vision Transformer
    architecture adapted to understand geospatial and temporal relations in satellite data.
    
    Arguments:
        model_size (str): Size of the model. Options: 'tiny', 'small', 'base', 'large'. 
            Default: 'base'.
        mask_ratio (float): Ratio of patches to mask during MAE training. Default: 0.75.
        patch_size (int): Size of image patches. Default: 8.
        shuffle (bool): Whether to shuffle patches. Default: False.
        metadata_path (str): Path to metadata YAML file containing sensor specifications.
            Default: None (will use default Clay metadata).
    """
    
    def __init__(
        self,
        model_size="base",
        mask_ratio=0.75,
        patch_size=8,
        shuffle=False,
        metadata_path=None,
        **kwargs
    ):
        super(ClayConfig, self).__init__()
        self.model_size = model_size
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.metadata_path = metadata_path
        
    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self


class Clay(nn.Module, PyTorchModelHubMixin):
    """
    Clay Foundation Model wrapper for easy integration with Hugging Face Hub.
    
    Clay is an open-source foundation model for Earth observation that processes
    satellite imagery to generate semantic embeddings. It uses a Vision Transformer
    architecture trained via self-supervised learning using a Masked Autoencoder (MAE).
    
    The model takes satellite imagery along with wavelength, location, and time information
    as input, and outputs embeddings which are mathematical representations of areas on
    Earth's surface at specific times.
    
    Example:
        >>> from rshf.clay import Clay
        >>> import torch
        >>> 
        >>> # Load pretrained model from checkpoint
        >>> model = Clay.from_checkpoint("clay-v1.5.ckpt")
        >>> model.eval()
        >>> 
        >>> # Prepare input data
        >>> chips = torch.randn(1, 10, 256, 256)  # [batch, bands, height, width]
        >>> wavelengths = torch.tensor([[485, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]], 
        >>>                            dtype=torch.float32)  # in nanometers
        >>> timestamps = torch.zeros(1, 4)  # [week, hour, lat, lon]
        >>> 
        >>> # Generate embeddings
        >>> with torch.no_grad():
        >>>     embeddings = model.encoder(chips, timestamps, wavelengths)
        >>> print(f"Embeddings shape: {embeddings.shape}")
    
    Args:
        config (ClayConfig or dict): Model configuration.
    """
    
    def __init__(self, config):
        super().__init__()
        
        if not CLAY_AVAILABLE:
            raise ImportError(
                "Clay model dependencies are not installed. Please install with: "
                "pip install git+https://github.com/Clay-foundation/model.git"
            )
        
        self.config = config
        if type(config) is dict:
            config = ClayConfig.from_dict(ClayConfig(), config)
            self.config = config
        
        # Initialize the Clay MAE module
        # Note: The actual model will be loaded from checkpoint
        self._model = None
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path, **kwargs):
        """
        Load a Clay model from a PyTorch Lightning checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the .ckpt file (can be local path or URL).
            **kwargs: Additional arguments passed to ClayMAEModule.load_from_checkpoint.
        
        Returns:
            Clay: Loaded Clay model instance.
        
        Example:
            >>> model = Clay.from_checkpoint("clay-v1.5.ckpt")
            >>> model.eval()
        """
        if not CLAY_AVAILABLE:
            raise ImportError(
                "Clay model dependencies are not installed. Please install with: "
                "pip install git+https://github.com/Clay-foundation/model.git"
            )
        
        # Load the PyTorch Lightning checkpoint
        clay_model = ClayMAEModule.load_from_checkpoint(checkpoint_path, **kwargs)
        
        # Create a wrapper instance
        config = ClayConfig(
            model_size=kwargs.get('model_size', 'base'),
            mask_ratio=kwargs.get('mask_ratio', 0.75),
            patch_size=kwargs.get('patch_size', 8),
            shuffle=kwargs.get('shuffle', False),
            metadata_path=kwargs.get('metadata_path', None)
        )
        
        instance = cls(config)
        instance._model = clay_model
        
        return instance
    
    @property
    def encoder(self):
        """Access the encoder part of the Clay model."""
        if self._model is None:
            raise ValueError(
                "Model not loaded. Use Clay.from_checkpoint() to load a pretrained model."
            )
        return self._model.encoder
    
    @property
    def decoder(self):
        """Access the decoder part of the Clay model."""
        if self._model is None:
            raise ValueError(
                "Model not loaded. Use Clay.from_checkpoint() to load a pretrained model."
            )
        return self._model.decoder
    
    def forward(self, datacube, timestamps, wavelengths, latlon=None, gsd=None):
        """
        Forward pass through the Clay model.
        
        Args:
            datacube (torch.Tensor): Input satellite imagery of shape 
                [batch, bands, height, width].
            timestamps (torch.Tensor): Temporal metadata of shape [batch, 4] 
                containing [week, hour, lat, lon].
            wavelengths (torch.Tensor): Wavelength information for each band in nanometers,
                shape [batch, bands].
            latlon (torch.Tensor, optional): Latitude/longitude coordinates.
            gsd (torch.Tensor, optional): Ground sampling distance.
        
        Returns:
            torch.Tensor: Embeddings from the encoder.
        """
        if self._model is None:
            raise ValueError(
                "Model not loaded. Use Clay.from_checkpoint() to load a pretrained model."
            )
        
        # Use the encoder to generate embeddings
        return self._model.encoder(datacube, timestamps, wavelengths, latlon, gsd)
    
    def reconstruct(self, datacube, timestamps, wavelengths, latlon=None, gsd=None):
        """
        Reconstruct the input image from masked patches.
        
        Args:
            datacube (torch.Tensor): Input satellite imagery.
            timestamps (torch.Tensor): Temporal metadata.
            wavelengths (torch.Tensor): Wavelength information.
            latlon (torch.Tensor, optional): Latitude/longitude coordinates.
            gsd (torch.Tensor, optional): Ground sampling distance.
        
        Returns:
            dict: Dictionary containing reconstructed images and other outputs.
        """
        if self._model is None:
            raise ValueError(
                "Model not loaded. Use Clay.from_checkpoint() to load a pretrained model."
            )
        
        return self._model(datacube, timestamps, wavelengths, latlon, gsd)
    
    def eval(self):
        """Set the model to evaluation mode."""
        if self._model is not None:
            self._model.eval()
        return super().eval()
    
    def train(self, mode=True):
        """Set the model to training mode."""
        if self._model is not None:
            self._model.train(mode)
        return super().train(mode)
