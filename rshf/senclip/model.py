import torch
import torch.nn as nn
import clip
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig
import ssl

#HACK: Ignore SSL errors when loading CLIP model
ssl._create_default_https_context = ssl._create_unverified_context


class SenCLIPConfig(PretrainedConfig):
    """
    Configuration class to store the configuration of a `SenCLIP` model.

    Arguments:
        architecture: str (default: 'ViT-L/14'). CLIP model architecture to load.
        device: str (default: 'cuda'). Device to load the CLIP model on.
    """
    def __init__(self, architecture='ViT-L/14', device='cuda'):
        super(SenCLIPConfig, self).__init__()
        self.architecture = architecture
        self.device = device

    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self


class SenCLIP(nn.Module, PyTorchModelHubMixin):
    """
    Main class implementing the SenCLIP model with contrastive loss and pooling layers.

    Args:
        Various configurations for the model, pooling, and contrastive loss.
    """
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        if type(config) is dict:
            self.config = PretrainedConfig().from_dict(config)
        
        self.device = "cpu" if not torch.cuda.is_available() else self.config.device
        self.clip, _ = clip.load(self.config.architecture, device=self.device)