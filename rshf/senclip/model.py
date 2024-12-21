import torch
import torch.nn as nn
import clip
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig
import ssl

#HACK: Ignore SSL errors when loading CLIP model
ssl._create_default_https_context = ssl._create_unverified_context


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