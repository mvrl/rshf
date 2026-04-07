import torch.nn as nn
import torch
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, PretrainedConfig
from torchvision import transforms


class Dinov3SatConfig(PretrainedConfig):
    """
    Configuration class to store the configuration of a `Dinov3_Sat` model.

    Arguments:
        model_name: str (default: 'facebook/dinov2-base'). HuggingFace model name or path.
        img_mean: list (default: [0.485, 0.456, 0.406]). Normalization mean per channel.
        img_std: list (default: [0.229, 0.224, 0.225]). Normalization std per channel.
    """
    def __init__(
        self,
        model_name='facebook/dinov2-base',
        img_mean=[0.485, 0.456, 0.406],
        img_std=[0.229, 0.224, 0.225],
    ):
        super(Dinov3SatConfig, self).__init__()
        self.model_name = model_name
        self.img_mean = img_mean
        self.img_std = img_std

    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self

class Dinov3_Sat(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        if type(config) is dict:
            self.config = PretrainedConfig().from_dict(config)
        self.model = AutoModel.from_pretrained(self.config.model_name)
    
    def transform(self, x, size=224):
        preprocess = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.Normalize(mean=self.config.img_mean, std=self.config.img_std),
        ])
        return preprocess(x)
        
    
    def forward(self, x):
        return self.model(x)