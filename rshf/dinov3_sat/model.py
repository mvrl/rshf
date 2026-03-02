import torch.nn as nn
import torch
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, PretrainedConfig
from torchvision import transforms

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