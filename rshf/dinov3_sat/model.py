import torch.nn as nn
import torch
from huggingface_hub import PyTorchModelHubMixin
from torchvision import transforms

class Dinov3_Sat(nn.Module, PyTorchModelHubMixin):
    SAT_DEFAULT_MEAN = (0.430, 0.411, 0.296)
    SAT_DEFAULT_STD = (0.213, 0.156, 0.143)
    
    def __init__(self, model_name="https://huggingface.co/MVRL/dinov3_vitl16_sat/resolve/main/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16', weights=model_name)
    
    def transform(self, x, size=224):
        preprocess = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.Normalize(mean=self.SAT_DEFAULT_MEAN, std=self.SAT_DEFAULT_STD),
        ])
        return preprocess(x)
        
    
    def forward(self, x):
        return self.model(x)