import open_clip
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class BioCLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_name="hf-hub:imageomics/bioclip"):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)