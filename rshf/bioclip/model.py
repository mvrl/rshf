import open_clip
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig


class BioCLIPConfig(PretrainedConfig):
    """
    Configuration class to store the configuration of a `BioCLIP` model.

    Arguments:
        model_name: str (default: 'hf-hub:imageomics/bioclip'). OpenCLIP model name or HuggingFace hub path.
    """
    def __init__(self, model_name='hf-hub:imageomics/bioclip'):
        super(BioCLIPConfig, self).__init__()
        self.model_name = model_name

    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self


class BioCLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: PretrainedConfig = None, model_name='hf-hub:imageomics/bioclip'):
        super().__init__()
        if config is not None:
            if type(config) is dict:
                config = BioCLIPConfig().from_dict(config)
            model_name = config.model_name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)