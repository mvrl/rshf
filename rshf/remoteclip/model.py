import open_clip
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig


class RemoteCLIPConfig(PretrainedConfig):
    """
    Configuration class to store the configuration of a `RemoteCLIP` model.

    Arguments:
        model_name: str (default: 'ViT-B-32'). OpenCLIP model architecture name.
    """
    def __init__(self, model_name='ViT-B-32'):
        super(RemoteCLIPConfig, self).__init__()
        self.model_name = model_name

    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self


class RemoteCLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: PretrainedConfig = None, model_name='ViT-B-32'):
        super().__init__()
        if config is not None:
            if type(config) is dict:
                config = RemoteCLIPConfig().from_dict(config)
            model_name = config.model_name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)