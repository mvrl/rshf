import torch
import torch.nn as nn
import math
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class ResidualFCNet(nn.Module, PyTorchModelHubMixin):

    def __init__(self, config: PretrainedConfig):
        super(ResidualFCNet, self).__init__()
        self.config = config
        if type(config) is dict:
            self.config = PretrainedConfig().from_dict(config)
        self.inc_bias = False
        if self.config.num_classes > 0: 
            self.class_emb = nn.Linear(self.config.num_filts, self.config.num_classes, bias=self.inc_bias)
        layers = []
        layers.append(nn.Linear(self.config.num_inputs, self.config.num_filts))
        layers.append(nn.ReLU(inplace=True))
        for i in range(self.config.depth):
            layers.append(ResLayer(self.config.num_filts))
        self.feats = torch.nn.Sequential(*layers)

    def forward(self, x, class_of_interest=None, return_feats=True):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)
        return torch.sigmoid(class_pred)

def preprocess_locs(locs):
    locs[:, 0] /= 180.0
    locs[:, 1] /= 90.0

    feats = torch.cat((torch.sin(math.pi*locs), torch.cos(math.pi*locs)), dim=1)

    return feats

