from torch import nn
import math
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from .spherical_harmonics import SphericalHarmonics
from huggingface_hub import PyTorchModelHubMixin

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

"""Sinusoidal Representation Network (SIREN)"""
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, degreeinput = False, dropout = True):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.degreeinput = degreeinput

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation, dropout = False)

    def forward(self, x, mods = None):

        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)
    
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None, dropout = False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        out = self.activation(out)
        return out

def get_positional_encoding(legendre_polys=40, harmonics_calculation='analytic'):
    return SphericalHarmonics(legendre_polys=legendre_polys,
                                harmonics_calculation=harmonics_calculation)


def get_neural_network(input_dim, num_classes=256, dim_hidden=512, num_layers=2):
    return SirenNet(
            dim_in=input_dim,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            dim_out=num_classes
        )

class LocationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.posenc = get_positional_encoding()
        self.nnet = get_neural_network(self.posenc.embedding_dim)

    def forward(self, x):
        x = self.posenc(x)
        return self.nnet(x)

class SatClip(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.location = LocationEncoder()
        self.location = self.location.double()
    
    def forward(self, x):
        return self.location(x)