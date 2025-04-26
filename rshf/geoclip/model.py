import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
import numpy as np
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336

@torch.jit.script
def gaussian_encoding(
        v: Tensor,
        b: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`
    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`
    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`
    See :class:`~rff.layers.GaussianEncoding` for more details.
    """
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


def sample_b(sigma: float, size: tuple) -> Tensor:
    r"""Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, \sigma^2)`
    Args:
        sigma (float): standard deviation
        size (tuple): size of the matrix sampled
    See :class:`~rff.layers.GaussianEncoding` for more details
    """
    return torch.randn(size) * sigma

class GaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features"""

    def __init__(self, sigma: Optional[float] = None,
                 input_size: Optional[float] = None,
                 encoded_size: Optional[float] = None,
                 b: Optional[Tensor] = None):
        r"""
        Args:
            sigma (Optional[float]): standard deviation
            input_size (Optional[float]): the number of input dimensions
            encoded_size (Optional[float]): the number of dimensions the `b` matrix maps to
            b (Optional[Tensor], optional): Optionally specify a :attr:`b` matrix already sampled
        Raises:
            ValueError:
                If :attr:`b` is provided and one of :attr:`sigma`, :attr:`input_size`,
                or :attr:`encoded_size` is provided. If :attr:`b` is not provided and one of
                :attr:`sigma`, :attr:`input_size`, or :attr:`encoded_size` is not provided.
        """
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.')

            b = sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`
        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        Returns:
            Tensor: Tensor mapping using random fourier features of shape :math:`(N, *, 2 \cdot \text{encoded_size})`
        """
        return gaussian_encoding(v, self.b)

def equal_earth_projection(L):
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180

class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma, input_size=2, encoded_size=256, dim=512):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=input_size, encoded_size=encoded_size)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(dim, 2*dim),
                                     nn.ReLU(),
                                     nn.Linear(2*dim, 2*dim),
                                     nn.ReLU(),
                                     nn.Linear(2*dim, 2*dim),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2*dim, dim))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

class GeoCLIPConfig(PretrainedConfig):
    def __init__(self, sigma=None, input_size=2, encoded_size=256, dim=512):
        super().__init__()
        self.sigma = sigma
        self.input_size = input_size
        self.encoded_size = encoded_size
        self.dim = dim
    
    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self


class LocationEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: PretrainedConfig):
        """
        Example Usage:
        >>> from rshf.geoclip import GeoCLIP
        >>> model = GeoCLIP.from_pretrained("rshf/geoclip")
        >>> locs = torch.FloatTensor([[-80.0, 40.0]]) # Lon/Lat
        >>> embeddings = model(locs)

        Define GeoCLIP model from scratch
        >>> from rshf.geoclip import GeoCLIP, GeoCLIPConfig
        >>> config = GeoCLIPConfig(sigma=[2, 2**2], input_size=2, encoded_size=256, dim=512)
        >>> model = GeoCLIP(config)
        """
        super(LocationEncoder, self).__init__()
        self.config = config
        if type(config) is dict:
            self.config = PretrainedConfig().from_dict(config)
        self.n = len(self.config.sigma)

        for i, s in enumerate(self.config.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s, input_size=self.config.input_size, encoded_size=self.config.encoded_size, dim=self.config.dim))

    def forward(self, location):
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], self.config.dim).to(location.device)

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)
        
        return location_features