import torch
import torch.nn as nn


class Direct(nn.Module):
    """
    Just turn it into a [-1,1] scaling based on the input range.
    """

    def __init__(self, lon_min, lon_max, lat_min, lat_max):
        """
        Args:
        """
        super(Direct, self).__init__()
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max

    def forward(self, coords):
        """"""
        lon, lat = coords[:, 0], coords[:, 1]
        lon = 2 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1
        lat = 2 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1
        return torch.stack([lon, lat], dim=1).float()
