# src/gaussian/gaussian_point.py

import torch
from torch import nn

class GaussianPointCloud(nn.Module):
    def __init__(self, positions, colors=None, scale=1.0):
        super().__init__()
        N = positions.shape[0]

        self.positions = nn.Parameter(positions)  # (N, 3)
        self.opacities = nn.Parameter(torch.ones(N, 1) * 0.1)  # 초기 opacity
        self.colors = nn.Parameter(colors if colors is not None else torch.ones(N, 3) * 0.5)  # (N, 3) RGB

        # isotropic scale
        self.scales = nn.Parameter(torch.ones(N, 1) * scale)  # (N, 1)

        # optional: Spherical harmonics coefficients / covariance matrix later

    def forward(self):
        return {
            'positions': self.positions,
            'colors': self.colors,
            'opacities': self.opacities,
            'scales': self.scales
        }
