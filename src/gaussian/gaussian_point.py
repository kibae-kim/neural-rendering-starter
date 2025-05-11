# src/gaussian/gaussian_point.py

import torch
from torch import nn

class GaussianPointCloud(nn.Module):
    def __init__(self, positions: torch.Tensor, colors: torch.Tensor = None, scales=None):
        """
        Parameters
        ----------
        positions : (N, 3) torch.Tensor
        colors    : (N, 3) torch.Tensor or None
        scales    : float or (N,1) torch.Tensor
                    If float, all Gaussians share same σ
                    If Tensor, per-Gaussian scale
        """
        super().__init__()
        N = positions.shape[0]

        self.positions = nn.Parameter(positions)  # (N, 3)
        self.opacities = nn.Parameter(torch.ones(N, 1) * 0.5)  #Changed0.5, initial Opacity

        if colors is not None:
            assert colors.shape == (N, 3), "Colors must be (N, 3)"
            self.colors = nn.Parameter(colors)
        else:
            self.colors = nn.Parameter(torch.ones(N, 3) * 0.5)

        # ----- Flexible scale (isotropic σ per point) -----
        if isinstance(scales, torch.Tensor):
            assert scales.shape == (N, 1), "Scales tensor must be of shape (N, 1)"
            self.scales = nn.Parameter(scales.clone())
        elif isinstance(scales, (float, int)):
            self.scales = nn.Parameter(torch.ones(N, 1) * float(scales))
        elif scales is None:
            self.scales = nn.Parameter(torch.ones(N, 1))  # default σ = 1.0
        else:
            raise TypeError("scales must be float, Tensor, or None")

        # optional: Spherical harmonics, covariance matrix, etc.

    def forward(self):
        return {
            'positions': self.positions,
            'colors': self.colors,
            'opacities': self.opacities,
            'scales': self.scales
        }
