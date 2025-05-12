
# render.py – Differentiable Gaussian Splatting Renderer
import torch
import torch.nn as nn

class DifferentiableRenderer(nn.Module):
    def __init__(self, means, covs, colors, image_size, tile_hw, near_plane, far_plane, cameras, device):
        super().__init__()
        self.means = nn.Parameter(means)    # [N,3]
        self.covs = nn.Parameter(covs)      # [N,3,3]
        self.colors = nn.Parameter(colors)  # [N,3]
        self.image_size = image_size
        self.tile_hw = tile_hw
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.cameras = cameras
        self.device = device

    def forward(self, image_ids):
        results = []
        for img_id in image_ids:
            cam = self.cameras[img_id]
            rays_o, rays_d = self._generate_rays(cam)
            R = rays_o.view(-1, 3)
            D = rays_d.view(-1, 3)
            weights = self._compute_gaussian_weights(R, D)  # [R_t, N]
            rgb = (weights @ self.colors) / (weights.sum(dim=-1, keepdim=True) + 1e-5)  # [R_t,3]
            rgb = rgb.transpose(0,1).view(3, self.tile_hw*self.tile_hw)
            results.append(rgb)
        return torch.stack(results, dim=0)  # [B,3,N_rays]

    def _generate_rays(self, cam):
        fx, fy, cx, cy = cam['intrinsics']
        c2w = torch.tensor(cam['extrinsics'], device=self.device, dtype=torch.float32)
        H = W = self.tile_hw
        i, j = torch.meshgrid(
            torch.arange(W, device=self.device),
            torch.arange(H, device=self.device),
            indexing='xy'
        )
        dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], dim=-1)
        # Rotate rays
        rays_d = (dirs[..., None, :] @ c2w[:3, :3].T).squeeze(-2)
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        return rays_o, rays_d

    def _compute_gaussian_weights(self, rays_o, rays_d):
        diff = self.means.unsqueeze(0) - rays_o.unsqueeze(1)  # [R,N,3]
        t = (diff * rays_d.unsqueeze(1)).sum(-1, keepdim=True)   # [R,N,1]
        closest = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t  # [R,N,3]
        pdiff = self.means.unsqueeze(0) - closest
        sigma = torch.sqrt(torch.mean(torch.diagonal(self.covs, dim1=1, dim2=2), dim=1))  # [N]
        sigma = sigma.unsqueeze(0)  # [1,N]
        dist2 = (pdiff ** 2).sum(-1)  # [R,N]
        weights = torch.exp(-0.5 * dist2 / (sigma**2 + 1e-5))
        return weights
