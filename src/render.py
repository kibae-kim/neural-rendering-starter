# Differentiable Gaussian‑splat renderer
#  (tile 32×32 px, chunk 4096 gaussians, tile‑range 지원)

import torch, yaml
from pathlib import Path
from torch import nn
from gaussian.init_from_sfm import read_cameras


class DifferentiableRenderer(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.H, self.W = image_size
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── intrinsics ────────────────────────────────────────────────
        root = Path(__file__).resolve().parents[1]
        cfg  = yaml.safe_load(open(root / "configs" / "gaussian_train.yaml"))
        cam0 = next(iter(read_cameras(str(
            root / cfg["data"]["root"] /
            cfg["data"]["colmap_output_dir"] / "sparse/0/cameras.txt")).values()))
        if cam0["model"].startswith("SIMPLE_RADIAL"):
            f, cx, cy = cam0["params"][:3]; fx = fy = f
        else:  fx, fy, cx, cy = cam0["params"][:4]

        self.fx = torch.tensor(fx, device=self.dev); self.fy = torch.tensor(fy, device=self.dev)
        self.cx = torch.tensor(cx, device=self.dev); self.cy = torch.tensor(cy, device=self.dev)

        xs = torch.arange(self.W, device=self.dev); ys = torch.arange(self.H, device=self.dev)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.pix = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)      # (HW,2)

    # --------------------------------------------------------------
    @staticmethod
    def quat2mat(q):
        q = q / q.norm(); w,x,y,z = q
        return torch.tensor([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]], dtype=q.dtype, device=q.device)

    # --------------------------------------------------------------
    def forward(self, cloud, pose,
                tile_hw: int = 32,          # 32×32px
                chunk_gauss: int = 4096,    # 4 k gauss
                tile_range: tuple | None = None):

        pos = cloud.positions.to(self.dev, torch.float16)
        col = cloud.colors.to(self.dev,   torch.float16)
        opa = cloud.opacities.to(self.dev,torch.float16)
        sc  = cloud.scales.to(self.dev,   torch.float16)
        qv  = pose["qvec"].to(self.dev);  tv = pose["tvec"].to(self.dev)

        HW = self.H * self.W
        step = tile_hw * tile_hw
        if tile_range is None:
            tiles = torch.arange(0, HW - step + 1, step, device=self.dev)
        else:
            tiles = torch.arange(tile_range[0], tile_range[1], step, device=self.dev)

        out_tiles = []
        for b in range(qv.shape[0]):
            R = self.quat2mat(qv[b]); t = tv[b].unsqueeze(0)
            proj = (R @ pos.t()).t().float() + t
            proj = torch.stack([(proj[:,0]/proj[:,2])*self.fx + self.cx,
                                (proj[:,1]/proj[:,2])*self.fy + self.cy], 1)

            for start in tiles:
                end = start + step                                    # 항상 32² 픽셀
                coords = self.pix[start:end]                          # (1024,2)

                img_acc = w_acc = 0.                                  # reset per‑tile

                for g0 in range(0, pos.size(0), chunk_gauss):
                    g1 = min(g0+chunk_gauss, pos.size(0))
                    pj = proj[g0:g1]; cl = col[g0:g1]; op = opa[g0:g1]
                    var = (sc[g0:g1].squeeze(-1)**2).unsqueeze(1)

                    diff  = (coords.unsqueeze(0) - pj.unsqueeze(1)).half()
                    dist2 = (diff**2).sum(-1).float()
                    w     = op * torch.exp(-0.5*dist2 / var)          # (g,T)

                    num = (w.unsqueeze(-1)*cl.unsqueeze(1)).sum(0)    # (T,3)
                    den = w.sum(0).unsqueeze(-1) + 1e-8
                    img_acc += num
                    w_acc   += den

                tile = (img_acc / w_acc).t().reshape(3, tile_hw, tile_hw)  # (3,32,32)
                out_tiles.append(tile)

        return torch.stack(out_tiles, 0)     # (#tiles,3,32,32)
