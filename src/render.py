# src/render.py  – Differentiable Gaussian-splat renderer (tile-wise, chunked, FP16→FP32 accumulation)
import yaml, torch
from pathlib import Path
from torch import nn
from gaussian.init_from_sfm import read_cameras

class DifferentiableRenderer(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.H, self.W = image_size
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── load intrinsics from configs/gaussian_train.yaml ────────────
        repo = Path(__file__).resolve().parents[1]
        cfg  = yaml.safe_load(open(repo / "configs" / "gaussian_train.yaml", "r"))
        camfile = (
            repo
            / cfg["data"]["root"]
            / cfg["data"]["colmap_output_dir"]
            / "sparse/0/cameras.txt"
        )
        cam0 = next(iter(read_cameras(str(camfile)).values()))
        if cam0["model"].startswith("SIMPLE_RADIAL"):
            f, cx, cy = cam0["params"][:3]; fx = fy = f
        else:  # PINHOLE
            fx, fy, cx, cy = cam0["params"][:4]

        self.fx = torch.tensor(fx, device=self.dev, dtype=torch.float32)
        self.fy = torch.tensor(fy, device=self.dev, dtype=torch.float32)
        self.cx = torch.tensor(cx, device=self.dev, dtype=torch.float32)
        self.cy = torch.tensor(cy, device=self.dev, dtype=torch.float32)

        xs = torch.arange(self.W, device=self.dev, dtype=torch.float32)
        ys = torch.arange(self.H, device=self.dev, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.pix = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (HW,2)

    @staticmethod
    def quat2mat(q):
        q = q / q.norm()
        w, x, y, z = q
        return torch.tensor([
            [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
        ], dtype=q.dtype, device=q.device)

    def forward(self, cloud, pose,
                tile_hw: int = 64,
                chunk_gauss: int = 8192,
                tile_range: tuple | None = None):
        # FP16 inputs
        pos = cloud.positions.to(self.dev, torch.float16)
        col = cloud.colors   .to(self.dev, torch.float16)
        opa = cloud.opacities.to(self.dev, torch.float16)
        sc  = cloud.scales   .to(self.dev, torch.float16)

        # single-image pose
        q = pose["qvec"][0].to(self.dev)
        t = pose["tvec"][0].to(self.dev)
        R = self.quat2mat(q)

        # project -> FP32 for stability
        p_cam = (R @ pos.t()).t().float() + t.unsqueeze(0)
        proj = torch.stack([
            (p_cam[:,0]/p_cam[:,2]) * self.fx + self.cx,
            (p_cam[:,1]/p_cam[:,2]) * self.fy + self.cy
        ], dim=1)  # (N,2)

        HW   = self.H * self.W
        step = tile_hw * tile_hw
        if tile_range is None:
            starts = torch.arange(0, HW - step + 1, step, device=self.dev)
        else:
            starts = torch.tensor([tile_range[0]], device=self.dev)

        out = []
        N = pos.shape[0]
        for s in starts:
            coords = self.pix[s : s + step]  # (step,2)

            # accumulate in FP32
            acc_num = torch.zeros(step, 3, dtype=torch.float32, device=self.dev)
            acc_den = torch.zeros(step, 1, dtype=torch.float32, device=self.dev)

            for g0 in range(0, N, chunk_gauss):
                g1 = min(g0 + chunk_gauss, N)
                pj = proj[g0:g1]
                cl = col[g0:g1]
                op = opa[g0:g1]
                var = (sc[g0:g1].squeeze(-1)**2).unsqueeze(1)

                diff  = (coords.unsqueeze(0) - pj.unsqueeze(1)).half()
                dist2 = (diff**2).sum(-1).float()
                w     = op * torch.exp(-0.5 * dist2 / var)

                num = (w.unsqueeze(-1) * cl.unsqueeze(1)).sum(0).float()
                den = w.sum(0).unsqueeze(-1).float() + 1e-8
                acc_num += num
                acc_den += den

            tile = (acc_num / acc_den.clamp(min=1e-8)) \
                        .t() \
                        .reshape(3, tile_hw, tile_hw)
            out.append(tile.half())

        return torch.stack(out, 0)  # (num_tiles, 3, tile_hw, tile_hw)
