# Differentiable Gaussian‑splat renderer  (tile 32×32 px, chunk 4096 gauss)
import yaml, torch
from pathlib import Path
from torch import nn
from gaussian.init_from_sfm import read_cameras


class DifferentiableRenderer(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.H, self.W = image_size
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── camera intrinsics ─────────────────────────────────────────
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load(open(root / "configs" / "gaussian_train.yaml"))
        cam_path = (root / cfg["data"]["root"] /
                    cfg["data"]["colmap_output_dir"] / "sparse/0/cameras.txt")
        cam = next(iter(read_cameras(str(cam_path)).values()))
        if cam["model"].startswith("SIMPLE_RADIAL"):
            f, cx, cy = cam["params"][:3]; fx = fy = f
        else:  fx, fy, cx, cy = cam["params"][:4]

        self.fx = torch.tensor(fx, device=self.dev)
        self.fy = torch.tensor(fy, device=self.dev)
        self.cx = torch.tensor(cx, device=self.dev)
        self.cy = torch.tensor(cy, device=self.dev)

        xs = torch.arange(self.W, device=self.dev)
        ys = torch.arange(self.H, device=self.dev)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.pix = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)  # (HW,2)

    # ----------------------------------------------------------------
    @staticmethod
    def quat2mat(q):
        q = q / q.norm(); w, x, y, z = q
        return torch.tensor([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=q.dtype, device=q.device)

    # ----------------------------------------------------------------
    def forward(self, cloud, pose,
                tile_hw: int = 32,
                chunk_gauss: int = 4096,
                tile_range: tuple | None = None):
        """
        tile_range : (start, end) 픽셀 인덱스. 주어지면 해당 구간 한 타일만 렌더.
        반환 shape  -  tile_range 주면 (3,32,32)
                    -  안 주면    (#tiles,3,32,32)
        """
        pos = cloud.positions.to(self.dev, torch.float16)
        col = cloud.colors.to(self.dev,   torch.float16)
        opa = cloud.opacities.to(self.dev, torch.float16)
        sc  = cloud.scales.to(self.dev,   torch.float16)
        q   = pose["qvec"].to(self.dev)
        t   = pose["tvec"].to(self.dev)

        step = tile_hw * tile_hw
        HW   = self.H * self.W
        if tile_range is None:
            tiles = torch.arange(0, HW - step + 1, step, device=self.dev)
        else:
            tiles = torch.tensor([tile_range[0]], device=self.dev)

        outputs = []
        for R, tr in zip(map(self.quat2mat, q), t):
            proj = (R @ pos.t()).t().float() + tr.unsqueeze(0)
            proj = torch.stack([(proj[:, 0] / proj[:, 2]) * self.fx + self.cx,
                                (proj[:, 1] / proj[:, 2]) * self.fy + self.cy], 1)

            for start in tiles:
                coords = self.pix[start:start+step]                    # (1024,2)
                img_acc = torch.zeros(step, 3, dtype=torch.float16, device=self.dev)
                w_acc   = torch.zeros(step, 1, dtype=torch.float16, device=self.dev)

                for g0 in range(0, pos.size(0), chunk_gauss):
                    g1 = min(g0 + chunk_gauss, pos.size(0))
                    pj = proj[g0:g1]; cl = col[g0:g1]; op = opa[g0:g1]
                    var = (sc[g0:g1].squeeze(-1)**2).unsqueeze(1)

                    diff  = (coords.unsqueeze(0) - pj.unsqueeze(1)).half()
                    dist2 = (diff**2).sum(-1).float()
                    w     = op * torch.exp(-0.5 * dist2 / var)

                    num = (w.unsqueeze(-1) * cl.unsqueeze(1)).sum(0)
                    den = w.sum(0).unsqueeze(-1) + 1e-8
                    img_acc += num.half(); w_acc += den.half()

                tile = (img_acc / w_acc).t().reshape(3, tile_hw, tile_hw)
                outputs.append(tile)

        return torch.stack(outputs, 0)  # (#tiles or 1, 3, 32, 32)
