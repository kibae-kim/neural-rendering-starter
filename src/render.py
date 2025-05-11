# src/render.py  ── Differentiable 2‑D splat renderer
# 메모리 폭주(OOM) 해결용 패치
#   • chunk_gauss 16 k → **4 k** (# CHANGED)
#   • 타일 루프마다 torch.cuda.empty_cache() 호출 (# NEW)
#   • 나머지 구조/주석은 유지
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from gaussian.init_from_sfm import read_cameras


class DifferentiableRenderer(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.H, self.W = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── load camera intrinsics ────────────────────────────────────────────
        root = Path(__file__).resolve().parents[1]
        cfg  = yaml.safe_load(open(root / "configs" / "gaussian_train.yaml"))
        cam0 = next(iter(read_cameras(str(
            root / cfg["data"]["root"] /
            cfg["data"]["colmap_output_dir"] / "sparse/0/cameras.txt"
        )).values()))

        if cam0["model"].startswith("SIMPLE_RADIAL"):
            f, cx, cy = cam0["params"][:3]; fx = fy = f
        elif cam0["model"] == "PINHOLE":
            fx, fy, cx, cy = cam0["params"][:4]
        else:
            raise NotImplementedError(cam0["model"])

        self.fx = torch.tensor(fx, device=self.device)
        self.fy = torch.tensor(fy, device=self.device)
        self.cx = torch.tensor(cx, device=self.device)
        self.cy = torch.tensor(cy, device=self.device)

        xs = torch.arange(self.W, device=self.device)
        ys = torch.arange(self.H, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.pixel_coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)  # (HW,2)

    # ------------------------------------------------------------------
    @staticmethod
    def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        q = q / q.norm(); w, x, y, z = q
        return torch.stack([
            torch.tensor([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)]),
            torch.tensor([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)]),
            torch.tensor([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)])
        ], 0).to(q)

    # ------------------------------------------------------------------
    def forward(self, gaussian_cloud, pose,
                tile_hw: int = 8,
                chunk_gauss: int = 4096   # CHANGED  4 k 씩 처리
                ):

        pos = gaussian_cloud.positions.to(self.device, torch.float16)
        col = gaussian_cloud.colors.to(self.device,   torch.float16)
        opa = gaussian_cloud.opacities.to(self.device, torch.float16)
        sc  = gaussian_cloud.scales.to(self.device,    torch.float16)
        qv  = pose["qvec"].to(self.device, torch.float32)
        tv  = pose["tvec"].to(self.device, torch.float32)

        B, N, HW = qv.shape[0], pos.shape[0], self.H * self.W
        tiles = torch.arange(0, HW, tile_hw*tile_hw, device=self.device)

        rendered = []
        for b in range(B):
            R = self.quaternion_to_rotation_matrix(qv[b])
            t = tv[b].unsqueeze(0)

            p_cam = (R @ pos.t()).t().float() + t
            proj = torch.stack([(p_cam[:, 0] / p_cam[:, 2]) * self.fx + self.cx,
                                (p_cam[:, 1] / p_cam[:, 2]) * self.fy + self.cy], 1)

            img_acc = torch.zeros(HW, 3, device=self.device, dtype=torch.float32)
            w_acc   = torch.zeros(HW, 1, device=self.device, dtype=torch.float32)

            # ── gauss‑chunk & tile loops ───────────────────────────────────
            for g0 in range(0, N, chunk_gauss):                          # CHANGED
                g1 = min(g0 + chunk_gauss, N)
                proj_g  = proj[g0:g1]
                col_g   = col[g0:g1]
                opa_g   = opa[g0:g1]
                var_g   = (sc[g0:g1].squeeze(-1) ** 2).unsqueeze(1)

                for start in tiles:
                    end = min(start + tile_hw*tile_hw, HW)
                    coords = self.pixel_coords[start:end]

                    diff  = (coords.unsqueeze(0) - proj_g.unsqueeze(1)).half()
                    dist2 = (diff**2).sum(-1).float()
                    w     = opa_g * torch.exp(-0.5 * dist2 / var_g)

                    num = (w.unsqueeze(-1) * col_g.unsqueeze(1)).sum(0)
                    den = w.sum(0).unsqueeze(-1) + 1e-8
                    img_acc[start:end] += num
                    w_acc[start:end]   += den

                    torch.cuda.empty_cache()            # NEW  중간 캐시 즉시 해제

            img = (img_acc / w_acc.clamp(min=1e-8)).t().reshape(3, self.H, self.W)
            rendered.append(img)

        return torch.stack(rendered, 0)  # (B,3,H,W)
