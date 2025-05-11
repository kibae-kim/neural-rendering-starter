# src/render.py  (updated for robust repo‑root paths & optional CPU fallback)

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from gaussian.init_from_sfm import read_cameras


class DifferentiableRenderer(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.H, self.W = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CHANGED

        # ───────────────────────────────────────────────────────────────
        # 레포 루트 기준으로 절대경로 구성  (CHANGED)
        # ───────────────────────────────────────────────────────────────
        REPO_ROOT = Path(__file__).resolve().parents[1]                              # CHANGED
        cfg_path  = REPO_ROOT / "configs" / "gaussian_train.yaml"                    # CHANGED
        cfg       = yaml.safe_load(open(cfg_path, 'r'))

        data_cfg  = cfg['data']
        cam_txt   = (REPO_ROOT / data_cfg['root'] / data_cfg['colmap_output_dir'] /
                     "sparse/0/cameras.txt")                                         # CHANGED

        cams = read_cameras(str(cam_txt))
        cam0 = next(iter(cams.values()))

        # intrinsics (현재 SIMPLE_RADIAL, PINHOLE 두 가지만 처리)
        if cam0['model'].startswith("SIMPLE_RADIAL"):
            f, cx, cy = cam0['params'][:3]
            fx = fy = f
        elif cam0['model'] == "PINHOLE":
            fx, fy, cx, cy = cam0['params'][:4]
        else:
            raise NotImplementedError(f"Unsupported camera model: {cam0['model']}")

        # torch tensor로 변환
        self.fx = torch.tensor(fx, dtype=torch.float32, device=self.device)          # CHANGED
        self.fy = torch.tensor(fy, dtype=torch.float32, device=self.device)          # CHANGED
        self.cx = torch.tensor(cx, dtype=torch.float32, device=self.device)          # CHANGED
        self.cy = torch.tensor(cy, dtype=torch.float32, device=self.device)          # CHANGED

        # 미리 pixel 좌표 그리드 계산
        xs = torch.arange(0, self.W, dtype=torch.float32, device=self.device)        # CHANGED
        ys = torch.arange(0, self.H, dtype=torch.float32, device=self.device)        # CHANGED
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        self.pixel_coords = torch.stack(
            [xx.reshape(-1), yy.reshape(-1)], dim=1
        )  # (H*W,2)  (device 이미 할당)                                       # CHANGED

    # --------------------------------------------------------------
    def quaternion_to_rotation_matrix(self, q):
        q = q / q.norm()
        w, x, y, z = q
        R = torch.tensor([
            [1 - 2 * (y*y + z*z),   2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z),   2 * (y*z - x*w)],
            [2 * (x*z - y*w),   2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]
        ], dtype=torch.float32, device=self.device)                                   # CHANGED
        return R

    # --------------------------------------------------------------
    def forward(self, gaussian_cloud, pose):
        # 모든 텐서를 동일 디바이스로
        positions = gaussian_cloud.positions.to(self.device)   # (N,3)               # CHANGED
        colors    = gaussian_cloud.colors.to(self.device)      # (N,3)               # CHANGED
        opacities = gaussian_cloud.opacities.to(self.device)   # (N,1)               # CHANGED
        scales    = gaussian_cloud.scales.to(self.device)      # (N,1)               # CHANGED
        qvecs     = pose['qvec'].to(self.device)               # (B,4)               # CHANGED
        tvecs     = pose['tvec'].to(self.device)               # (B,3)               # CHANGED

        B, N, HW = qvecs.shape[0], positions.shape[0], self.H * self.W
        rendered = []

        for b in range(B):
            R = self.quaternion_to_rotation_matrix(qvecs[b])
            t = tvecs[b].unsqueeze(0)
            # world → camera
            p_cam = (R @ positions.t()).t() + t  # (N,3)

            x = p_cam[:, 0] / p_cam[:, 2]
            y = p_cam[:, 1] / p_cam[:, 2]
            u = x * self.fx + self.cx
            v = y * self.fy + self.cy
            proj = torch.stack([u, v], dim=1)    # (N,2)

            diff  = self.pixel_coords.unsqueeze(0) - proj.unsqueeze(1)  # (N,HW,2)
            dist2 = (diff ** 2).sum(-1)                                 # (N,HW)
            var   = (scales.squeeze(-1) ** 2).unsqueeze(1)
            w     = opacities * torch.exp(-0.5 * dist2 / var)           # (N,HW)

            num = (w.unsqueeze(-1) * colors.unsqueeze(1)).sum(0)        # (HW,3)
            den = w.sum(0).unsqueeze(-1) + 1e-8
            img = (num / den).t().reshape(3, self.H, self.W)            # (3,H,W)

            rendered.append(img)

        return torch.stack(rendered, dim=0)  # (B,3,H,W)
