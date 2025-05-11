# src/render.py  ── Differentiable 2‑D splat renderer
# * 메모리 최적화를 위해 ⓐ 가우시안 → chunk 계산  ⓑ 타일‑스트리밍  ⓒ FP16  지원
# * 바꾼 줄/블록은  # CHANGED  /  # NEW  주석으로 명시
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

        # ── config & camera intrinsics ─────────────────────────────────────────
        REPO_ROOT = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "gaussian_train.yaml", "r"))

        cam_txt = (REPO_ROOT / cfg["data"]["root"] /
                   cfg["data"]["colmap_output_dir"] / "sparse/0/cameras.txt")
        cam0 = next(iter(read_cameras(str(cam_txt)).values()))

        if cam0["model"].startswith("SIMPLE_RADIAL"):
            f, cx, cy = cam0["params"][:3]; fx = fy = f
        elif cam0["model"] == "PINHOLE":
            fx, fy, cx, cy = cam0["params"][:4]
        else:
            raise NotImplementedError(cam0["model"])

        # 텐서 등록
        self.fx = torch.tensor(fx, device=self.device, dtype=torch.float32)
        self.fy = torch.tensor(fy, device=self.device, dtype=torch.float32)
        self.cx = torch.tensor(cx, device=self.device, dtype=torch.float32)
        self.cy = torch.tensor(cy, device=self.device, dtype=torch.float32)

        # 픽셀 좌표 grid 미리 생성
        xs = torch.arange(self.W, device=self.device, dtype=torch.float32)
        ys = torch.arange(self.H, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.pixel_coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)  # (HW,2)

    # ------------------------------------------------------------------
    @staticmethod
    def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        q = q / q.norm()
        w, x, y, z = q
        return torch.stack([
            torch.tensor([1 - 2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)]),
            torch.tensor([2*(x*y + z*w), 1 - 2*(x*x+z*z),   2*(y*z - x*w)]),
            torch.tensor([2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x+y*y)])
        ], 0).to(q)

    # ------------------------------------------------------------------
    def forward(self, gaussian_cloud, pose,
                tile_hw: int = 8,        # CHANGED  8‑px 타일(초저메모리)
                chunk_gauss: int = 16384 # NEW      16 k 가우시안 단위 chunk
                ):
        """
        tile_hw   : 타일 한 변 픽셀 수 (작을수록 메모리↓, 속도↓)
        chunk_gauss: 한 번에 처리할 가우시안 수
        """
        # ── 모든 텐서를 같은 디바이스 & dtype(FP16)로 ─────────────────────
        pos = gaussian_cloud.positions.to(self.device, torch.float16)      # CHANGED
        col = gaussian_cloud.colors.to(self.device,   torch.float16)       # CHANGED
        opa = gaussian_cloud.opacities.to(self.device, torch.float16)      # CHANGED
        sc  = gaussian_cloud.scales.to(self.device,    torch.float16)      # CHANGED
        qv  = pose["qvec"].to(self.device, torch.float32)
        tv  = pose["tvec"].to(self.device, torch.float32)

        B, N, HW = qv.shape[0], pos.shape[0], self.H * self.W
        tiles = torch.arange(0, HW, tile_hw*tile_hw, device=self.device)

        rendered = []
        for b in range(B):
            R = self.quaternion_to_rotation_matrix(qv[b])
            t = tv[b].unsqueeze(0)

            # ── world → camera  (FP16) ─────────────────────────────────
            p_cam = (R @ pos.t()).t().float() + t          # (N,3)   keep proj in FP32
            x = p_cam[:, 0] / p_cam[:, 2]
            y = p_cam[:, 1] / p_cam[:, 2]
            proj = torch.stack([x * self.fx + self.cx,
                                y * self.fy + self.cy], 1)  # (N,2)

            # 결과 버퍼
            img_acc = torch.zeros(HW, 3, device=self.device, dtype=torch.float32) # NEW
            w_acc   = torch.zeros(HW, 1, device=self.device, dtype=torch.float32) # NEW

            # ── 가우시안 chunk 루프 ──────────────────────────────────
            for g0 in range(0, N, chunk_gauss):                                # NEW
                g1 = min(g0+chunk_gauss, N)
                pos_chunk  = proj[g0:g1]          # (g,2)
                col_chunk  = col[g0:g1]
                opa_chunk  = opa[g0:g1]
                var_chunk  = (sc[g0:g1].squeeze(-1) ** 2).unsqueeze(1)  # (g,1)

                # ── 타일 루프 ────────────────────────────────────
                for start in tiles:
                    end   = min(start + tile_hw*tile_hw, HW)
                    coords = self.pixel_coords[start:end]          # (T,2)

                    # diff (g,T,2) in FP16
                    diff  = (coords.unsqueeze(0) - pos_chunk.unsqueeze(1)).half()
                    dist2 = (diff ** 2).sum(-1).float()            # -> FP32
                    w     = opa_chunk * torch.exp(-0.5 * dist2 / var_chunk)  # (g,T)

                    num = (w.unsqueeze(-1) * col_chunk.unsqueeze(1)).sum(0)  # (T,3)
                    den = w.sum(0).unsqueeze(-1) + 1e-8                      # (T,1)

                    img_acc[start:end] += num
                    w_acc[start:end]   += den

            img_flat = (img_acc / w_acc.clamp(min=1e-8)).t().reshape(3, self.H, self.W)
            rendered.append(img_flat)

        return torch.stack(rendered, 0)         # (B,3,H,W)
