# Differentiable 2‑D splat renderer (ultra‑low‑mem)
import torch, yaml
from pathlib import Path
from torch import nn
from gaussian.init_from_sfm import read_cameras


class DifferentiableRenderer(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.H, self.W = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── intrinsics ──────────────────────────────────────────
        root = Path(__file__).resolve().parents[1]
        cfg  = yaml.safe_load(open(root / "configs" / "gaussian_train.yaml"))
        cam0 = next(iter(read_cameras(str(
            root / cfg["data"]["root"] /
            cfg["data"]["colmap_output_dir"] / "sparse/0/cameras.txt"
        )).values()))
        if cam0["model"].startswith("SIMPLE_RADIAL"):
            f, cx, cy = cam0["params"][:3]; fx = fy = f
        else:                                  # PINHOLE 외 형식 생략
            fx, fy, cx, cy = cam0["params"][:4]
        self.fx, self.fy = map(lambda v: torch.tensor(v, device=self.device), (fx, fy))
        self.cx, self.cy = map(lambda v: torch.tensor(v, device=self.device), (cx, cy))

        xs = torch.arange(self.W, device=self.device); ys = torch.arange(self.H, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.pixel_coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)  # (HW,2)

    # ----------------------------------------------------------
    @staticmethod
    def quat2mat(q):
        q = q / q.norm(); w, x, y, z = q
        return torch.stack([
            torch.tensor([1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)]),
            torch.tensor([2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)]),
            torch.tensor([2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)])
        ], 0).to(q)

    # ----------------------------------------------------------
    def forward(self, cloud, pose,
                tile_hw=8,
                chunk_gauss=256,            # CHANGED  더 작은 청크
                tile_range=None):           # NEW  (start,end) 만 렌더
        pos = cloud.positions.to(self.device, torch.float16)
        col = cloud.colors.to(self.device,   torch.float16)
        opa = cloud.opacities.to(self.device,torch.float16)
        sc  = cloud.scales.to(self.device,   torch.float16)
        qv  = pose["qvec"].to(self.device); tv = pose["tvec"].to(self.device)

        B, N, HW = qv.shape[0], pos.shape[0], self.H*self.W
        if tile_range is None:
            tiles = torch.arange(0, HW, tile_hw*tile_hw, device=self.device)
        else:  # 단일 타일 전용
            tiles = torch.tensor(tile_range, device=self.device)

        rendered = []
        for b in range(B):
            R = self.quat2mat(qv[b]); t = tv[b].unsqueeze(0)
            proj = (R @ pos.t()).t().float() + t
            proj = torch.stack([(proj[:,0]/proj[:,2])*self.fx + self.cx,
                                (proj[:,1]/proj[:,2])*self.fy + self.cy], 1)

            img_acc = torch.zeros(len(tiles)*tile_hw*tile_hw, 3,
                                  device=self.device, dtype=torch.float16)
            w_acc   = torch.zeros_like(img_acc[..., :1])

            for g0 in range(0, N, chunk_gauss):
                g1 = min(g0+chunk_gauss, N)
                proj_g, col_g, opa_g = proj[g0:g1], col[g0:g1], opa[g0:g1]
                var_g = (sc[g0:g1].squeeze(-1)**2).unsqueeze(1)

                for i, start in enumerate(tiles):
                    end = min(start + tile_hw*tile_hw, HW)
                    coords = self.pixel_coords[start:end]

                    diff  = (coords.unsqueeze(0) - proj_g.unsqueeze(1)).half()
                    dist2 = (diff**2).sum(-1).float()
                    w     = opa_g * torch.exp(-0.5*dist2 / var_g)

                    num = (w.unsqueeze(-1) * col_g.unsqueeze(1)).sum(0)
                    den = w.sum(0).unsqueeze(-1) + 1e-8
                    img_acc[i*tile_hw*tile_hw:(i+1)*tile_hw*tile_hw] += num
                    w_acc[i*tile_hw*tile_hw:(i+1)*tile_hw*tile_hw]  += den

            img = (img_acc / w_acc.clamp(min=1e-8)).t().reshape(3, len(tiles)*tile_hw, tile_hw)
            rendered.append(img[:, :self.H, :self.W])     # 안전 절삭

        return torch.stack(rendered, 0)  # (B,3,H,W)
