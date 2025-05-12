# src/train.py  –  3‑D Gaussian‑Splat 학습 스크립트
# ----------------------------------------------------------
# * 숫자 id 기준으로 이미지 ↔ COLMAP metadata 매핑 (KeyError 방지)
# * tile 32 px, chunk_gauss 4096, 타일마다 backward → 그래프 즉시 해제
# * 이미지 1장마다 optimizer.step 1회 (속도/메모리 균형)
# * AMP FP16 켜서 A100 40 GB 기준 VRAM ≈ 7 GB
# ----------------------------------------------------------

import argparse, yaml, torch
from pathlib import Path
from PIL import Image
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

from gaussian.init_from_sfm import (
    read_points3D, read_images, create_gaussian_cloud_from_points
)
from render import DifferentiableRenderer


# ───────────────── Dataset ─────────────────────────────────
class NeRFDataset(Dataset):
    """
    이미지 파일명과 images.txt 간 불일치를 방지하기 위해
    '숫자 id' 만 추출해 매핑한다 (ex: r_0092.png → id 92).
    """
    def __init__(self, img_dir: Path, images_txt: Path, imsize):
        self.paths = sorted(p for p in Path(img_dir).glob("*.png")
                            if "_depth" not in p.name)

        meta_raw = read_images(str(images_txt))
        self.id2meta = {}
        for m in meta_raw.values():
            digits = "".join(filter(str.isdigit, Path(m["file"]).stem))
            if digits:
                self.id2meta[int(digits)] = m  # id → meta

        self.tf = transforms.Compose([
            transforms.Resize(tuple(imsize)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        digits = "".join(filter(str.isdigit, p.stem))
        if digits == "":
            raise RuntimeError(f"{p.name} contains no numeric id")
        meta = self.id2meta.get(int(digits))
        if meta is None:
            raise RuntimeError(f"metadata for id={digits} ({p.name}) not found")

        img = self.tf(Image.open(p).convert("RGB"))
        pose = {
            "qvec": torch.from_numpy(meta["qvec"]),
            "tvec": torch.from_numpy(meta["tvec"])
        }
        return img, pose
# ───────────────────────────────────────────────────────────


# ────────────────── main ───────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/gaussian_train.yaml",
                    help="YAML config path")
    return ap.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]

    cfg = yaml.safe_load(open(repo / args.cfg))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Gaussian cloud 초기화 ──────────────────────────────
    pts_path = (repo / cfg["data"]["root"] /
                cfg["data"]["colmap_output_dir"] / "sparse/0/points3D.txt")
    xyz, rgb = read_points3D(str(pts_path))
    cloud = create_gaussian_cloud_from_points((xyz, rgb)).to(dev)

    # ── Renderer ─────────────────────────────────────────
    renderer = DifferentiableRenderer(cfg["render"]["image_size"]).to(dev)

    # ── DataLoader ───────────────────────────────────────
    img_dir = repo / cfg["data"]["root"] / cfg["data"]["images_dir"]
    img_txt = (repo / cfg["data"]["root"] /
               cfg["data"]["colmap_output_dir"] / "sparse/0/images.txt")
    ds = NeRFDataset(img_dir, img_txt, cfg["render"]["image_size"])
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                    pin_memory=(dev.type == "cuda"))

    # ── Optimizer & AMP ──────────────────────────────────
    opt = torch.optim.Adam(cloud.parameters(), lr=cfg["train"]["learning_rate"])
    scaler = amp.GradScaler()

    tile_hw = 32
    step_pix = tile_hw * tile_hw   # 1024
    chunk_gauss = 4096

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss_epoch = 0.0

        for img, pose in dl:
            img = img.to(dev, non_blocking=True)
            pose = {k: v.to(dev, non_blocking=True) for k, v in pose.items()}

            HW = img.shape[-2] * img.shape[-1]
            tiles = torch.arange(0, HW - step_pix + 1, step_pix, device=dev)

            opt.zero_grad(set_to_none=True)

            for start in tiles:
                end = start + step_pix
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = renderer(
                        cloud, pose,
                        tile_hw=tile_hw, chunk_gauss=chunk_gauss,
                        tile_range=(start, end)
                    )[0]                                # (3,32,32)

                    loss = F.mse_loss(
                        pred.view(3, -1),               # (3,1024)
                        img.view(3, -1)[:, start:end]   # (3,1024)
                    )

                scaler.scale(loss).backward()
                loss_epoch += loss.item()

            scaler.step(opt)
            scaler.update()

        print(f"Epoch {epoch:03d} | Loss {loss_epoch / len(dl):.6f}")


if __name__ == "__main__":
    main()
