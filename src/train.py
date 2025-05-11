# src/train.py  ── 3‑D Gaussian Splatting training script
# * OOM 회피를 위해 batch_size=1  +  tile_hw=8  +  AMP‑FP16 고정
# * 파일↔메타 robust 매칭, resume 지원, depth PNG 자동 무시
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path
import yaml
from PIL import Image
import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gaussian.init_from_sfm import (
    read_points3D, read_images, create_gaussian_cloud_from_points
)
from render import DifferentiableRenderer


# ───────────────────── Dataset ──────────────────────────────────────────────
class NeRFDataset(Dataset):
    def __init__(self, images_dir, images_txt, image_size):
        # RGB만, depth 제외                                                # CHANGED
        self.image_paths = sorted(
            p for p in Path(images_dir).glob("r_*.png") if "_depth" not in p.name
        )

        # meta: 파일명 숫자 index 기반 매핑                                 # NEW
        raw_meta = read_images(images_txt)
        self.id2meta = {
            int(''.join(filter(str.isdigit, Path(m["file"]).stem))): m
            for m in raw_meta.values()
        }

        self.tfm = transforms.Compose([
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        tgt = self.tfm(img)
        num = int(''.join(filter(str.isdigit, p.stem)))
        m   = self.id2meta[num]
        pose = {
            "qvec": torch.from_numpy(m["qvec"]),
            "tvec": torch.from_numpy(m["tvec"])
        }
        return tgt, pose


# ───────────────────── CLI ─────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",    default="configs/gaussian_train.yaml")
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


# ───────────────────── Main ────────────────────────────────────────────────
def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[1]

    cfg = yaml.safe_load(open(ROOT / args.cfg, "r"))
    train_cfg, data_cfg = cfg["train"], cfg["data"]
    render_cfg, log_cfg = cfg["render"], cfg["logging"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Gaussian cloud ─────────────────────────────────────────────────
    xyz, rgb = read_points3D(
        ROOT / data_cfg["root"] / data_cfg["colmap_output_dir"] / "sparse/0/points3D.txt"
    )
    gaussians = create_gaussian_cloud_from_points(
        (xyz, rgb),
        scale=train_cfg.get("gaussian_scale", 1.0),
        scale_variance=cfg["gaussian"].get("scale_variance", False)
    ).to(device)

    # ── Renderer (tile‑wise) ───────────────────────────────────────────
    renderer = DifferentiableRenderer(render_cfg["image_size"]).to(device)

    # ── DataLoader (batch=1, worker=0) ─────────────────────────────────
    images_dir = ROOT / data_cfg["root"] / data_cfg["images_dir"]
    images_txt = ROOT / data_cfg["root"] / data_cfg["colmap_output_dir"] / "sparse/0/images.txt"
    ds = NeRFDataset(images_dir, images_txt, render_cfg["image_size"])
    dl = DataLoader(ds,
                    batch_size=1,                   # CHANGED (최소 메모리)
                    shuffle=True,
                    num_workers=0,                  # CHANGED (Crash 방지)
                    pin_memory=(device.type == "cuda"))

    # ── Preview reference ─────────────────────────────────────────────
    _, prv_pose = ds[0]
    prv_pose = {k: v.unsqueeze(0).to(device) for k, v in prv_pose.items()}

    # ── Optimizer & AMP‑FP16 ──────────────────────────────────────────
    opt = torch.optim.Adam(gaussians.parameters(), lr=train_cfg["learning_rate"])
    scaler = amp.GradScaler() if device.type == "cuda" else None

    # ── Resume ───────────────────────────────────────────────────────
    start = 1
    out_dir = (ROOT / log_cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and args.resume.lower() not in ("none", ""):
        ckpt = torch.load(args.resume, map_location=device)
        gaussians.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start = ckpt["epoch"] + 1
        print(f"[Resume] {args.resume} → epoch {start}")

    print(f"[DEBUG] dataset={len(ds)} imgs | batches/epoch={len(dl)}")

    # ── Train ────────────────────────────────────────────────────────
    for epoch in range(start, train_cfg["epochs"] + 1):
        loss_sum = 0.0
        for tgt, pose in dl:
            tgt = tgt.to(device, non_blocking=True)
            pose = {k: v.to(device, non_blocking=True) for k, v in pose.items()}

            with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
                img = renderer(gaussians, pose, tile_hw=8)     # CHANGED (tile=8)
                loss = F.mse_loss(img, tgt)

            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()

            loss_sum += loss.item()

        print(f"E{epoch:03d}/{train_cfg['epochs']} | Loss {loss_sum/len(dl):.6f}")

        # ── Save & preview ─────────────────────────────────────────
        if epoch % log_cfg["save_every"] == 0:
            torch.save({"model": gaussians.state_dict(),
                        "opt":   opt.state_dict(),
                        "epoch": epoch},
                       out_dir / f"epoch{epoch:03d}.pt")

            with torch.no_grad():
                prv = renderer(gaussians, prv_pose, tile_hw=8)[0]  # CHANGED
            from imageio import imwrite
            imwrite(out_dir / f"render{epoch:03d}.png",
                    (prv.cpu().permute(1, 2, 0).clamp(0, 1).numpy()*255).astype("uint8"))


if __name__ == "__main__":
    main()
