# src/train.py  –  Gaussian‑Splat training loop (single‑GPU, ultra‑low‑mem)
# 변경 내역
#   • batch_size 1  (OOM 방지)                # CHANGED
#   • renderer 호출 3곳 모두 tile_hw=8        # CHANGED
#   • 가우시안 ↔ meta 매칭 robust (index)      # (기존 유지)
#   • step마다 메모리 캐시 체크 & 비우기        # NEW
# -------------------------------------------------------------------------

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
    read_points3D, read_images, read_cameras, create_gaussian_cloud_from_points
)
from render import DifferentiableRenderer


# --------------------------- Dataset ---------------------------
class NeRFDataset(Dataset):
    def __init__(self, images_dir, images_txt, cameras_txt, image_size):
        self.image_paths = sorted(
            p for p in Path(images_dir).glob("r_*.png") if "_depth" not in p.name
        )

        raw_meta = read_images(images_txt)
        self.id2meta = {}
        for m in raw_meta.values():
            num = int(''.join(filter(str.isdigit, Path(m['file']).stem)))
            self.id2meta[num] = m

        self.transform = transforms.Compose([
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        target = self.transform(img)

        num = int(''.join(filter(str.isdigit, img_path.stem)))
        meta = self.id2meta[num]
        pose = {
            'qvec': torch.from_numpy(meta['qvec']),
            'tvec': torch.from_numpy(meta['tvec'])
        }
        return target, pose
# ----------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/gaussian_train.yaml")
    p.add_argument("--resume", default=None,
                   help="'path/to/ckpt.pt' or 'None'")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    cfg = yaml.safe_load(open(root / args.cfg))
    train_cfg, data_cfg = cfg['train'], cfg['data']
    render_cfg, log_cfg = cfg['render'], cfg['logging']
    gaussian_cfg = cfg['gaussian']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Gaussian cloud ─────────────────────────────────────────
    xyz, rgb = read_points3D(root / data_cfg['root'] /
                             data_cfg['colmap_output_dir'] / "sparse/0/points3D.txt")
    gaussians = create_gaussian_cloud_from_points(
        (xyz, rgb),
        scale=train_cfg.get("gaussian_scale", 1.0),
        scale_variance=gaussian_cfg.get("scale_variance", False)
    ).to(device)

    # ── Renderer ───────────────────────────────────────────────
    renderer = DifferentiableRenderer(render_cfg['image_size']).to(device)

    # ── Dataset & Loader ───────────────────────────────────────
    images_dir = root / data_cfg['root'] / data_cfg['images_dir']
    images_txt = root / data_cfg['root'] / data_cfg['colmap_output_dir'] / "sparse/0/images.txt"
    cameras_txt = root / data_cfg['root'] / data_cfg['colmap_output_dir'] / "sparse/0/cameras.txt"

    dataset = NeRFDataset(images_dir, images_txt, cameras_txt, render_cfg['image_size'])
    dl = DataLoader(dataset,
                    batch_size=1,                    # CHANGED  ultra‑low‑mem
                    shuffle=True,
                    num_workers=0,
                    pin_memory=(device.type == "cuda"))

    # preview
    pv_img, pv_pose = dataset[0]
    pv_pose = {k: v.unsqueeze(0).to(device) for k, v in pv_pose.items()}

    optimizer = torch.optim.Adam(gaussians.parameters(), lr=train_cfg['learning_rate'])
    scaler = amp.GradScaler() if device.type == "cuda" else None

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 1
    output_dir = (root / log_cfg['output_dir']).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = None if args.resume in (None, "None", "none", "") else args.resume
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        gaussians.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] {resume_path} -> epoch {start_epoch}")

    print(f"[DEBUG] dataset={len(dataset)} imgs | batches/epoch={len(dl)}")

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(start_epoch, train_cfg['epochs'] + 1):
        epoch_loss = 0.0
        for target, pose in dl:
            target = target.to(device, non_blocking=True)
            pose = {k: v.to(device, non_blocking=True) for k, v in pose.items()}

            if scaler:
                with amp.autocast(device_type="cuda"):
                    rendered = renderer(gaussians, pose, tile_hw=8)  # CHANGED
                    loss = F.mse_loss(rendered, target)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                rendered = renderer(gaussians, pose, tile_hw=8)      # CHANGED
                loss = F.mse_loss(rendered, target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            epoch_loss += loss.item()

            # ── 캐시 모니터 & 해제  (15 GB 초과 시) ────────────   # NEW
            if torch.cuda.is_available() and torch.cuda.memory_reserved() > 15*2**30:
                torch.cuda.empty_cache()

        print(f"E{epoch:03d}/{train_cfg['epochs']} | Loss {epoch_loss/len(dl):.6f}")

        # ── Checkpoint & preview ─────────────────────────────
        if epoch % log_cfg['save_every'] == 0:
            torch.save({"model": gaussians.state_dict(),
                        "opt": optimizer.state_dict(),
                        "epoch": epoch},
                       output_dir / f"epoch{epoch:03d}.pt")

            with torch.no_grad():
                preview = renderer(gaussians, pv_pose, tile_hw=8)[0]  # CHANGED
            from imageio import imwrite
            imwrite(output_dir / f"render{epoch:03d}.png",
                    (preview.cpu().permute(1, 2, 0).clamp(0, 1).numpy()*255).astype('uint8'))


if __name__ == "__main__":
    main()
