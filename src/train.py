# src/train.py  ── 3‑D Gaussian Splatting training script
# (handles resume=None gracefully, tile‑wise rendering, robust file→meta match)

import argparse
from pathlib import Path
import yaml
from PIL import Image
import torch
import torch.nn.functional as F
from torch import amp                      # GradScaler / autocast (PyTorch ≥2.1)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gaussian.init_from_sfm import (
    read_points3D, read_images, read_cameras, create_gaussian_cloud_from_points
)
from render import DifferentiableRenderer


# --------------------------- Dataset ---------------------------
class NeRFDataset(Dataset):
    def __init__(self, images_dir, images_txt, cameras_txt, image_size):
        # RGB만 로드(_depth 제외)                                 # CHANGED
        self.image_paths = sorted(
            p for p in Path(images_dir).glob("r_*.png") if "_depth" not in p.name
        )

        # meta를 번호(index)→meta로 매핑해 견고하게 검색        # NEW
        images_meta_raw = read_images(images_txt)
        self.id2meta = {}
        for m in images_meta_raw.values():
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
        meta = self.id2meta[num]                                 # robust lookup
        pose = {
            'qvec': torch.from_numpy(meta['qvec']),
            'tvec': torch.from_numpy(meta['tvec'])
        }
        return target, pose
# ----------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",    default="configs/gaussian_train.yaml",
                   help="YAML config path")
    p.add_argument("--resume", default=None,
                   help="checkpoint to resume (.pt) | 'None' for fresh run")
    return p.parse_args()


def main():
    args = parse_args()
    REPO_ROOT = Path(__file__).resolve().parents[1]

    # ════════════════ Load config ════════════════
    cfg = yaml.safe_load(open(REPO_ROOT / args.cfg, 'r'))
    train_cfg, data_cfg = cfg['train'], cfg['data']
    render_cfg, logging_cfg, gaussian_cfg = cfg['render'], cfg['logging'], cfg['gaussian']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ════════════════ Gaussian cloud init ════════════════
    xyz, rgb = read_points3D(REPO_ROOT / data_cfg['root'] /
                             data_cfg['colmap_output_dir'] / "sparse/0/points3D.txt")
    gaussians = create_gaussian_cloud_from_points(
        (xyz, rgb),
        scale=train_cfg.get("gaussian_scale", 1.0),
        scale_variance=gaussian_cfg.get("scale_variance", False)
    ).to(device)

    # ════════════════ Renderer ════════════════
    renderer = DifferentiableRenderer(render_cfg['image_size']).to(device)

    # ════════════════ Dataset / DataLoader ════════════════
    images_dir = REPO_ROOT / data_cfg['root'] / data_cfg['images_dir']
    images_txt = REPO_ROOT / data_cfg['root'] / data_cfg['colmap_output_dir'] / "sparse/0/images.txt"
    cameras_txt = REPO_ROOT / data_cfg['root'] / data_cfg['colmap_output_dir'] / "sparse/0/cameras.txt"

    dataset = NeRFDataset(images_dir, images_txt, cameras_txt, render_cfg['image_size'])
    dl = DataLoader(dataset,
                    batch_size=train_cfg.get('batch_size', 4),        # CHANGED (작게)
                    shuffle=True,
                    num_workers=0,                                    # CHANGED
                    pin_memory=(device.type == "cuda"))

    # preview sample (첫 프레임 고정)
    preview_target, preview_pose = dataset[0]
    preview_pose = {k: v.unsqueeze(0).to(device) for k, v in preview_pose.items()}

    # ════════════════ Optimizer & AMP ════════════════
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=train_cfg['learning_rate'])
    scaler = amp.GradScaler() if device.type == "cuda" else None

    # ════════════════ Resume ════════════════
    start_epoch, output_dir = 1, (REPO_ROOT / logging_cfg['output_dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = args.resume
    if isinstance(resume_path, str) and resume_path.lower() in ("none", ""):
        resume_path = None

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        gaussians.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] loaded {resume_path} @ epoch {ckpt['epoch']}")
    else:
        print("[Resume] None → starting fresh training")

    # ════════════════ Debug info ════════════════
    print(f"[DEBUG] dataset={len(dataset)} imgs | batches per epoch={len(dl)}")

    # ════════════════ Training loop ════════════════
    for epoch in range(start_epoch, train_cfg['epochs'] + 1):
        epoch_loss = 0.0
        for target, pose in dl:
            target = target.to(device, non_blocking=True)
            pose = {k: v.to(device, non_blocking=True) for k, v in pose.items()}

            if scaler:
                with amp.autocast(device_type="cuda"):
                    rendered = renderer(gaussians, pose, tile_hw=32)     # CHANGED
                    loss = F.mse_loss(rendered, target)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                rendered = renderer(gaussians, pose, tile_hw=32)         # CHANGED
                loss = F.mse_loss(rendered, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        print(f"E{epoch:03d}/{train_cfg['epochs']} | Loss {epoch_loss / len(dl):.6f}")

        # ════════════════ Checkpoint & preview ════════════════
        if epoch % logging_cfg['save_every'] == 0:
            torch.save({"model": gaussians.state_dict(),
                        "opt":   optimizer.state_dict(),
                        "epoch": epoch},
                       output_dir / f"epoch{epoch:03d}.pt")

            with torch.no_grad():
                preview = renderer(gaussians, preview_pose, tile_hw=32)[0]  # CHANGED
            from imageio import imwrite
            imwrite(output_dir / f"render{epoch:03d}.png",
                    (preview.cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype('uint8'))


if __name__ == "__main__":
    main()
