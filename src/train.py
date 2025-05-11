# src/train.py  ── 3‑D Gaussian Splatting training script
# (handles resume=None gracefully, uses new torch.amp API)

import argparse
from pathlib import Path

import yaml
from PIL import Image
import torch
import torch.nn.functional as F
from torch import amp                       # ⬅️  GradScaler / autocast (PyTorch ≥2.1)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gaussian.init_from_sfm import (
    read_points3D, read_images, read_cameras, create_gaussian_cloud_from_points
)
from render import DifferentiableRenderer


# --------------------------- Dataset ---------------------------
class NeRFDataset(Dataset):
    def __init__(self, images_dir, images_txt, cameras_txt, image_size):
        self.image_paths = sorted(Path(images_dir).glob("r_*.png"))
        self.images_meta = read_images(images_txt)
        self.cameras     = read_cameras(cameras_txt)
        self.transform   = transforms.Compose([
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img      = Image.open(img_path).convert("RGB")
        target   = self.transform(img)

        meta = next(v for v in self.images_meta.values()
                    if v['file'] == img_path.name)
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
    cfg_path = (REPO_ROOT / args.cfg).resolve()
    cfg      = yaml.safe_load(open(cfg_path, 'r'))

    train_cfg    = cfg['train']
    data_cfg     = cfg['data']
    render_cfg   = cfg['render']
    logging_cfg  = cfg['logging']
    gaussian_cfg = cfg['gaussian']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ════════════════ Gaussian cloud init ════════════════
    points3d_file = (REPO_ROOT / data_cfg['root'] /
                     data_cfg['colmap_output_dir'] / "sparse/0/points3D.txt")
    xyz, rgb = read_points3D(points3d_file)
    gaussians = create_gaussian_cloud_from_points(
        (xyz, rgb),
        scale=train_cfg.get("gaussian_scale", 1.0),
        scale_variance=gaussian_cfg.get("scale_variance", False)
    ).to(device)

    # ════════════════ Renderer ════════════════
    cameras_txt = (REPO_ROOT / data_cfg['root'] /
                   data_cfg['colmap_output_dir'] / "sparse/0/cameras.txt")
    images_txt  = (REPO_ROOT / data_cfg['root'] /
                   data_cfg['colmap_output_dir'] / "sparse/0/images.txt")
    renderer = DifferentiableRenderer(render_cfg['image_size']).to(device)

    # ════════════════ Dataset / DataLoader ════════════════
    images_dir = REPO_ROOT / data_cfg['root'] / data_cfg['images_dir']
    dataset = NeRFDataset(str(images_dir), str(images_txt), str(cameras_txt),
                          render_cfg['image_size'])
    dl = DataLoader(dataset,
                    batch_size=train_cfg['batch_size'],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=(device.type == "cuda"))

    # preview sample (첫 프레임 고정)
    preview_target, preview_pose = dataset[0]
    preview_target = preview_target.unsqueeze(0).to(device)
    preview_pose = {
        'qvec': preview_pose['qvec'].unsqueeze(0).to(device),
        'tvec': preview_pose['tvec'].unsqueeze(0).to(device)
    }

    # ════════════════ Optimizer & AMP ════════════════
    optimizer = torch.optim.Adam(gaussians.parameters(),
                                 lr=train_cfg['learning_rate'])
    scaler = amp.GradScaler(device_type="cuda") if device.type == "cuda" else None

    # ════════════════ Resume ════════════════
    start_epoch = 1
    output_dir  = (REPO_ROOT / logging_cfg['output_dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = args.resume
    if isinstance(resume_path, str) and resume_path.lower() in ("none", ""):
        resume_path = None

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        gaussians.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] loaded {resume_path} @ epoch {ckpt['epoch']}")
    else:
        print("[Resume] None → starting fresh training")

    # ════════════════ Training loop ════════════════
    for epoch in range(start_epoch, train_cfg['epochs'] + 1):
        epoch_loss = 0.0
        for target, pose in dl:
            target = target.to(device, non_blocking=True)
            pose['qvec'] = pose['qvec'].to(device, non_blocking=True)
            pose['tvec'] = pose['tvec'].to(device, non_blocking=True)

            if scaler:   # CUDA + mixed precision
                with amp.autocast(device_type="cuda", enabled=True):
                    rendered = renderer(gaussians, pose)
                    loss = F.mse_loss(rendered, target)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:        # CPU or no‑AMP
                rendered = renderer(gaussians, pose)
                loss = F.mse_loss(rendered, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dl)
        print(f"E{epoch:03d}/{train_cfg['epochs']} | Loss {avg_loss:.6f}")

        # ════════════════ Checkpoint & preview ════════════════
        if epoch % logging_cfg['save_every'] == 0:
            ckpt_path = output_dir / f"epoch{epoch:03d}.pt"
            torch.save(
                {"model": gaussians.state_dict(),
                 "opt":   optimizer.state_dict(),
                 "epoch": epoch},
                ckpt_path)

            with torch.no_grad():
                preview = renderer(gaussians, preview_pose)[0]
            img = preview.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            from imageio import imwrite
            imwrite(output_dir / f"render{epoch:03d}.png",
                    (img * 255).astype('uint8'))


if __name__ == "__main__":
    main()
