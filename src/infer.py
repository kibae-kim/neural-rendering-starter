# src/infer.py
# ------------------------------------------------------------------
# Render a trained Gaussian cloud along a user‑provided camera path
# (camera_path.json: list of {"qvec":[4], "tvec":[3]} objects)
# ------------------------------------------------------------------
import argparse, json
from pathlib import Path
import imageio.v2 as imageio
import torch
from gaussian.gaussian_point import GaussianPointCloud
from gaussian.init_from_sfm import create_gaussian_cloud_from_points  # only for typing
from render import DifferentiableRenderer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint .pt file")
    p.add_argument("--camera_path", required=True, help="JSON list of poses")
    p.add_argument("--out", default="turntable.mp4", help="output mp4 or folder")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--image_size", nargs=2, type=int, default=[800, 800])
    return p.parse_args()


def load_checkpoint(ckpt_path: str) -> GaussianPointCloud:
    """Loads GaussianPointCloud only (no optimizer)"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # positions, colors, scales sizes are inferred from saved state
    dummy_xyz = torch.zeros_like(ckpt["model"]["positions"])
    dummy_rgb = torch.zeros_like(ckpt["model"]["colors"])
    cloud = GaussianPointCloud(dummy_xyz, dummy_rgb, scales=dummy_xyz[:, :1])
    cloud.load_state_dict(ckpt["model"])
    return cloud


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cloud = load_checkpoint(args.ckpt).to(device).eval()
    renderer = DifferentiableRenderer(image_size=args.image_size).to(device).eval()

    # Load camera path poses
    poses = json.load(open(args.camera_path, "r"))
    frames = []
    for p in poses:
        pose = {
            "qvec": torch.tensor(p["qvec"], dtype=torch.float32, device=device).unsqueeze(0),
            "tvec": torch.tensor(p["tvec"], dtype=torch.float32, device=device).unsqueeze(0)
        }
        with torch.no_grad():
            img = renderer(cloud, pose)[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        frames.append((img * 255).astype("uint8"))

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".mp4":
        imageio.mimwrite(out_path, frames, fps=args.fps, codec="libx264")
        print(f"[infer] wrote video {out_path}")
    else:  # folder with numbered PNGs
        out_path.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            imageio.imwrite(out_path / f"frame_{i:04d}.png", f)
        print(f"[infer] wrote {len(frames)} PNGs to {out_path}")


if __name__ == "__main__":
    main()
