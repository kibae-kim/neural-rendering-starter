import argparse, yaml, torch
from pathlib import Path
from PIL import Image
from torch import amp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from gaussian.init_from_sfm import (
    read_points3D, read_images, read_cameras, create_gaussian_cloud_from_points
)
from render import DifferentiableRenderer


class NeRFDataset(Dataset):
    def __init__(self, data_root, images_txt, cameras_txt, image_size):
        self.root = Path(data_root)
        self.files = sorted(
            p for p in (self.root / "images").glob("*.png")
            if "_depth" not in p.name
        )
        self.images_meta = read_images(images_txt)
        self.cameras     = read_cameras(cameras_txt)
        self.tf = transforms.Compose([
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = self.tf(Image.open(p).convert("RGB"))

        meta = next(
            (m for m in self.images_meta.values() if m["file"] == p.name),
            None
        )
        if meta is None:
            return None

        qvec = torch.from_numpy(meta["qvec"]).float()
        tvec = torch.from_numpy(meta["tvec"]).float()
        return img, {"qvec": qvec.unsqueeze(0), "tvec": tvec.unsqueeze(0)}


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, poses = zip(*batch)
    imgs  = torch.stack(imgs, 0)
    qvecs = torch.cat([p["qvec"] for p in poses], 0)
    tvecs = torch.cat([p["tvec"] for p in poses], 0)
    return imgs, {"qvec": qvecs, "tvec": tvecs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cfg", "--config",
        dest="cfg",
        type=str,
        default="configs/gaussian_train.yaml",
        help="Path to config YAML"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg, 'r'))
    train_cfg    = cfg["train"]
    data_cfg     = cfg["data"]
    gaussian_cfg = cfg["gaussian"]
    render_cfg   = cfg["render"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gaussian cloud
    pts = (
        Path(data_cfg["root"])
        / data_cfg["colmap_output_dir"]
        / "sparse/0/points3D.txt"
    )
    xyz, rgb = read_points3D(pts)
    cloud = create_gaussian_cloud_from_points(
        (xyz, rgb),
        scale=train_cfg.get("gaussian_scale", 1.0),
        scale_variance=gaussian_cfg.get("scale_variance", False)
    ).to(device)

    # Renderer
    renderer = DifferentiableRenderer(render_cfg["image_size"]).to(device)

    # Dataset + DataLoader
    ds = NeRFDataset(
        data_cfg["root"],
        Path(data_cfg["root"]) / data_cfg["colmap_output_dir"] / "sparse/0/images.txt",
        Path(data_cfg["root"]) / data_cfg["colmap_output_dir"] / "sparse/0/cameras.txt",
        render_cfg["image_size"]
    )
    dl = DataLoader(ds, batch_size=1, shuffle=True,
                    num_workers=0, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(cloud.parameters(),
                                 lr=train_cfg["learning_rate"])
    scaler    = amp.GradScaler(device_type="cuda")  # <--- NEW API

    tile_sz = 64 * 64
    for epoch in range(1, train_cfg["epochs"] + 1):
        total_loss = 0.0
        steps      = 0

        for batch in dl:
            if batch is None:
                continue
            imgs, pose = batch
            imgs = imgs.to(device); pose = {k: v.to(device) for k, v in pose.items()}

            H, W = render_cfg["image_size"]
            HW   = H * W
            starts = torch.arange(0, HW - tile_sz + 1, tile_sz, device=device)

            optimizer.zero_grad(set_to_none=True)
            for s in starts:
                e = s + tile_sz
                # <--- use new torch.amp.autocast
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = renderer(
                        cloud, pose,
                        tile_hw=64, chunk_gauss=8192,
                        tile_range=(s.item(), e.item())
                    )  # shape (1,3,64,64)
                    pred_flat = out.view(1, 3, -1)           # (1,3,P)
                    P = pred_flat.size(-1)
                    gt_flat   = imgs.view(1, 3, -1)[..., s:s+P]
                    loss = F.mse_loss(pred_flat, gt_flat)

                scaler.scale(loss).backward()
                total_loss += loss.item()
                steps      += 1

            scaler.step(optimizer)
            scaler.update()

        avg = total_loss / max(1, steps)
        print(f"Epoch {epoch:03d}/{train_cfg['epochs']}  Loss: {avg:.6f}")


if __name__ == "__main__":
    main()
