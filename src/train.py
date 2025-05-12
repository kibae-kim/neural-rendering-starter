# src/train.py  – Gaussian-Splat 학습 (최대 2 에포크만 수행)
import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch import amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from gaussian.init_from_sfm import (
    read_points3D,
    read_images,
    read_cameras,
    create_gaussian_cloud_from_points
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
            return None  # 메타데이터 없으면 건너뜀
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
        help="YAML 콘피그 경로"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r"))
    train_cfg    = cfg["train"]
    data_cfg     = cfg["data"]
    gaussian_cfg = cfg["gaussian"]
    render_cfg   = cfg["render"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gaussian cloud 준비
    pts_path = (
        Path(data_cfg["root"])
        / data_cfg["colmap_output_dir"]
        / "sparse/0/points3D.txt"
    )
    xyz, rgb = read_points3D(pts_path)
    cloud = create_gaussian_cloud_from_points(
        (xyz, rgb),
        scale=train_cfg.get("gaussian_scale", 1.0),
        scale_variance=gaussian_cfg.get("scale_variance", False)
    ).to(device)

    # 렌더러 인스턴스화
    renderer = DifferentiableRenderer(render_cfg["image_size"]).to(device)
    # renderer = torch.compile(renderer)  # PyTorch ≥2.0일 때 선택

    # 데이터셋 및 로더
    ds = NeRFDataset(
        data_cfg["root"],
        Path(data_cfg["root"]) / data_cfg["colmap_output_dir"] / "sparse/0/images.txt",
        Path(data_cfg["root"]) / data_cfg["colmap_output_dir"] / "sparse/0/cameras.txt",
        render_cfg["image_size"]
    )
    dl = DataLoader(
        ds,
        batch_size=2,         # 배치 크기
        shuffle=True,
        num_workers=4,        # 워커 수
        pin_memory=True,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.Adam(cloud.parameters(),
                                 lr=train_cfg["learning_rate"])
    scaler    = amp.GradScaler()  # PyTorch ≥2.1: 인자 없이 사용

    # 학습 파라미터
    tile_hw     = 128
    tile_size   = tile_hw * tile_hw
    chunk_gauss = 16384

    # 최대 2 에포크까지만 수행하도록 제한
    max_epochs = 2

    for epoch in range(1, max_epochs + 1):
        total_loss = 0.0
        steps      = 0

        for batch in dl:
            if batch is None:
                continue
            imgs, pose = batch
            imgs = imgs.to(device)
            pose = {k: v.to(device) for k, v in pose.items()}

            H, W = render_cfg["image_size"]
            HW   = H * W
            starts = torch.arange(0, HW - tile_size + 1, tile_size, device=device)

            optimizer.zero_grad(set_to_none=True)
            for s in starts:
                e = s + tile_size
                with amp.autocast("cuda", dtype=torch.float16):
                    out = renderer(
                        cloud, pose,
                        tile_hw=tile_hw,
                        chunk_gauss=chunk_gauss,
                        tile_range=(s.item(), e.item())
                    )
                    pred_flat = out.view(imgs.size(0), 3, -1)
                    P         = pred_flat.size(-1)
                    gt_flat   = imgs.view(imgs.size(0), 3, -1)[..., s : s + P]
                    loss      = F.mse_loss(pred_flat, gt_flat)

                # NaN/Inf 체크 (필요시 주석 제거)
                # if torch.isnan(loss) or torch.isinf(loss):
                #     print(f"⚠️ Epoch {epoch} Step {steps}: bad loss={loss.item()}")
                #     return

                scaler.scale(loss).backward()
                total_loss += loss.item()
                steps      += 1

            scaler.step(optimizer)
            scaler.update()

        avg = total_loss / max(1, steps)
        print(f"Epoch {epoch:03d}/{max_epochs}  Loss: {avg:.6f}")

    print(f"Training is terminated at epoch: {max_epochs}.")

if __name__ == "__main__":
    main()
