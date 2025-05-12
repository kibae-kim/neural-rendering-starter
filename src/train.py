# Gaussian‑Splat training  (tile‑32 px backward, image‑wise step)
import yaml, torch
from pathlib import Path
from PIL import Image
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from gaussian.init_from_sfm import (
    read_points3D, read_images, create_gaussian_cloud_from_points)
from render import DifferentiableRenderer

# ---------------------- Dataset ----------------------
class NeRFDataset(Dataset):
    def __init__(self, img_dir, images_txt, imsize):
        self.paths = sorted(p for p in Path(img_dir).glob("r_*.png")
                            if "_depth" not in p.name)

        meta_raw = read_images(images_txt)
        # 파일명 그대로 키로 매핑  (KeyError 방지)
        self.file2meta = {Path(m['file']).name: m for m in meta_raw.values()}

        self.tf = transforms.Compose([
            transforms.Resize(tuple(imsize)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        m = self.file2meta.get(p.name)
        if m is None:
            raise RuntimeError(f"metadata for {p.name} not found")

        img = self.tf(Image.open(p).convert("RGB"))
        pose = {
            'qvec': torch.from_numpy(m['qvec']),
            'tvec': torch.from_numpy(m['tvec'])
        }
        return img, pose
# -----------------------------------------------------

def main():
    root = Path(__file__).resolve().parents[1]
    cfg  = yaml.safe_load(open(root / "configs" / "gaussian_train.yaml"))
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gaussian cloud
    pts_path = (root / cfg['data']['root'] /
                cfg['data']['colmap_output_dir'] / "sparse/0/points3D.txt")
    xyz, rgb = read_points3D(pts_path)
    cloud = create_gaussian_cloud_from_points((xyz, rgb)).to(dev)

    # Renderer
    renderer = DifferentiableRenderer(cfg['render']['image_size']).to(dev)

    # Dataloader
    img_dir = root / cfg['data']['root'] / cfg['data']['images_dir']
    img_txt = (root / cfg['data']['root'] /
               cfg['data']['colmap_output_dir'] / "sparse/0/images.txt")
    ds = NeRFDataset(img_dir, img_txt, cfg['render']['image_size'])
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(cloud.parameters(), lr=cfg['train']['learning_rate'])
    scaler = amp.GradScaler()

    tile_pix = 32 * 32
    for epoch in range(1, cfg['train']['epochs'] + 1):
        loss_sum = 0.
        for img, pose in dl:
            img = img.to(dev); pose = {k: v.to(dev) for k, v in pose.items()}
            HW = img.shape[-2] * img.shape[-1]
            tiles = torch.arange(0, HW - tile_pix + 1, tile_pix, device=dev)

            opt.zero_grad(set_to_none=True)

            for start in tiles:
                end = start + tile_pix
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = renderer(cloud, pose,
                                    tile_hw=32, chunk_gauss=4096,
                                    tile_range=(start, end))[0]  # (3,32,32)
                    loss = F.mse_loss(pred.view(3, -1),
                                      img.view(3, -1)[:, start:end])
                scaler.scale(loss).backward()
                loss_sum += loss.item()

            scaler.step(opt); scaler.update()

        print(f"Epoch {epoch:03d} | Loss {loss_sum / len(dl):.6f}")

if __name__ == "__main__":
    main()
