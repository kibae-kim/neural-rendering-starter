# Gaussian‑Splat training – tile 64px, chunk 8192, AMP‑FP16
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

# ------------------- Dataset -------------------
class NeRFDataset(Dataset):
    def __init__(self, img_dir, images_txt, imsize):
        self.paths = sorted(p for p in Path(img_dir).glob("*.png")
                            if "_depth" not in p.name)

        meta_raw = read_images(images_txt)
        self.id2meta = {int("".join(filter(str.isdigit, Path(m['file']).stem))): m
                        for m in meta_raw.values()}

        self.tf = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        digits = "".join(filter(str.isdigit, p.stem))
        if not digits or int(digits) not in self.id2meta:
            return None   # 메타 없으면 스킵

        m = self.id2meta[int(digits)]
        img = self.tf(Image.open(p).convert("RGB"))
        pose = {'qvec': torch.from_numpy(m['qvec']).float().unsqueeze(0),
                'tvec': torch.from_numpy(m['tvec']).float().unsqueeze(0)}
        return img, pose
# ------------------------------------------------

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    imgs, poses = zip(*batch)
    imgs  = torch.stack(imgs, 0)
    qvecs = torch.cat([p['qvec'] for p in poses], 0)
    tvecs = torch.cat([p['tvec'] for p in poses], 0)
    return imgs, {'qvec': qvecs, 'tvec': tvecs}

def main():
    root = Path(__file__).resolve().parents[1]
    cfg  = yaml.safe_load(open(root / "configs" / "gaussian_train.yaml"))
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xyz, rgb = read_points3D(root / cfg['data']['root'] /
                             cfg['data']['colmap_output_dir'] / "sparse/0/points3D.txt")
    cloud = create_gaussian_cloud_from_points((xyz, rgb)).to(dev)

    renderer = DifferentiableRenderer(cfg['render']['image_size']).to(dev)

    ds = NeRFDataset(root/cfg['data']['root']/cfg['data']['images_dir'],
                     root/cfg['data']['root']/cfg['data']['colmap_output_dir']/
                     "sparse/0/images.txt",
                     cfg['render']['image_size'])
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate)

    opt = torch.optim.Adam(cloud.parameters(), lr=cfg['train']['learning_rate'])
    scaler = amp.GradScaler()

    tile_pix = 64*64
    for epoch in range(1, cfg['train']['epochs']+1):
        loss_sum = 0.
        for batch in dl:
            if batch is None: continue              # 전부 스킵 된 경우
            img, pose = batch
            img  = img.to(dev, non_blocking=True)
            pose = {k: v.to(dev) for k, v in pose.items()}
            HW = img.shape[-2] * img.shape[-1]
            tiles = torch.arange(0, HW - tile_pix + 1, tile_pix, device=dev)

            opt.zero_grad(set_to_none=True)
            for s in tiles:
                e = s + tile_pix
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = renderer(cloud, pose,
                                    tile_hw=64, chunk_gauss=8192,
                                    tile_range=(s.item(), e.item()))  # (B,3,64,64)
                    pred = pred.view(img.size(0), 3, -1)
                    gt   = img.view(img.size(0), 3, -1)[:, :, s:e]
                    loss = F.mse_loss(pred, gt)
                scaler.scale(loss).backward()
                loss_sum += loss.item()

            scaler.step(opt); scaler.update()

        print(f"Epoch {epoch:03d} | Loss {loss_sum / len(ds):.6f}")

if __name__ == "__main__":
    main()
