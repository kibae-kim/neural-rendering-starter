# Gaussian‑Splat training (tile‑wise backward → O(GB) 메모리)
import argparse, yaml, torch
from pathlib import Path
from PIL import Image
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from gaussian.init_from_sfm import read_points3D, read_images, create_gaussian_cloud_from_points
from render import DifferentiableRenderer

# ---------------- Dataset ----------------
class NeRFDataset(Dataset):
    def __init__(self, img_dir, images_txt, image_size):
        self.paths = sorted(p for p in Path(img_dir).glob("r_*.png") if "_depth" not in p.name)
        meta_raw = read_images(images_txt)
        self.idx2meta = {int(''.join(filter(str.isdigit, m['file'].split('.')[0]))): m
                         for m in meta_raw.values()}
        self.tf = transforms.Compose([transforms.Resize(tuple(image_size)), transforms.ToTensor()])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]; img = self.tf(Image.open(p).convert("RGB"))
        idx = int(''.join(filter(str.isdigit, p.stem)))
        m = self.idx2meta[idx]
        pose = {'qvec': torch.from_numpy(m['qvec']), 'tvec': torch.from_numpy(m['tvec'])}
        return img, pose
# -----------------------------------------

def main():
    root = Path(__file__).resolve().parents[1]
    cfg  = yaml.safe_load(open(root/"configs/gaussian_train.yaml"))
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gaussian cloud
    xyz,rgb = read_points3D(root/cfg['data']['root']/cfg['data']['colmap_output_dir']/"sparse/0/points3D.txt")
    cloud   = create_gaussian_cloud_from_points((xyz,rgb)).to(dev)

    # Renderer
    renderer = DifferentiableRenderer(cfg['render']['image_size']).to(dev)

    # DataLoader
    ds = NeRFDataset(root/cfg['data']['root']/cfg['data']['images_dir'],
                     root/cfg['data']['root']/cfg['data']['colmap_output_dir']/"sparse/0/images.txt",
                     cfg['render']['image_size'])
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(cloud.parameters(), lr=cfg['train']['learning_rate'])
    scaler = amp.GradScaler()

    for epoch in range(1, cfg['train']['epochs']+1):
        loss_epoch = 0.
        for img, pose in dl:
            img = img.to(dev)
            pose = {k: v.to(dev) for k,v in pose.items()}

            opt.zero_grad(set_to_none=True)
            loss_sum = 0.

            # tile list (8×8 px)
            HW = img.shape[-2]*img.shape[-1]
            tiles = torch.arange(0, HW, 64, device=dev)

            for t in tiles:
                end = min(t+64, HW)
                with amp.autocast(device_type="cuda", dtype=torch.float16): #Changed
                    #pred_tile = renderer(cloud, pose, tile_hw=8,
                    #                     chunk_gauss=256, tile_range=(t, end))[0]
                    #gt_tile   = img.view(3,-1)[:, t:end]
                    #loss_t = F.mse_loss(pred_tile, gt_tile)
                    pred_tile = renderer(cloud, pose, tile_hw=8, chunk_gauss=256,
                                        tile_range=(t, end))[0] #(3,16,8)
                    pred_flat = pred_tile.view(3, -1) #(3,128)->(3,64)
                    loss_t = F.mse_loss(pred_flat, img.view(3,-1)[:, t:end])

                scaler.scale(loss_t).backward()
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                loss_sum += loss_t.detach()

            loss_epoch += loss_sum.item()
            torch.cuda.empty_cache()

        print(f"E{epoch:03d}  loss={loss_epoch/len(dl):.6f}")

if __name__ == "__main__":
    main()
