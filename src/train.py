# tile‑wise backward, image‑wise optimizer.step (tile=32px, chunk 4096)
import yaml, torch
from torch import amp
import torch.nn.functional as F
from pathlib import Path
from gaussian.init_from_sfm import (
    read_points3D, read_images, create_gaussian_cloud_from_points)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from render import DifferentiableRenderer

# ---------- Dataset ----------
class DS(Dataset):
    def __init__(self, root_img, img_meta, imsize):
        self.paths = sorted(p for p in Path(root_img).glob("r_*.png") if "_depth" not in p.name)
        meta_raw = read_images(img_meta)
        self.idx2meta = {int(''.join(filter(str.isdigit, m['file'].split('.')[0]))): m
                         for m in meta_raw.values()}
        self.tf = transforms.Compose([transforms.Resize(tuple(imsize)), transforms.ToTensor()])

    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        p = self.paths[i]; img = self.tf(Image.open(p).convert("RGB"))
        idx = int(''.join(filter(str.isdigit, p.stem)))
        m = self.idx2meta[idx]
        pose = {'qvec': torch.from_numpy(m['qvec']), 'tvec': torch.from_numpy(m['tvec'])}
        return img, pose

# ---------- main ----------
def main():
    root = Path(__file__).resolve().parents[1]
    cfg  = yaml.safe_load(open(root/"configs/gaussian_train.yaml"))
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cloud
    xyz,rgb = read_points3D(root/cfg['data']['root']/cfg['data']['colmap_output_dir']/"sparse/0/points3D.txt")
    cloud = create_gaussian_cloud_from_points((xyz,rgb)).to(dev)

    # renderer
    renderer = DifferentiableRenderer(cfg['render']['image_size']).to(dev)

    # dataloader
    ds = DS(root/cfg['data']['root']/cfg['data']['images_dir'],
            root/cfg['data']['root']/cfg['data']['colmap_output_dir']/"sparse/0/images.txt",
            cfg['render']['image_size'])
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(cloud.parameters(), lr=cfg['train']['learning_rate'])
    scaler = amp.GradScaler()

    step_pix = 32*32
    for epoch in range(1, cfg['train']['epochs']+1):
        loss_epoch = 0.
        for img, pose in dl:
            img = img.to(dev); pose = {k:v.to(dev) for k,v in pose.items()}
            HW = img.shape[-2]*img.shape[-1]
            tiles = torch.arange(0, HW - step_pix + 1, step_pix, device=dev)

            opt.zero_grad(set_to_none=True)

            for start in tiles:
                end = start + step_pix
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = renderer(cloud, pose,
                                    tile_hw=32, chunk_gauss=4096,
                                    tile_range=(start, end))[0]     # (3,32,32)
                    loss = F.mse_loss(pred.view(3,-1), img.view(3,-1)[:, start:end])

                scaler.scale(loss).backward()
                loss_epoch += loss.item()

            scaler.step(opt); scaler.update()

        print(f"E{epoch:03d} | loss={loss_epoch/len(dl):.6f}")

if __name__ == "__main__":
    main()
