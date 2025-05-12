# train.py – Gaussian Splatting training script
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from gaussian.init_from_sfm import read_points3D, read_images, read_cameras, create_gaussian_cloud_from_points
from render import DifferentiableRenderer

class NeRFDataset(Dataset):
    def __init__(self, data_dir, images_txt, cameras_txt, image_size, transform=None):
        self.data_dir = Path(data_dir)
        self.image_paths = sorted((self.data_dir / 'images').glob('*'))
        self.images = read_images(Path(images_txt))
        self.cameras = read_cameras(Path(cameras_txt))
        self.image_size = image_size
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # assume filenames like "<id>.png" or "<id>.jpg"
        image_id = int(img_path.stem)
        return img, image_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Gaussian Splatting Renderer')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda', help='Compute device')
    args = parser.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config, 'r'))
    data_dir = cfg['data_dir']
    images_txt = cfg['images_txt']
    cameras_txt = cfg['cameras_txt']
    image_size = cfg.get('image_size', 64)
    batch_size = cfg.get('batch_size', 1)
    lr = cfg.get('lr', 1e-4)
    num_epochs = cfg.get('num_epochs', 10)
    near_plane = cfg.get('near_plane', 0.1)
    far_plane = cfg.get('far_plane', 100.0)

    # Read SfM data and create Gaussian cloud
    points3D = read_points3D(Path(data_dir) / 'points3D.txt')
    images = read_images(Path(images_txt))
    cameras = read_cameras(Path(cameras_txt))
    means, covs, colors = create_gaussian_cloud_from_points(points3D)

    device = torch.device(args.device)
    means = means.to(device)
    covs = covs.to(device)
    colors = colors.to(device)

    # Initialize renderer
    renderer = DifferentiableRenderer(
        means=means, covs=covs, colors=colors,
        image_size=image_size, tile_hw=image_size,
        near_plane=near_plane, far_plane=far_plane,
        cameras=cameras, device=device
    ).to(device)

    # Prepare dataset and loader
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    dataset = NeRFDataset(data_dir, images_txt, cameras_txt, image_size, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Optimizer and AMP scaler
    optimizer = torch.optim.Adam(renderer.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs+1):
        renderer.train()
        for i, (imgs, image_ids) in enumerate(loader):
            optimizer.zero_grad()
            imgs = imgs.to(device)  # [B,3,H,W]
            gt = imgs.view(imgs.shape[0], 3, -1)  # [B,3,N_rays]
            image_ids = image_ids.tolist()

            with torch.cuda.amp.autocast():
                pred = renderer(image_ids)  # [B,3,N_rays]
                # Align shapes
                if pred.shape[-1] != gt.shape[-1]:
                    gt = gt[..., :pred.shape[-1]]
                loss = F.mse_loss(pred, gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(renderer.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()

            if (i+1) % 10 == 0:
                print(f'Epoch {epoch} [{i+1}/{len(loader)}] Loss: {loss.item():.6f}')

    # Save final model
    torch.save(renderer.state_dict(), 'renderer_final.pth')
