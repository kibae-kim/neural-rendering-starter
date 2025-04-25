#init_from_sfm.py

import numpy as np
from pathlib import Path
from gaussian.gaussian_point import GaussianPointCloud
import touch

##########################################
# Parsing COLMAP's result to numpy array #
##########################################

def read_points3D(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        elems = line.strip().split()
        xyz = np.array(elems[1:4], dtype=np.float32)
        rgb = np.array(elems[4:7], dtype=np.float32) / 255.0
        points.append((xyz, rgb))
    return points

def read_images(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    images = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('#') or line.strip() == '':
            i += 1
            continue
        elems = line.strip().split()
        image_id = int(elems[0])
        qvec = np.array(elems[1:5], dtype=np.float32)  # rotation
        tvec = np.array(elems[5:8], dtype=np.float32)  # translation
        filename = elems[-1]
        images[image_id] = {
            'qvec': qvec,
            'tvec': tvec,
            'file': filename
        }
        i += 2  # 이미지 정보는 2줄 단위
    return images

def read_cameras(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    cameras = {}
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        elems = line.strip().split()
        camera_id = int(elems[0])
        model = elems[1]
        width = int(elems[2])
        height = int(elems[3])
        params = np.array(elems[4:], dtype=np.float32)
        cameras[camera_id] = {
            'model': model,
            'width': width,
            'height': height,
            'params': params
        }
    return cameras

#############
# Generator #
#############

def create_gaussian_cloud_from_points(points_list, scale=1.0):
    """
    points_list: List of (xyz, rgb) tuples
    """
    positions = []
    colors = []
    for xyz, rgb in points_list:
        positions.append(xyz)
        colors.append(rgb)
    positions = torch.tensor(positions, dtype=torch.float32)
    colors = torch.tensor(colors, dtype=torch.float32)
    return GaussianPointCloud(positions, colors, scale)

