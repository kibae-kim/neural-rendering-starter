# src/gaussian/init_from_sfm.py
# ------------------------------------------------------------------
#  COLMAP → GaussianPointCloud 파서 + 빌더
#  - .txt / .bin 모두 지원 (points3D)
#  - scale_variance → (N,1) σ 텐서 생성하여 생성자에 넘김
# ------------------------------------------------------------------

import struct
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

from gaussian.gaussian_point import GaussianPointCloud
# GaussianPointCloud.__init__(positions, colors, scales) 형식이어야 함
# ------------------------------------------------------------------


# ======================= 1. COLMAP parsing ========================

def _read_points3d_txt(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            elems = ln.split()
            xyz.append(list(map(float, elems[1:4])))
            rgb.append([int(c) / 255.0 for c in elems[4:7]])
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32)


def _read_points3d_bin(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reference: https://github.com/colmap/colmap/blob/dev/src/base/reconstruction_io.cc
    """
    xyz, rgb = [], []
    with open(path, "rb") as f:
        while True:
            header = f.read(43 + 8)  # 정확히 51 bytes = 43 + 8 (error)
            if not header:
                break
            if len(header) < 51:
                break  # corrupt or truncated

            # 첫 43바이트: id + xyz + rgb + error
            unpacked = struct.unpack("<QdddBBBd", header)
            _, x, y, z, r, g, b, _ = unpacked
            xyz.append([x, y, z])
            rgb.append([r / 255.0, g / 255.0, b / 255.0])

            # 다음 8바이트: track 길이
            track_len_bytes = f.read(8)
            if not track_len_bytes:
                break
            track_len = struct.unpack("<Q", track_len_bytes)[0]

            # track (image_id, point2D_idx) x track_len
            f.seek(track_len * 8 * 2, 1)

    return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32)


def read_points3D(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    xyz : (N,3) float32
    rgb : (N,3) float32, range 0‒1
    """
    path = str(path)
    return _read_points3d_bin(path) if path.endswith(".bin") else _read_points3d_txt(path)


def read_images(path: Union[str, Path]):
    if str(path).endswith(".bin"):
        raise NotImplementedError(
            "images.bin parsing not implemented. "
            "Convert via `colmap model_converter ... --output_type TXT`, "
            "or implement binary parser first."
        )
    images = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#") or not ln.strip():
            i += 1
            continue
        elems = ln.split()
        iid = int(elems[0])
        images[iid] = dict(
            qvec=np.asarray(elems[1:5], np.float32),
            tvec=np.asarray(elems[5:8], np.float32),
            camera_id=int(elems[8]),
            file=elems[9],
        )
        i += 2
    return images


def read_cameras(path: Union[str, Path]):
    if str(path).endswith(".bin"):
        raise NotImplementedError(
            "cameras.bin parsing not implemented. "
            "Field layout: <camera_id><model_id><w><h><params[N]>.\n"
            "Refer to COLMAP `camera_models.h` and extend parser accordingly."
        )
    cams = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            elems = ln.split()
            cid = int(elems[0])
            cams[cid] = dict(
                model=elems[1],
                width=int(elems[2]),
                height=int(elems[3]),
                params=np.asarray(elems[4:], np.float32),
            )
    return cams


# ======================= 2. Cloud generator ========================

def create_gaussian_cloud_from_points(
    points_xyz_rgb: Tuple[np.ndarray, np.ndarray],
    scale: float = 1.0,
    scale_variance: bool = False,
) -> GaussianPointCloud:
    """
    Parameters
    ----------
    points_xyz_rgb : (xyz, rgb) ndarray tuple
    scale          : base σ value when scale_variance=False
    scale_variance : if True, per‑point σ sampled in [0.5*scale, 1.5*scale]

    Returns
    -------
    GaussianPointCloud
    """
    xyz, rgb = points_xyz_rgb
    xyz_t = torch.from_numpy(xyz)
    rgb_t = torch.from_numpy(rgb)

    if scale_variance:
        sigmas = torch.rand(len(xyz), 1, dtype=torch.float32) * scale + 0.5 * scale
    else:
        sigmas = torch.full((len(xyz), 1), scale, dtype=torch.float32)  # 안전하게 σ 고정값 적용

    return GaussianPointCloud(
        positions=xyz_t,
        colors=rgb_t,
        scales=sigmas,
    )
