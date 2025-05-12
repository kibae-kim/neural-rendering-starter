# scripts/generate_camera_path.py

import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────
# 프로젝트 루트/src 를 모듈 검색 경로에 추가 (한 줄)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
# ──────────────────────────────────────────────────────────

import json
from pathlib import Path
from gaussian.init_from_sfm import read_images

# 1) images.txt 경로 (lego 씬 기준)
images_txt = Path("data/nerf_synthetic/lego/colmap_output/sparse/0/images.txt")

# 2) COLMAP 이미지 메타 읽기
images_meta = read_images(images_txt)

# 3) ID 순서대로 정렬해서 qvec/tvec 리스트 만들기
camera_path = []
for img_id, meta in sorted(images_meta.items(), key=lambda x: x[0]):
    camera_path.append({
        "qvec": meta["qvec"].tolist(),
        "tvec": meta["tvec"].tolist()
    })

# 4) JSON으로 저장 (data/camera_path.json)
out_path = Path("data") / "camera_path.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(camera_path, f, indent=2)

print(f"camera_path.json saved to {out_path}")
