
# neural-rendering-starter  
**CS 445 – Kibae Kim (kibaek2)**

A minimal but end‑to‑end pipeline for learning and rendering **3‑D Gaussian Splatting** on the classic **NeRF‑synthetic “lego”** scene.  
Everything is pure PyTorch; no external C++/CUDA kernels are required.

---

## 📂 Directory layout (after clone)

```

neural-rendering-starter/
├── configs/
│   └── gaussian\_train.yaml      # training hyper‑params
├── data/
│   └── nerf\_synthetic/
│       └── lego/
│           ├── images/          # 200 RGB input frames  (r\_0.png … r\_199.png)
│           └── colmap\_output/   # sparse COLMAP model (cameras.txt, images.txt, points3D.txt | .bin)
├── src/
│   ├── gaussian/
│   │   ├── **init**.py
│   │   ├── gaussian\_point.py
│   │   └── init\_from\_sfm.py
│   ├── render.py
│   ├── train.py
│   └── **init**.py
└── requirements.txt

````

### Where did the **lego** data come from?

| Folder | Origin | Notes |
|--------|--------|-------|
| `images/` | [NeRF‑synthetic dataset](https://github.com/bmild/nerf) – **lego** scene, converted to 800×800 PNG | Licensing: Creative Commons BY‑SA‑4.0 (same as original NeRF repo) |
| `colmap_output/` | Generated **offline** with COLMAP 3.8:<br>`colmap automatic_reconstructor …` | Pre‑computed so that Colab users don’t need to run COLMAP (≈ minutes).<br>Both `.txt` and `.bin` formats are accepted. |

If you want to rebuild these files yourself, see `scripts/convert_colmap.py` or run COLMAP on the `images/` folder.

---

## 🛠 Installation

```bash
git clone https://github.com/<your‑github>/neural-rendering-starter.git
cd neural-rendering-starter
pip install -r requirements.txt
````

*Default `requirements.txt` assumes CUDA 11.8 wheels.  Adjust versions if your GPU/driver is older.*

---

## 🚀 Training (single GPU or Colab)


https://colab.research.google.com/drive/1-FKB89IfXQunT8GlKgti-ktlwxYejFoW#scrollTo=g3fCCwIqtVua

```bash
# fresh run
python src/train.py --cfg configs/gaussian_train.yaml

# resume from a checkpoint
python src/train.py --cfg configs/gaussian_train.yaml \
                    --resume output/exp1/epoch010.pt
```

* Checkpoints (`epochXXX.pt`) and preview renders (`renderXXX.png`) are written to `output/exp1/` every `save_every` epochs.
* Mixed‑precision (AMP) is on automatically if CUDA is available.

---

## ✨ Preview / Evaluation (optional)

```bash
# render a 360° camera path once training is done
python src/infer.py --ckpt output/exp1/epoch100.pt \
                    --camera_path data/camera_path.json
```

---

## 📝 Requirements

Installed automatically via **requirements.txt**:

* `torch`, `torchvision`
* `numpy`, `Pillow`, `PyYAML`
* `imageio`, `tqdm`
* *(optional)* `pycolmap` – only if you need BIN→TXT conversion inside Colab

---

## 🔖 Citation / Credits

* **NeRF‑synthetic dataset** – Ben Mildenhall et al., *ECCV 2020*
* **COLMAP** – Johannes L. Schönberger et al.
* **3‑D Gaussian Splatting** – Kerbl et al., *SIGGRAPH Asia 2023*

---

