import csv
from pathlib import Path
import os

import cv2
import numpy as np
import torch

# --- Paths ---

PROJECT_ROOT = Path.home() / "focuszoom"
DATASET_CSV = PROJECT_ROOT / "data" / "dataset" / "dataset.csv"
DA_ROOT = PROJECT_ROOT / "Depth-Anything-V2"
CKPT_PATH = DA_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"

# --- DepthAnythingV2 setup (from official README) ---

import sys
sys.path.append(str(DA_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore


def load_model(device: str = "cpu"):
    print("Loading DepthAnythingV2 model from:", CKPT_PATH)
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    encoder = "vits"  # we have depth_anything_v2_vits.pth
    model = DepthAnythingV2(**model_configs[encoder])
    state = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


# --- Utility: save depth as npy + color PNG ---


def save_depth(depth: np.ndarray, out_npy: Path, out_png: Path):
    """
    depth: HxW numpy array (raw depth from model.infer_image)
    """
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), depth)

    # normalize for visualization
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max - d_min < 1e-8:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    else:
        depth_norm = (depth - d_min) / (d_max - d_min)

    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_png), depth_color)


# --- Read top-N sharp frames from focus_scores.csv ---


def get_top_n_frames(photo: str, n: int = 3):
    focus_dir = PROJECT_ROOT / "outputs" / f"focus_{photo}"
    scores_csv = focus_dir / "focus_scores.csv"
    if not scores_csv.exists():
        print(f"  [SKIP] focus_scores.csv not found for {photo}: {scores_csv}")
        return []

    rows = []
    with open(scores_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                score = float(r["focus_score"])
            except Exception:
                continue
            rows.append((r["image"], score))

    if not rows:
        print(f"  [SKIP] No rows in {scores_csv}")
        return []

    rows.sort(key=lambda x: x[1], reverse=True)
    top = [name for name, _ in rows[:n]]
    print("  Top", n, "frames from focus_scores:", top)
    return top


# --- Find helicon (all-in-focus) image in a stack ---


def find_helicon_image(base_dir: Path):
    """
    Try to find an all-in-focus image produced by Helicon Focus.
    We check for a few common filenames.
    """
    candidates = [
        "helicon_focus_aligned_tiff_cropped.jpg",
        "helicon_focus_aligned_tiff.jpg",
        "helicon_focus_cropped.jpg",
        "helicon_focus.jpg",
    ]

    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p

    # sometimes inside 'aligned' or other folder – we keep it simple for now
    return None


# --- Process one stack (one row from dataset.csv) ---


def process_stack(model, device: str, set_name: str, lens: str, photo: str):
    base_dir = PROJECT_ROOT / "data" / "dataset" / set_name / lens / photo
    jpg_dir = base_dir / "jpg"

    if not jpg_dir.exists():
        print(f"[SKIP] JPG dir does not exist: {jpg_dir}")
        return

    out_dir = PROJECT_ROOT / "outputs" / f"depth_{photo}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing depth for stack: set={set_name}, lens={lens}, photo={photo}")

    # 1) depth for top-N sharp frames
    top_frames = get_top_n_frames(photo, n=3)
    for img_name in top_frames:
        img_path = jpg_dir / img_name
        if not img_path.exists():
            print(f"  [WARN] image listed in focus_scores but not found: {img_path}")
            continue

        print("  Depth for sharp frame:", img_name)
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("    [WARN] could not read image, skipping.")
            continue

        # DepthAnythingV2 uses raw BGR / RGB; README example uses cv2.imread
        depth = model.infer_image(bgr)  # HxW numpy array

        stem = os.path.splitext(img_name)[0]
        out_npy = out_dir / f"{stem}_depth.npy"
        out_png = out_dir / f"{stem}_depth.png"
        save_depth(depth, out_npy, out_png)
        print("    Saved depth to:", out_png)

    # 2) depth for all-in-focus helicon image, if available
    helicon_path = find_helicon_image(base_dir)
    if helicon_path is not None:
        print("  Depth for all-in-focus (helicon):", helicon_path.name)
        bgr = cv2.imread(str(helicon_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("    [WARN] could not read helicon image, skipping.")
        else:
            depth_h = model.infer_image(bgr)
            out_npy = out_dir / "helicon_focus_depth.npy"
            out_png = out_dir / "helicon_focus_depth.png"
            save_depth(depth_h, out_npy, out_png)
            print("    Saved helicon depth to:", out_png)
    else:
        print("  No helicon_focus*.jpg found for this stack.")

    print()


# --- Main ---


def main():
    print("Reading dataset.csv from:", DATASET_CSV)
    if not DATASET_CSV.exists():
        print("ERROR: dataset.csv not found")
        return

    with open(DATASET_CSV, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    print("Total rows:", len(rows))
    if not rows:
        return

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    model = load_model(device=device)

    # In your dataset, all rows are 'train', but keep filter for clarity
    for r in rows:
        set_name = r["set"]
        if set_name not in ("train", "test"):
            continue

        lens = r["lens"]
        photo = r["photo"]
        process_stack(model, device, set_name, lens, photo)


if __name__ == "__main__":
    main()
