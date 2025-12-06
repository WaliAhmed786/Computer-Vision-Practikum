import csv
from pathlib import Path
import os

import cv2
import numpy as np
import torch
import sys

PROJECT_ROOT = Path.home() / "focuszoom"
DATASET_CSV = PROJECT_ROOT / "data" / "dataset" / "dataset.csv"
DA_ROOT = PROJECT_ROOT / "Depth-Anything-V2"
CKPT_PATH = DA_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"

# make Depth-Anything-V2 importable
sys.path.append(str(DA_ROOT))
from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore


# ------------------ MODEL LOADING ------------------


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

    encoder = "vits"  # we use the small vits checkpoint
    model = DepthAnythingV2(**model_configs[encoder])
    state = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


# ------------------ SAVE DEPTH HELPERS ------------------


def save_depth(depth: np.ndarray, out_npy: Path, out_png: Path):
    """
    depth: HxW numpy array (raw depth from model.infer_image)
    """
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), depth)

    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max - d_min < 1e-8:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    else:
        depth_norm = (depth - d_min) / (d_max - d_min)

    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_png), depth_color)


# ------------------ FOCUS SCORES -> SHARP + BLURRIEST ------------------


def get_top_and_bottom_frames(photo: str, n_top: int = 3, n_bottom: int = 3):
    """
    Reads outputs/focus_<photo>/focus_scores.csv and returns:
      - list of top-n sharpest image names
      - list of bottom-n blurriest image names
    """
    focus_dir = PROJECT_ROOT / "outputs" / f"focus_{photo}"
    scores_csv = focus_dir / "focus_scores.csv"
    if not scores_csv.exists():
        print(f"  [SKIP] focus_scores.csv not found for {photo}: {scores_csv}")
        return [], []

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
        return [], []

    # sort descending by score (sharpest first)
    rows.sort(key=lambda x: x[1], reverse=True)

    top_frames = [name for name, _ in rows[:n_top]]
    bottom_frames = [name for name, _ in rows[-n_bottom:]] if len(rows) >= n_bottom else [name for name, _ in rows[-len(rows):]]

    print(f"  Top {n_top} frames:", top_frames)
    print(f"  Bottom {n_bottom} frames:", bottom_frames)
    return top_frames, bottom_frames


# ------------------ FIND HELICON IMAGE ------------------


def find_helicon_image(base_dir: Path):
    """
    Try to find an all-in-focus image produced by Helicon Focus.
    We check for a few common filenames.
    """
    candidates = [
        "helicon_focus_aligned_tiff_cropped.jpg",
        "helicon_focus_cropped.jpg",
        "helicon_focus_aligned_tiff.jpg",
        "helicon_focus.jpg",
    ]

    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p

    return None


# ------------------ PROCESS ONE STACK ------------------


def process_stack(model, device: str, set_name: str, lens: str, photo: str):
    base_dir = PROJECT_ROOT / "data" / "dataset" / set_name / lens / photo
    jpg_dir = base_dir / "jpg"

    if not jpg_dir.exists():
        print(f"[SKIP] JPG dir does not exist: {jpg_dir}")
        return

    out_dir = PROJECT_ROOT / "outputs" / f"depth_{photo}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing depth for stack: set={set_name}, lens={lens}, photo={photo}")

    # 1) get top-3 sharpest + bottom-3 blurriest
    top_frames, bottom_frames = get_top_and_bottom_frames(photo, n_top=3, n_bottom=3)
    # combine, but avoid duplicates (in case stack is tiny)
    all_frames = []
    for name in top_frames + bottom_frames:
        if name not in all_frames:
            all_frames.append(name)

    # 2) depth for those frames
    for img_name in all_frames:
        img_path = jpg_dir / img_name
        if not img_path.exists():
            print(f"  [WARN] image listed in focus_scores but not found: {img_path}")
            continue

        stem = os.path.splitext(img_name)[0]
        out_npy = out_dir / f"{stem}_depth.npy"
        out_png = out_dir / f"{stem}_depth.png"

        if out_png.exists() and out_npy.exists():
            print(f"  [SKIP] depth already exists for {img_name}")
            continue

        print("  Depth for frame:", img_name)
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("    [WARN] could not read image, skipping.")
            continue

        depth = model.infer_image(bgr)
        save_depth(depth, out_npy, out_png)
        print("    Saved depth to:", out_png)

    # 3) depth for all-in-focus helicon image, if available
    helicon_path = find_helicon_image(base_dir)
    if helicon_path is not None:
        out_npy = out_dir / "helicon_focus_depth.npy"
        out_png = out_dir / "helicon_focus_depth.png"

        if out_png.exists() and out_npy.exists():
            print("  [SKIP] helicon depth already exists.")
        else:
            print("  Depth for all-in-focus (helicon):", helicon_path.name)
            bgr = cv2.imread(str(helicon_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print("    [WARN] could not read helicon image, skipping.")
            else:
                depth_h = model.infer_image(bgr)
                save_depth(depth_h, out_npy, out_png)
                print("    Saved helicon depth to:", out_png)
    else:
        print("  No helicon_focus*.jpg found for this stack.")


# ------------------ MAIN ------------------


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

    for r in rows:
        set_name = r["set"]
        if set_name not in ("train", "test"):
            continue

        lens = r["lens"]
        photo = r["photo"]
        process_stack(model, device, set_name, lens, photo)


if __name__ == "__main__":
    main()
