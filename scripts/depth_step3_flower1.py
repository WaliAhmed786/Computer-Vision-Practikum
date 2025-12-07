import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import sys

PROJECT_ROOT = Path.home() / "focuszoom"
DA_ROOT = PROJECT_ROOT / "Depth-Anything-V2"

sys.path.append(str(DA_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  


JPG_DIR = PROJECT_ROOT / "data" / "dataset" / "train" / "olympus_macro_lens" / "flower1" / "jpg"
SCORES_CSV = PROJECT_ROOT / "outputs" / "focus_flower1" / "focus_scores.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "depth_flower1"

CKPT_PATH = DA_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"


model_configs = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
ENCODER_KEY = "vits"


def load_model():
    """Load DepthAnythingV2-small (vits) on CPU."""
    print("Loading DepthAnythingV2 model from:", CKPT_PATH)
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

    cfg = model_configs[ENCODER_KEY]
    model = DepthAnythingV2(**cfg)

    state = torch.load(str(CKPT_PATH), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_depth(model, bgr_image: np.ndarray) -> np.ndarray:
    """
    Run DepthAnythingV2 on a single BGR image (as read by cv2.imread).
    Returns a HxW float32 depth map (numpy array).
    """
    with torch.no_grad():
        depth = model.infer_image(bgr_image)
    depth = depth.astype(np.float32)
    return depth


def main(top_n: int = 3):
    print("Reading focus scores from:", SCORES_CSV)

    if not SCORES_CSV.exists():
        print("ERROR: scores CSV not found.")
        return


    rows = []
    with open(SCORES_CSV, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["image"], float(r["focus_score"])))

    if not rows:
        print("No rows in scores CSV.")
        return

    rows.sort(key=lambda x: x[1], reverse=True)

    print(f"Total frames with scores: {len(rows)}")
    print(f"Using top {top_n} frames for depth estimation:")
    for name, score in rows[:top_n]:
        print(" ", name, "->", score)

    OUT_DIR.mkdir(parents=True, exist_ok=True)


    model = load_model()


    for name, score in rows[:top_n]:
        img_path = JPG_DIR / name
        print("\nProcessing image:", img_path)

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("  ERROR: cannot read image, skipping.")
            continue


        depth = infer_depth(model, bgr)


        base = img_path.stem
        npy_path = OUT_DIR / f"{base}_depth.npy"
        np.save(str(npy_path), depth)
        print("  Saved depth npy to:", npy_path)


        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min < 1e-8:
            d_norm = np.zeros_like(depth, dtype=np.float32)
        else:
            d_norm = (depth - d_min) / (d_max - d_min)
        d_u8 = (d_norm * 255).astype(np.uint8)
        png_path = OUT_DIR / f"{base}_depth.png"
        cv2.imwrite(str(png_path), d_u8)
        print("  Saved depth vis to:", png_path)

    print("\nDone. Depth results in:", OUT_DIR)


if __name__ == "__main__":
    main(top_n=3)

