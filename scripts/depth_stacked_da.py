from pathlib import Path
import cv2
import numpy as np
import torch
import sys

PROJECT_ROOT = Path.home() / "focuszoom"
DA_ROOT = PROJECT_ROOT / "Depth-Anything-V2"
CKPT_PATH = DA_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"

sys.path.append(str(DA_ROOT))
from depth_anything_v2.dpt import DepthAnythingV2


def load_model(device: str = "cpu"):
    print("Loading DepthAnythingV2 model from:", CKPT_PATH)
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    }

    model = DepthAnythingV2(**model_configs["vits"])
    state = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def save_depth(depth: np.ndarray, out_npy: Path, out_png: Path):
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python depth_stacked_da.py <scene_name>")
        print("Example: python depth_stacked_da.py needle1")
        return

    scene = sys.argv[1]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    model = load_model(device=device)

    img_path = PROJECT_ROOT / "outputs" / "stacked" / scene / f"{scene}_stacked.jpg"
    out_dir = PROJECT_ROOT / "outputs" / f"depth_{scene}"
    out_npy = out_dir / f"{scene}_stacked_da_depth.npy"
    out_png = out_dir / f"{scene}_stacked_da_depth.png"

    print(f"Scene: {scene}")
    print("Reading image:", img_path)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print("ERROR: could not read image.")
        return

    print("Running DepthAnythingV2 inference...")
    depth = model.infer_image(bgr)
    save_depth(depth, out_npy, out_png)
    print("Saved DA depth to:", out_png)


if __name__ == "__main__":
    main()
