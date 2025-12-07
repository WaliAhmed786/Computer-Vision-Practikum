import csv
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path.home() / "focuszoom"
JPG_DIR = PROJECT_ROOT / "data" / "dataset" / "train" / "olympus_macro_lens" / "needle1" / "jpg"
OUT_DIR = PROJECT_ROOT / "outputs" / "focus_needle1"
POINTS_CSV = OUT_DIR / "focus_points.csv"


def focus_map_laplacian(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    L = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    fm = (L - L.mean())**2
    fm = cv2.normalize(fm, None, 0, 1.0, cv2.NORM_MINMAX)
    return fm


def compute_focus_point(fm, top_percent=0.10):
    """
    fm: focus map (H x W)
    top_percent: fraction of most in-focus pixels to use (e.g. 0.10 = top 10%)

    returns (x_focus, y_focus, region_area)
    x is column index, y is row index
    """
    h, w = fm.shape
    flat = fm.flatten()
    sorted_vals = np.sort(flat)

    k = max(1, int(len(sorted_vals) * top_percent))
    thr = sorted_vals[-k]

    mask = fm >= thr
    idx_y, idx_x = np.where(mask)

    if len(idx_x) == 0:

        return w / 2.0, h / 2.0, 0

    x_mean = float(idx_x.mean())
    y_mean = float(idx_y.mean())
    region_area = int(mask.sum())
    return x_mean, y_mean, region_area


def main():
    print("Computing focus points for needle1 ...")
    print("JPG_DIR:", JPG_DIR)

    if not JPG_DIR.exists():
        print("ERROR: JPG_DIR does not exist!")
        return

    jpg_files = sorted(list(JPG_DIR.glob("*.jpg")) + list(JPG_DIR.glob("*.JPG")))
    print("Found", len(jpg_files), "JPG files.")
    if not jpg_files:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for p in jpg_files:
        print("Processing:", p.name)
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print("  WARNING: could not read, skipping.")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


        fm = focus_map_laplacian(rgb)


        x_focus, y_focus, region_area = compute_focus_point(fm, top_percent=0.10)


        flat = np.sort(fm.flatten())
        k = max(1, int(0.10 * len(flat)))
        focus_score = float(flat[-k:].mean())

        rows.append({
            "image": p.name,
            "focus_score": f"{focus_score:.6f}",
            "x_focus": f"{x_focus:.2f}",
            "y_focus": f"{y_focus:.2f}",
            "region_area": region_area,
        })


        overlay = rgb.copy()
        cv2.circle(
            overlay,
            (int(round(x_focus)), int(round(y_focus))),
            10,
            (255, 0, 0),
            thickness=2,
        )
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        out_img_path = OUT_DIR / f"{p.stem}_focuspoint.jpg"
        cv2.imwrite(str(out_img_path), overlay_bgr)


    with open(POINTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "focus_score", "x_focus", "y_focus", "region_area"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Saved focus points CSV to:", POINTS_CSV)
    print("Example rows:")
    for r in rows[:5]:
        print(" ", r)


if __name__ == "__main__":
    main()
