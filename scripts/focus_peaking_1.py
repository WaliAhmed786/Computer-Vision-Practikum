import cv2
import numpy as np
from pathlib import Path
import os
import csv

DATA_DIR = Path.home() / "focuszoom" / "data" / "dataset" / "train" / "olympus_macro_lens" / "needle1" / "jpg"
OUT_DIR = Path.home() / "focuszoom" / "outputs" / "focus_needle1"

def focus_map_laplacian(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    L = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    fm = (L - L.mean())**2
    fm = cv2.normalize(fm, None, 0, 1.0, cv2.NORM_MINMAX)
    return fm

def focus_score(fm):
    flat = np.sort(fm.flatten())
    k = max(1, int(0.10 * len(flat)))
    return float(flat[-k:].mean())

def main():
    print("DATA_DIR:", DATA_DIR)
    if not DATA_DIR.exists():
        print("ERROR: folder not found")
        return

    jpg_files = sorted(list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.JPG")))
    print("Found", len(jpg_files), "JPG files")
    if not jpg_files:
        return

    overlays_dir = OUT_DIR / "overlays"
    maps_dir = OUT_DIR / "maps"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    scores = []

    for p in jpg_files:
        print("Processing:", p.name)
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print("  could not read, skipping")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        fm = focus_map_laplacian(rgb)
        s = focus_score(fm)
        scores.append((p.name, s))


        fm_u8 = (fm * 255).astype(np.uint8)
        heat = cv2.applyColorMap(fm_u8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = (0.65 * rgb + 0.35 * heat_rgb).astype(np.uint8)

        base = os.path.splitext(p.name)[0]
        cv2.imwrite(str(maps_dir / f"{base}_fm.png"), fm_u8)
        cv2.imwrite(str(overlays_dir / f"{base}_overlay.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


    scores.sort(key=lambda x: x[1], reverse=True)


    csv_path = OUT_DIR / "focus_scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "focus_score"])
        for name, s in scores:
            w.writerow([name, f"{s:.6f}"])

    print("Saved overlays to:", overlays_dir)
    print("Saved maps to:", maps_dir)
    print("Saved scores to:", csv_path)
    print("Top 5 sharpest frames:")
    for name, s in scores[:5]:
        print(" ", name, "->", s)

if __name__ == "__main__":
    main()
