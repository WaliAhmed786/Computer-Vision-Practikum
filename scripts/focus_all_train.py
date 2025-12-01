import csv
from pathlib import Path
import cv2
import numpy as np
import os

PROJECT_ROOT = Path.home() / "focuszoom"
DATASET_CSV = PROJECT_ROOT / "data" / "dataset" / "dataset.csv"


def focus_map_laplacian(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    L = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    fm = (L - L.mean())**2
    fm = cv2.normalize(fm, None, 0, 1.0, cv2.NORM_MINMAX)
    return fm


def focus_score(fm, top_percent=0.10):
    flat = np.sort(fm.flatten())
    k = max(1, int(top_percent * len(flat)))
    return float(flat[-k:].mean())


def compute_focus_point(fm, top_percent=0.10):
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


def process_stack(set_name: str, lens: str, photo: str):
    """
    Run focus peaking + focus point computation for one stack:
    data/dataset/<set>/<lens>/<photo>/jpg
    """
    base_dir = PROJECT_ROOT / "data" / "dataset" / set_name / lens / photo
    jpg_dir = base_dir / "jpg"

    if not jpg_dir.exists():
        print(f"  [SKIP] JPG dir does not exist: {jpg_dir}")
        return

    out_dir = PROJECT_ROOT / "outputs" / f"focus_{photo}"
    overlays_dir = out_dir / "overlays"
    maps_dir = out_dir / "maps"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(list(jpg_dir.glob("*.jpg")) + list(jpg_dir.glob("*.JPG")))
    if not jpg_files:
        print(f"  [SKIP] No JPG files in {jpg_dir}")
        return

    print(f"  Processing stack: set={set_name}, lens={lens}, photo={photo}")
    print(f"    JPG files: {len(jpg_files)}")

    score_rows = []
    point_rows = []

    for p in jpg_files:
        print("    Image:", p.name)
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print("      WARNING: could not read, skipping.")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Focus map
        fm = focus_map_laplacian(rgb)

        # Focus score
        s = focus_score(fm, top_percent=0.10)
        score_rows.append((p.name, s))

        # Save focus map + overlay (like before)
        fm_u8 = (fm * 255).astype(np.uint8)
        heat = cv2.applyColorMap(fm_u8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = (0.65 * rgb + 0.35 * heat_rgb).astype(np.uint8)

        base = os.path.splitext(p.name)[0]
        cv2.imwrite(str(maps_dir / f"{base}_fm.png"), fm_u8)
        cv2.imwrite(
            str(overlays_dir / f"{base}_overlay.jpg"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        # Focus point
        x_focus, y_focus, region_area = compute_focus_point(fm, top_percent=0.10)
        point_rows.append({
            "image": p.name,
            "focus_score": s,
            "x_focus": x_focus,
            "y_focus": y_focus,
            "region_area": region_area,
        })

        # Save overlay with focus point
        overlay_fp = rgb.copy()
        cv2.circle(
            overlay_fp,
            (int(round(x_focus)), int(round(y_focus))),
            10,
            (255, 0, 0),  # red
            thickness=2,
        )
        overlay_fp_bgr = cv2.cvtColor(overlay_fp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{base}_focuspoint.jpg"), overlay_fp_bgr)

    # Sort scores (highest = sharpest)
    score_rows.sort(key=lambda x: x[1], reverse=True)

    # Save scores CSV
    scores_csv = out_dir / "focus_scores.csv"
    with open(scores_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "focus_score"])
        for name, s in score_rows:
            w.writerow([name, f"{s:.6f}"])

    # Save focus points CSV
    points_csv = out_dir / "focus_points.csv"
    with open(points_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "focus_score", "x_focus", "y_focus", "region_area"],
        )
        writer.writeheader()
        for r in point_rows:
            writer.writerow({
                "image": r["image"],
                "focus_score": f"{r['focus_score']:.6f}",
                "x_focus": f"{r['x_focus']:.2f}",
                "y_focus": f"{r['y_focus']:.2f}",
                "region_area": r["region_area"],
            })

    print("  Saved focus_scores to:", scores_csv)
    print("  Saved focus_points to:", points_csv)
    if score_rows:
        print("  Top 3 sharpest frames:")
        for name, s in score_rows[:3]:
            print("   ", name, "->", s)
    print()


def main():
    print("Reading dataset.csv from:", DATASET_CSV)
    if not DATASET_CSV.exists():
        print("ERROR: dataset.csv not found")
        return

    with open(DATASET_CSV, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    print("Total rows:", len(rows))
    # Here all are train, but we filter anyway for clarity
    train_rows = [r for r in rows if r.get("set") == "train"]
    print("Train rows:", len(train_rows))

    for r in train_rows:
        set_name = r["set"]
        lens = r["lens"]
        photo = r["photo"]
        process_stack(set_name, lens, photo)


if __name__ == "__main__":
    main()
