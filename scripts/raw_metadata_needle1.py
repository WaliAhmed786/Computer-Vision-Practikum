import csv
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path.home() / "focuszoom"

RAW_DIR = PROJECT_ROOT / "data" / "dataset" / "train" / "olympus_macro_lens" / "needle1" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "metadata_needle1"
OUT_CSV = OUT_DIR / "raw_metadata_needle1.csv"


def exiftool_metadata(raw_path: Path):
    """
    Call exiftool on a single RAW file and return a dict with selected fields.
    Uses JSON output for easier parsing.
    """
    # Tags we care about (you can add more later)
    tags = [
        "Make",
        "Model",
        "LensModel",
        "FocalLength",
        "FNumber",
        "ISO",
        "ExposureTime",
        "FocusDistance",
    ]

    cmd = ["exiftool", "-j", "-n"] + [f"-{t}" for t in tags] + [str(raw_path)]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  exiftool error for {raw_path.name}: {e.stderr.strip()}")
        return None

    try:
        data_list = json.loads(result.stdout)
        if not data_list:
            return None
        data = data_list[0]  # exiftool -j returns a list with one dict per file
    except json.JSONDecodeError as e:
        print(f"  JSON parse error for {raw_path.name}: {e}")
        return None

    # Normalize keys we want
    md = {
        "raw_file": raw_path.name,
        "cam_make": data.get("Make"),
        "cam_model": data.get("Model"),
        "lens_model": data.get("LensModel"),
        "focal_length_mm": data.get("FocalLength"),
        "fnumber": data.get("FNumber"),
        "iso": data.get("ISO"),
        "exposure_time": data.get("ExposureTime"),
        "focus_distance": data.get("FocusDistance"),
    }
    return md


def main():
    print("RAW_DIR:", RAW_DIR)
    if not RAW_DIR.exists():
        print("ERROR: RAW_DIR does not exist")
        return

    raw_files = sorted(list(RAW_DIR.glob("*.rw2")) + list(RAW_DIR.glob("*.RW2")))
    print("Found", len(raw_files), "RAW files.")
    if not raw_files:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in raw_files:
        print("Reading:", p.name)
        md = exiftool_metadata(p)
        if md is None:
            print("  WARNING: no metadata for", p.name)
            continue
        rows.append(md)

    if not rows:
        print("No metadata extracted.")
        return

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Saved RAW metadata CSV to:", OUT_CSV)
    print("Example rows:")
    for r in rows[:5]:
        print(" ", r)


if __name__ == "__main__":
    main()

