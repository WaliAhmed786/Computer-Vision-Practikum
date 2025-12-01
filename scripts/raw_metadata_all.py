import csv
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path.home() / "focuszoom"
DATASET_CSV = PROJECT_ROOT / "data" / "dataset" / "dataset.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "metadata_all"
OUT_CSV = OUT_DIR / "raw_metadata_all.csv"


def exiftool_metadata(raw_path: Path):
    """
    Call exiftool on a single RAW file and return a dict with selected fields.
    Uses JSON output for easier parsing.
    """
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
        print(f"    exiftool error for {raw_path.name}: {e.stderr.strip()}")
        return None

    try:
        data_list = json.loads(result.stdout)
        if not data_list:
            return None
        data = data_list[0]
    except json.JSONDecodeError as e:
        print(f"    JSON parse error for {raw_path.name}: {e}")
        return None

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for r in rows:
        set_name = r["set"]
        lens = r["lens"]
        photo = r["photo"]

        base_dir = PROJECT_ROOT / "data" / "dataset" / set_name / lens / photo
        raw_dir = base_dir / "raw"

        if not raw_dir.exists():
            print(f"[SKIP] raw dir does not exist: {raw_dir}")
            continue

        raw_files = sorted(list(raw_dir.glob("*.rw2")) + list(raw_dir.glob("*.RW2")))
        if not raw_files:
            print(f"[SKIP] no RAW files in: {raw_dir}")
            continue

        print(f"Processing stack: set={set_name}, lens={lens}, photo={photo}")
        print(f"  RAW files: {len(raw_files)}")

        for p in raw_files:
            print(f"    Reading: {p.name}")
            md = exiftool_metadata(p)
            if md is None:
                print("      WARNING: no metadata returned")
                continue

            # add context (which stack this RAW belongs to)
            md["set"] = set_name
            md["lens"] = lens
            md["photo"] = photo

            all_rows.append(md)

    if not all_rows:
        print("No metadata extracted for any stack.")
        return

    # Use keys from first row as CSV columns + make sure context columns are present
    fieldnames = [
        "set",
        "lens",
        "photo",
        "raw_file",
        "cam_make",
        "cam_model",
        "lens_model",
        "focal_length_mm",
        "fnumber",
        "iso",
        "exposure_time",
        "focus_distance",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print("Saved RAW metadata for ALL stacks to:", OUT_CSV)
    print("Total RAW rows:", len(all_rows))
    print("Example rows:")
    for r in all_rows[:5]:
        print(" ", r)


if __name__ == "__main__":
    main()
