import csv
from pathlib import Path

PROJECT_ROOT = Path.home() / "focuszoom"
DATASET_CSV = PROJECT_ROOT / "data" / "dataset" / "dataset.csv"


def main():
    print("Reading:", DATASET_CSV)
    if not DATASET_CSV.exists():
        print("ERROR: dataset.csv not found!")
        return


    with open(DATASET_CSV, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    print("Total rows in dataset.csv:", len(rows))
    if not rows:
        return

    print("Columns:", reader.fieldnames)

    print("\nFirst 5 rows:")
    for r in rows[:5]:
        print(" ", r)


    split_counts = {}
    for r in rows:
        split = r.get("set", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1

    print("\nSet counts:")
    for k, v in split_counts.items():
        print(" ", k, "->", v)

    print("\nChecking a few sample paths:")
    for r in rows[:10]:
        s = r["set"]
        lens = r["lens"]
        photo = r["photo"]

        base_dir = PROJECT_ROOT / "data" / "dataset" / s / lens / photo
        jpg_dir = base_dir / "jpg"
        raw_dir = base_dir / "raw"

        print(f"  {s} | {lens} | {photo}")
        print("    base_dir:", base_dir, "exists:", base_dir.exists())
        print("    jpg_dir :", jpg_dir, "exists:", jpg_dir.exists())
        print("    raw_dir :", raw_dir, "exists:", raw_dir.exists())


if __name__ == "__main__":
    main()

