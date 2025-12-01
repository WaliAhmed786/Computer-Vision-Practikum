from pathlib import Path
import rawpy
import cv2
import numpy as np


DATA_DIR = Path.home() / "focuszoom" / "data" / "dataset" / "train" / "olympus_macro_lens" / "needle1" / "raw"

def main():
    print("DATA_DIR:", DATA_DIR)
    if not DATA_DIR.exists():
        print("ERROR: Folder does not exist.")
        return

    # list all files in this folder
    files = list(DATA_DIR.iterdir())
    print("Total files in folder:", len(files))

    # keep only RAW-like extensions (including .rw2)
    exts = [".rw2", ".dng", ".nef", ".cr2", ".arw", ".tif", ".tiff"]
    raws = [f for f in files if f.suffix.lower() in exts]

    print("RAW candidates found:", len(raws))
    if not raws:
        print("No RAW/TIFF files found.")
        return

    first = raws[0]
    print("Reading file:", first.name)

    # read RAW with rawpy
    with rawpy.imread(str(first)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            output_bps=16,
            no_auto_bright=True
        )
    rgb8 = (rgb16 / 256).astype(np.uint8)

    print("Image shape:", rgb8.shape, "dtype:", rgb8.dtype)

    outdir = Path.home() / "focuszoom" / "outputs" / "samples"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"preview_{first.stem}.jpg"

    cv2.imwrite(str(out_path), cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
    print("Saved preview to:", out_path)

if __name__ == "__main__":
    main()

