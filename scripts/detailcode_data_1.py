from pathlib import Path
import cv2
import rawpy
import numpy as np

DATA_DIR = Path.home() / "focuszoom" / "data" / "dataset" / "train" / "olympus_macro_lens" / "flower1" / "raw"

def main():
    print("=== Inspect Data Script Starting ===")
    print("Using DATA_DIR:", DATA_DIR)

    if not DATA_DIR.exists():
        print("ERROR: DATA_DIR does not exist!")
        return


    files = list(DATA_DIR.iterdir())
    print(f"Found {len(files)} files in DATA_DIR.")
    for f in files[:10]:
        print(" -", f.name)


    candidates = [f for f in files if f.suffix.lower() in [".dng", ".nef", ".cr2", ".arw", ".tif", ".tiff", ".rw2"]]
    print(f"Found {len(candidates)} RAW/TIFF candidates.")

    if not candidates:
        print("No RAW/TIFF files found in this folder. Nothing more to do.")
        return

    first = candidates[0]
    print("Trying to read:", first.name)

    outdir = Path.home() / "focuszoom" / "outputs" / "samples"
    outdir.mkdir(parents=True, exist_ok=True)

    if first.suffix.lower() in [".dng", ".nef", ".cr2", ".arw", ".rw2"]:
        print("Detected real RAW format, using rawpy...")
        with rawpy.imread(str(first)) as raw:
            rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True)
        rgb8 = (rgb16 / 256).astype(np.uint8)
    else:
        print("Detected TIFF-like file, using OpenCV...")
        bgr = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            print("ERROR: OpenCV could not read the file.")
            return
        if bgr.ndim == 2:
            rgb8 = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("Image shape:", rgb8.shape, "dtype:", rgb8.dtype, "min:", rgb8.min(), "max:", rgb8.max())

    out_path = outdir / "debug_preview_from_raw_or_tiff.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
    print("Saved preview to:", out_path)
    print("=== Done ===")

if __name__ == "__main__":
    main()

