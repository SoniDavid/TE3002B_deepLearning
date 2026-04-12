"""
Step 1 — Load & Preprocess Images
Goal: Load images from ALL splits (train + test + validate), crop to bounding
      box, resize to 128x128, convert to grayscale, verify visually that
      cropping isolates the arrow correctly.
      No predefined dataset split is assumed here.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # challenge_w2/
ARCHIVE_DIR = os.path.join(ROOT_DIR, "archive")
PLOTS_DIR   = os.path.join(ROOT_DIR, "plots")
IMG_SIZE    = (128, 128)   # width x height after crop
CLASS_NAMES = {0: "LEFT", 1: "RIGHT"}
ALL_SPLITS  = ["train", "test", "validate"]

# ── Helper functions ───────────────────────────────────────────────────────────

def read_label_file(img_filename: str, label_dir: str):
    """
    Read the YOLO label file for an image.
    Returns (class_id, cx, cy, bw, bh) as (int, float, float, float, float),
    or None if the file is missing or empty.
    """
    stem = os.path.splitext(img_filename)[0]
    label_path = os.path.join(label_dir, stem + ".txt")
    if not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def preprocess(img_path: str, label_info: tuple, size: tuple[int, int]) -> np.ndarray:
    """
    Crop image to the bounding box, then resize to `size` and convert to grayscale.

    label_info: (class_id, cx, cy, bw, bh) — normalized YOLO coordinates.
    Returns a uint8 numpy array of shape (size[1], size[0]).
    """
    _, cx, cy, bw, bh = label_info
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Convert normalized bbox → pixel coordinates
    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))

    crop = img.crop((x1, y1, x2, y2))
    crop = crop.convert("L")                      # grayscale after crop
    crop = crop.resize(size, Image.BILINEAR)
    return np.array(crop)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Collect images from ALL splits — no predefined separation used
    samples = []
    for split in ALL_SPLITS:
        img_dir = os.path.join(ARCHIVE_DIR, "images", split)
        lbl_dir = os.path.join(ARCHIVE_DIR, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            info = read_label_file(fname, lbl_dir)
            if info is None:
                continue
            samples.append((os.path.join(img_dir, fname), info))

    print(f"Total images with labels: {len(samples)}")

    # Count per class
    counts = {0: 0, 1: 0}
    for _, info in samples:
        counts[info[0]] = counts.get(info[0], 0) + 1
    print(f"  Class 0 (LEFT) : {counts.get(0, 0)}")
    print(f"  Class 1 (RIGHT): {counts.get(1, 0)}")

    # ── Show crops side-by-side with the full image ────────────────────────────
    # Pick 5 of each class
    left_samples  = [(f, i) for f, i in samples if i[0] == 0][:5]
    right_samples = [(f, i) for f, i in samples if i[0] == 1][:5]
    display_samples = left_samples + right_samples   # 10 images

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Step 1 — Cropped & resized to 128×128 grayscale", fontsize=13)

    for ax, (img_path, info) in zip(axes.flat, display_samples):
        arr = preprocess(img_path, info, IMG_SIZE)
        label = info[0]
        bbox_area = info[3] * info[4] * 100   # bw*bh as % of image
        ax.imshow(arr, cmap="gray")
        ax.set_title(f"{CLASS_NAMES[label]}  (bbox {bbox_area:.0f}%)\n{os.path.basename(img_path)}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "step1_preview.png")
    plt.savefig(out_path, dpi=100)
    print(f"\nSaved preview to {out_path}")
    plt.show()

    # ── Sanity check: print one array ─────────────────────────────────────────
    first_path, first_info = samples[0]
    arr = preprocess(first_path, first_info, IMG_SIZE)
    print(f"\nSample image  : {os.path.basename(first_path)}  →  label={first_info[0]} ({CLASS_NAMES[first_info[0]]})")
    print(f"Bbox coverage : {first_info[3]*first_info[4]*100:.1f}% of original image")
    print(f"Array shape   : {arr.shape}")
    print(f"Dtype         : {arr.dtype}")
    print(f"Pixel range   : [{arr.min()}, {arr.max()}]")


if __name__ == "__main__":
    main()
