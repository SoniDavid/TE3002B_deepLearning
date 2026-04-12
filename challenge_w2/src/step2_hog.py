"""
Step 2 — Extract HOG Features (visualization)
Goal: Visualize the full feature pipeline on one image per class:
        grayscale → Canny edges → column-pooled HOG

Column-pooled HOG (pixels_per_cell=(128,8)):
  Each cell spans the FULL image height → vertical-position invariant.
  16 horizontal columns capture the LEFT/RIGHT asymmetry of the arrow.
  → 270 features (vs 8,100 with the old 8×8 spatial grid).

Canny preprocessing:
  Isolates the arrow's strong structural edges, suppresses weak background
  gradients — essential when the ROI may contain non-arrow content.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog, canny

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = os.path.join(ROOT_DIR, "archive")
PLOTS_DIR   = os.path.join(ROOT_DIR, "plots")
IMG_SIZE    = (128, 128)
ALL_SPLITS  = ["train", "test", "validate"]
CANNY_SIGMA = 2.0

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(128, 8),   # full-height cells → vertical invariance
    cells_per_block=(1, 2),     # horizontal block normalization
    visualize=True,
    channel_axis=None,
)

CLASS_NAMES = {0: "LEFT", 1: "RIGHT"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_label_file(img_filename: str, label_dir: str):
    """Returns (class_id, cx, cy, bw, bh) or None if missing/empty."""
    stem = os.path.splitext(img_filename)[0]
    path = os.path.join(label_dir, stem + ".txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def preprocess(img_path: str, label_info: tuple):
    """
    Crop to bbox → grayscale → resize → normalize → Canny edges.
    Returns (gray, edges): both float32 [0,1] arrays of shape (128,128).
    """
    _, cx, cy, bw, bh = label_info
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))

    crop = img.crop((x1, y1, x2, y2)).convert("L")
    crop = crop.resize(IMG_SIZE, Image.BILINEAR)
    gray  = np.array(crop, dtype=np.float32) / 255.0
    edges = canny(gray, sigma=CANNY_SIGMA).astype(np.float32)
    return gray, edges


def find_first_of_class(target_class: int):
    """Returns (full_img_path, label_info) across all splits."""
    for split in ALL_SPLITS:
        img_dir = os.path.join(ARCHIVE_DIR, "images", split)
        lbl_dir = os.path.join(ARCHIVE_DIR, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            info = read_label_file(fname, lbl_dir)
            if info is not None and info[0] == target_class:
                return os.path.join(img_dir, fname), info
    return None, None

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle("Step 2 — Feature Pipeline: Grayscale → Canny → Column-pooled HOG", fontsize=12)

    for row, cls in enumerate([0, 1]):
        img_path, info = find_first_of_class(cls)
        if img_path is None:
            print(f"No image found for class {cls}")
            continue
        fname = os.path.basename(img_path)

        gray, edges = preprocess(img_path, info)

        # HOG on edge map (same as used in training)
        features, hog_image = hog(edges, **HOG_PARAMS)

        # Col 0: original grayscale
        axes[row, 0].imshow(gray, cmap="gray")
        axes[row, 0].set_title(f"Grayscale — {CLASS_NAMES[cls]}\n({fname})", fontsize=8)
        axes[row, 0].axis("off")

        # Col 1: Canny edge map
        axes[row, 1].imshow(edges, cmap="gray")
        axes[row, 1].set_title(f"Canny edges (σ={CANNY_SIGMA}) — {CLASS_NAMES[cls]}", fontsize=8)
        axes[row, 1].axis("off")

        # Col 2: column-pooled HOG visualization
        axes[row, 2].imshow(hog_image, cmap="magma", aspect="auto")
        axes[row, 2].set_title(f"HOG visualization — {CLASS_NAMES[cls]}", fontsize=8)
        axes[row, 2].axis("off")

        # Col 3: feature value histogram
        axes[row, 3].hist(features, bins=30, color="steelblue", edgecolor="white", linewidth=0.3)
        axes[row, 3].set_title(f"Feature distribution — {CLASS_NAMES[cls]}\n({len(features)} values)", fontsize=8)
        axes[row, 3].set_xlabel("Feature value")
        axes[row, 3].set_ylabel("Count")

        print(f"\nClass {cls} ({CLASS_NAMES[cls]})  —  {fname}")
        print(f"  Feature vector shape : {features.shape}")
        print(f"  Min / Max / Mean     : {features.min():.4f} / {features.max():.4f} / {features.mean():.4f}")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "step2_hog_viz.png")
    plt.savefig(out_path, dpi=100)
    print(f"\nSaved HOG visualization to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
