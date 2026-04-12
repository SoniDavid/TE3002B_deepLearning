"""
Step 3 — Build Feature Matrix X and Label Vector y

Feature pipeline (position-invariant):
  1. Crop to YOLO bounding box → grayscale → resize 128×128 → normalize [0,1]
  2. Canny edge detection  — isolates the arrow's strong structure,
     suppresses weak background gradients from the ROI environment
  3. Column-pooled HOG  (pixels_per_cell=(128,8), cells_per_block=(1,2))
     — each HOG cell spans the full image height → invariant to vertical
       position of the arrow; 16 horizontal columns preserve LEFT/RIGHT
       asymmetry → 270 features (vs 8,100 before)

Augmentation (horizontal shift ±15 %):
  Each image generates 3 variants (original + shifted left + shifted right)
  so the model sees the arrow at different horizontal positions, making it
  robust to the arrow not being centred in the ROI.

Split: ALL images pooled (train + test + validate), random stratified
       train_test_split (80/20) — dataset folder structure is NOT used.
"""

import os
import numpy as np
from PIL import Image
from skimage.feature import hog, canny
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = os.path.join(ROOT_DIR, "archive")
OUT_DIR     = os.path.join(ROOT_DIR, "data", "features")
IMG_SIZE    = (128, 128)
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# Column-pooled HOG: each cell is 128px tall (full height) × 8px wide
# → 1×16 cell grid → invariant to vertical position, preserves left/right
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(128, 8),
    cells_per_block=(1, 2),
    visualize=False,
    channel_axis=None,
)

# Horizontal shifts applied to each image for augmentation
SHIFT_FRACS = [0.0, +0.15, -0.15]   # original, shifted right, shifted left
CANNY_SIGMA = 2.0                    # smoothing before edge detection

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


def preprocess(img_path: str, label_info: tuple) -> np.ndarray:
    """
    Crop to bbox → grayscale → resize 128×128 → normalize [0,1] → Canny edges.
    Returns a float32 edge map of shape (128, 128).
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
    gray = np.array(crop, dtype=np.float32) / 255.0

    # Canny edges: isolates the arrow's salient structure,
    # suppresses weak background gradients
    edges = canny(gray, sigma=CANNY_SIGMA).astype(np.float32)
    return edges


def shift_image(img: np.ndarray, shift_frac: float) -> np.ndarray:
    """
    Shift a 128×128 float32 array horizontally by shift_frac × width pixels.
    The vacated side is filled with zeros (no content).
    Positive shift_frac → content moves right (arrow appears more to the right).
    Negative shift_frac → content moves left.
    """
    if shift_frac == 0.0:
        return img
    px = int(abs(shift_frac) * img.shape[1])
    out = np.zeros_like(img)
    if shift_frac > 0:
        out[:, px:] = img[:, :img.shape[1] - px]
    else:
        out[:, :img.shape[1] - px] = img[:, px:]
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_splits = ["train", "test", "validate"]
    X_list, y_list = [], []
    total_skipped = 0

    for split in all_splits:
        img_dir = os.path.join(ARCHIVE_DIR, "images", split)
        lbl_dir = os.path.join(ARCHIVE_DIR, "labels", split)

        if not os.path.isdir(img_dir):
            print(f"  [{split}] directory not found, skipping.")
            continue

        filenames = sorted(os.listdir(img_dir))
        split_count = 0

        for fname in filenames:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            info = read_label_file(fname, lbl_dir)
            if info is None:
                total_skipped += 1
                continue

            edge_map = preprocess(os.path.join(img_dir, fname), info)

            # Generate augmented variants (original + horizontal shifts)
            for shift_frac in SHIFT_FRACS:
                shifted = shift_image(edge_map, shift_frac)
                feat = hog(shifted, **HOG_PARAMS)
                X_list.append(feat)
                y_list.append(info[0])

            split_count += 1
            if split_count % 500 == 0:
                print(f"  [{split}] processed {split_count} images so far...")

        print(f"  [{split}] Done — {split_count} images × {len(SHIFT_FRACS)} shifts.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\nTotal samples (images × augmentation): {len(X)}")
    print(f"Skipped (no label)                   : {total_skipped}")
    print(f"Feature vector size                  : {X.shape[1]}")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution (full augmented dataset):")
    for cls, cnt in zip(unique, counts):
        label = "LEFT" if cls == 0 else "RIGHT"
        print(f"  Class {cls} ({label}): {cnt} ({cnt/len(y)*100:.1f}%)")

    # Random stratified split — NOT the archive's folder split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print(f"\nStratified split (test_size={TEST_SIZE}, seed={RANDOM_SEED}):")
    print(f"  Train : {X_train.shape[0]} samples  {X_train.shape}")
    print(f"  Test  : {X_test.shape[0]} samples  {X_test.shape}")

    for split_name, y_split in [("Train", y_train), ("Test", y_test)]:
        u, c = np.unique(y_split, return_counts=True)
        parts = [f"class {cls}={'LEFT' if cls==0 else 'RIGHT'} {cnt} ({cnt/len(y_split)*100:.1f}%)"
                 for cls, cnt in zip(u, c)]
        print(f"  {split_name} balance: {', '.join(parts)}")

    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)

    print(f"\nAll arrays saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
