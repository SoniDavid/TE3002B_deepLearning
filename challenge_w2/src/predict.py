"""
predict.py — Run the trained model on one or more images.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg path/to/label.txt   # also show true label
    python predict.py archive/images/test/                   # run on a whole folder
"""

import os
import sys
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog, canny

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # challenge_w2/
MODEL_DIR   = os.path.join(ROOT_DIR, "data", "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
IMG_SIZE    = (128, 128)
THRESHOLD   = 0.5
CANNY_SIGMA = 2.0

# Column-pooled HOG: must match step3_build_matrix.py exactly
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(128, 8),
    cells_per_block=(1, 2),
    visualize=False,
    channel_axis=None,
)

# ── Load model once ───────────────────────────────────────────────────────────
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_bbox(label_path: str):
    """Read YOLO bbox from a label file. Returns (cx, cy, bw, bh) or None."""
    if not label_path or not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    return float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def preprocess(img_path: str, bbox=None) -> np.ndarray:
    """
    Crop (optional) → grayscale → resize → normalize → Canny edges.
    Pipeline must match step3_build_matrix.py exactly.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    if bbox is not None:
        cx, cy, bw, bh = bbox
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        img = img.crop((x1, y1, x2, y2))

    img  = img.convert("L").resize(IMG_SIZE, Image.BILINEAR)
    gray = np.array(img, dtype=np.float32) / 255.0
    return canny(gray, sigma=CANNY_SIGMA).astype(np.float32)


def predict(img_path: str, label_path: str = None):
    """
    Predict the class of a single image.
    Returns a dict with keys: prediction, confidence, label_name.
    """
    bbox = read_bbox(label_path)
    img_array = preprocess(img_path, bbox)
    features  = hog(img_array, **HOG_PARAMS).reshape(1, -1)
    features  = scaler.transform(features)

    proba = model.predict_proba(features)[0]   # [P(LEFT), P(RIGHT)]
    pred  = int(model.predict(features)[0])

    return {
        "prediction":  pred,
        "label_name":  "RIGHT" if pred == 1 else "LEFT",
        "confidence":  proba[pred],
        "p_left":      proba[0],
        "p_right":     proba[1],
        "used_bbox":   bbox is not None,
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def run_single(img_path: str, label_path: str = None):
    result = predict(img_path, label_path)
    bbox_note = "with bbox crop" if result["used_bbox"] else "full image (no bbox)"

    print(f"\nImage     : {img_path}  [{bbox_note}]")
    print(f"Prediction: {result['label_name']}  (class {result['prediction']})")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"  P(LEFT)  = {result['p_left']:.4f}")
    print(f"  P(RIGHT) = {result['p_right']:.4f}")


def run_folder(folder_path: str):
    """Run on all images in a folder, look for matching label files automatically."""
    # Try to find a sibling labels/ folder
    labels_dir = folder_path.replace("images", "labels")

    files = sorted(f for f in os.listdir(folder_path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))

    correct, total = 0, 0
    print(f"\nRunning on {len(files)} images in {folder_path}\n")
    print(f"  {'File':<30}  {'Pred':>6}  {'Conf':>7}  {'True':>6}  {'OK?'}")
    print("  " + "-" * 62)

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(labels_dir, stem + ".txt")
        lbl_path = lbl_path if os.path.exists(lbl_path) else None

        result = predict(img_path, lbl_path)

        # Read true label if available
        true_str = "?"
        ok_str = ""
        if lbl_path:
            with open(lbl_path) as f:
                true_cls = int(f.readline().strip().split()[0])
            true_str = "RIGHT" if true_cls == 1 else "LEFT"
            ok = result["prediction"] == true_cls
            ok_str = "✓" if ok else "✗"
            correct += int(ok)
            total += 1

        print(f"  {fname:<30}  {result['label_name']:>6}  {result['confidence']*100:>6.1f}%"
              f"  {true_str:>6}  {ok_str}")

    if total > 0:
        print(f"\nAccuracy on {total} labeled images: {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py image.jpg")
        print("  python predict.py image.jpg label.txt   # provide bbox label")
        print("  python predict.py archive/images/test/  # run on folder")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        run_folder(target)
    else:
        label_path = sys.argv[2] if len(sys.argv) > 2 else None
        run_single(target, label_path)
