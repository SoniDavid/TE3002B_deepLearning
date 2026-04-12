"""
realtime_predict.py — Real-time arrow classification using webcam.

Strategy (Option A — Fixed ROI):
  A green rectangle is drawn on the live camera feed.
  Whatever is inside that rectangle is cropped, preprocessed (grayscale →
  resize to 128x128), and fed through HOG + Logistic Regression every frame.
  The prediction and confidence are overlaid on the frame.

Temporal memory (EMA):
  Raw per-frame probabilities are smoothed with an exponential moving average:
      smoothed_p = α × p_current + (1−α) × p_previous
  Low α (MEMORY_ALPHA) → more memory, more stable; high α → faster reaction.
  The decision and displayed bars both use the smoothed probabilities.

Controls:
  Q or ESC  — quit
  R         — reset temporal memory (restart EMA from 0.5 / 0.5)
  +/-       — grow / shrink the ROI rectangle
  Arrow keys — move the ROI rectangle
  M / m     — increase / decrease memory strength (adjust α live)

Run with:
    conda activate opencv-env
    python realtime_predict.py
"""

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, canny

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # challenge_w2/
MODEL_DIR   = os.path.join(ROOT_DIR, "data", "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

IMG_SIZE    = (128, 128)
CANNY_SIGMA = 2.0

# Column-pooled HOG: must match step3_build_matrix.py exactly
HOG_PARAMS  = dict(
    orientations=9,
    pixels_per_cell=(128, 8),
    cells_per_block=(1, 2),
    visualize=False,
    channel_axis=None,
)

CLASS_NAMES   = {0: "LEFT", 1: "RIGHT"}
CLASS_COLORS  = {0: (235, 120, 50), 1: (50, 200, 100)}   # BGR: orange, green
CONFIDENCE_THRESHOLD = 0.6   # below this, show "uncertain" instead

# ── Temporal memory ───────────────────────────────────────────────────────────
# EMA: smoothed_p = ALPHA * p_raw + (1 - ALPHA) * smoothed_p_prev
# Lower ALPHA → more memory (more stable, slower to react)
# Higher ALPHA → less memory (faster, more twitchy)
MEMORY_ALPHA     = 0.15          # default blending factor
MEMORY_ALPHA_MIN = 0.05          # slowest (longest memory)
MEMORY_ALPHA_MAX = 0.80          # fastest (almost no memory)
MEMORY_ALPHA_STEP = 0.05         # step when pressing M / m

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model loaded.")

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr):
    """
    Convert an OpenCV BGR crop to a HOG feature vector.
      1. BGR → grayscale
      2. Resize to 128×128
      3. Normalize to [0, 1]
      4. Canny edge detection  — isolates arrow structure, ignores background
      5. Column-pooled HOG     — vertically invariant, 270 features
    Pipeline must match step3_build_matrix.py exactly.
    """
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    norm    = resized.astype(np.float32) / 255.0
    edges   = canny(norm, sigma=CANNY_SIGMA).astype(np.float32)
    return hog(edges, **HOG_PARAMS)


def predict_crop(crop_bgr):
    """
    Run the full inference pipeline on a BGR crop.
    Returns (class_id, confidence, p_left, p_right).
    """
    features = preprocess_crop(crop_bgr).reshape(1, -1)
    features = scaler.transform(features)
    proba    = model.predict_proba(features)[0]   # [P(LEFT), P(RIGHT)]
    pred     = int(np.argmax(proba))
    return pred, float(proba[pred]), float(proba[0]), float(proba[1])

# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_roi(frame, x1, y1, x2, y2, color=(0, 220, 0), thickness=2):
    """Draw the ROI rectangle with corner accents."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    corner_len = 20
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)


def draw_label(frame, text, x, y, color, font_scale=1.0, thickness=2):
    """Draw text with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_probability_bars(frame, p_left, p_right, p_left_raw, p_right_raw, x, y):
    """
    Draw horizontal bars for smoothed P(LEFT) and P(RIGHT).
    A thin white tick marks the raw (unsmoothed) probability for comparison.
    """
    bar_w = 160
    bar_h = 14
    labels = [("LEFT",  p_left,  p_left_raw,  (235, 120, 50)),
              ("RIGHT", p_right, p_right_raw, (50,  200, 100))]

    for i, (name, prob, raw, color) in enumerate(labels):
        bar_y = y + i * (bar_h + 8)
        # Background
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Smoothed bar (filled)
        fill = int(prob * bar_w)
        cv2.rectangle(frame, (x, bar_y), (x + fill, bar_y + bar_h), color, -1)
        # Border
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        # Raw probability tick (white vertical line)
        raw_x = x + int(raw * bar_w)
        cv2.line(frame, (raw_x, bar_y), (raw_x, bar_y + bar_h), (255, 255, 255), 2)
        # Label
        cv2.putText(frame, f"{name} {prob*100:.0f}%",
                    (x + bar_w + 6, bar_y + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    # Force display variable (needed when running from some terminals)
    import os
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    print("Opening camera 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Trying index 1...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No camera found. Check that no other app is using it.")
        return

    # Warm up — discard first few frames (some cameras take a moment to adjust)
    for _ in range(3):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read from camera.")
        cap.release()
        return

    fh, fw = frame.shape[:2]
    print(f"Camera open: {fw}x{fh}. Window should appear now. Press Q to quit.")

    # Create and raise window before the loop
    WINDOW_NAME = "Arrow Classifier — Real Time"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, fw, fh)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)  # flush so the window actually appears

    # Initial ROI — centered square, 40% of frame height
    roi_size = int(fh * 0.40)
    cx, cy   = fw // 2, fh // 2
    step     = 10    # pixels per keypress

    # Temporal memory — EMA state
    alpha        = MEMORY_ALPHA
    ema_p_left   = 0.5   # smoothed P(LEFT)
    ema_p_right  = 0.5   # smoothed P(RIGHT)

    print(f"Frame size: {fw}×{fh}")
    print("Controls: Q/ESC=quit  R=reset memory  +/-=resize ROI  "
          "arrow keys=move ROI  M/m=more/less memory")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Clamp ROI to frame boundaries
        x1 = max(0,  cx - roi_size // 2)
        y1 = max(0,  cy - roi_size // 2)
        x2 = min(fw, cx + roi_size // 2)
        y2 = min(fh, cy + roi_size // 2)

        # ── Inference ─────────────────────────────────────────────────────────
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            _, _, p_left_raw, p_right_raw = predict_crop(crop)
        else:
            p_left_raw, p_right_raw = 0.5, 0.5

        # ── Temporal memory — EMA update ──────────────────────────────────────
        ema_p_left  = alpha * p_left_raw  + (1 - alpha) * ema_p_left
        ema_p_right = alpha * p_right_raw + (1 - alpha) * ema_p_right

        # Re-normalise (floating point drift guard)
        total = ema_p_left + ema_p_right
        ema_p_left  /= total
        ema_p_right /= total

        # Decision from smoothed probabilities
        pred = 0 if ema_p_left >= ema_p_right else 1
        conf = ema_p_left if pred == 0 else ema_p_right

        # ── Draw ROI ──────────────────────────────────────────────────────────
        roi_color = CLASS_COLORS[pred] if conf >= CONFIDENCE_THRESHOLD else (180, 180, 180)
        draw_roi(frame, x1, y1, x2, y2, color=roi_color)

        # ── Draw prediction label ─────────────────────────────────────────────
        if conf >= CONFIDENCE_THRESHOLD:
            label_text = f"{CLASS_NAMES[pred]}  {conf*100:.0f}%"
            label_color = CLASS_COLORS[pred]
        else:
            label_text  = f"uncertain  {conf*100:.0f}%"
            label_color = (180, 180, 180)

        draw_label(frame, label_text, x1, y1 - 10, label_color, font_scale=0.9)

        # ── Draw probability bars (smoothed + raw tick) ───────────────────────
        draw_probability_bars(frame, ema_p_left, ema_p_right,
                              p_left_raw, p_right_raw, x=10, y=fh - 60)

        # ── Draw memory indicator ─────────────────────────────────────────────
        mem_label = f"memory  α={alpha:.2f}  (M/m to adjust, R to reset)"
        cv2.putText(frame, mem_label, (10, fh - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 255), 1, cv2.LINE_AA)

        # ── Draw instructions ─────────────────────────────────────────────────
        instructions = [
            "Q / ESC : quit",
            "R       : reset memory",
            "M / m   : more / less memory",
            "+  /  - : resize ROI",
            "Arrows  : move ROI",
        ]
        for i, txt in enumerate(instructions):
            cv2.putText(frame, txt, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):   # Q or ESC
            break
        elif key == ord("r") or key == ord("R"):   # reset EMA memory
            ema_p_left  = 0.5
            ema_p_right = 0.5
            print("Memory reset.")
        elif key == ord("M"):   # increase α → less memory (faster)
            alpha = min(MEMORY_ALPHA_MAX, round(alpha + MEMORY_ALPHA_STEP, 2))
            print(f"α = {alpha:.2f}  (less memory)")
        elif key == ord("m"):   # decrease α → more memory (slower)
            alpha = max(MEMORY_ALPHA_MIN, round(alpha - MEMORY_ALPHA_STEP, 2))
            print(f"α = {alpha:.2f}  (more memory)")
        elif key == ord("+") or key == ord("="):
            roi_size = min(min(fw, fh) - 20, roi_size + step * 2)
        elif key == ord("-"):
            roi_size = max(40, roi_size - step * 2)
        elif key == 82:   # up arrow
            cy = max(roi_size // 2, cy - step)
        elif key == 84:   # down arrow
            cy = min(fh - roi_size // 2, cy + step)
        elif key == 81:   # left arrow
            cx = max(roi_size // 2, cx - step)
        elif key == 83:   # right arrow
            cx = min(fw - roi_size // 2, cx + step)

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()
