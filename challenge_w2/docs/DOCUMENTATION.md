# Arrow Classification — Full Documentation

**Course:** IRS 6to — Machine Learning, Week 2  
**Topic:** Binary Classification with Logistic Regression and ROC Analysis  
**Task:** Classify images as LEFT arrow (class 0) or RIGHT arrow (class 1)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [The Full Pipeline](#3-the-full-pipeline)
4. [Step 1 — Image Preprocessing and Bounding Box Crop](#4-step-1--image-preprocessing-and-bounding-box-crop)
5. [Step 2 — HOG Feature Extraction](#5-step-2--hog-feature-extraction)
6. [Step 3 — Feature Matrix](#6-step-3--feature-matrix)
7. [Step 4 — Logistic Regression Training](#7-step-4--logistic-regression-training)
8. [Step 5 — Probability Scores](#8-step-5--probability-scores)
9. [Step 6 — ROC Curve and AUC](#9-step-6--roc-curve-and-auc)
10. [Step 7 — Threshold Selection (Youden Index)](#10-step-7--threshold-selection-youden-index)
11. [Real-Time Inference with OpenCV](#11-real-time-inference-with-opencv)
12. [What the Model Actually Learned](#12-what-the-model-actually-learned)
13. [File Reference](#13-file-reference)

---

## 1. Project Overview

The goal is to build a machine learning system that looks at an image containing an arrow sign and classifies it as pointing **LEFT** or **RIGHT**.

This is a **binary classification** problem. The approach does not use deep learning or object detection. Instead it relies on two classical computer vision / machine learning techniques:

- **HOG (Histogram of Oriented Gradients)** — converts an image into a compact numerical description of its edge structure
- **Logistic Regression** — learns which edge patterns are associated with each class and outputs a probability

The model is then evaluated using the **ROC curve** and **AUC**, which measure how well the model separates the two classes across all possible decision thresholds.

---

## 2. Dataset

The dataset is formatted in **YOLO v5 style**, which is an object detection format. Even though we are not doing object detection, the bounding box annotations are very useful — they tell us exactly where the arrow is in each image.

### Directory Structure

```
archive/
├── images/
│   ├── train/       2,673 images
│   ├── test/          402 images
│   └── validate/      136 images
└── labels/
    ├── train/       one .txt file per image
    ├── test/
    └── validate/
```

### Label File Format

Each `.txt` file contains one line per annotated object:

```
class_id   cx     cy     width  height
    0      0.570  0.333  0.858  0.658
```

All five values are **normalized to [0, 1]** relative to the image dimensions:
- `class_id` — 0 = LEFT, 1 = RIGHT
- `cx, cy` — center of the bounding box as fractions of image width/height
- `width, height` — size of the bounding box as fractions of image width/height

### Class Distribution

| Split    | LEFT (0) | RIGHT (1) | Total |
|----------|----------|-----------|-------|
| Train    | 1,243    | 1,402     | 2,645 |
| Test     | 201      | 201       | 402   |
| Validate | 64       | 68        | 132   |

The classes are **nearly balanced**, which means accuracy is a reliable metric here (no class imbalance problem). The test set is perfectly balanced (50/50).

### Why the Background is a Problem

74% of images have the arrow occupying **less than 30% of the frame**. The rest is background — hands, walls, ceiling, room furniture. If we feed the full image to the model, most of the gradient information (HOG features) will describe the background rather than the arrow, and the model will learn irrelevant patterns.

The solution is to crop each image to its bounding box before any further processing.

---

## 3. The Full Pipeline

```
Raw image (e.g. 640×480 RGB)
        │
        ▼
┌─────────────────────────────┐
│  Read YOLO label file       │  → get bounding box coordinates
│  Convert to pixel coords    │  → x1, y1, x2, y2
│  Crop image to bbox         │  → arrow-only region
│  Convert to grayscale       │  → remove color, keep shape
│  Resize to 128×128          │  → uniform input size
│  Normalize pixels to [0,1]  │  → float32 array
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  HOG Feature Extraction     │  → 8,100 gradient histogram values
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  StandardScaler             │  → zero mean, unit variance
│  (fit only on train data)   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Logistic Regression        │  → P(RIGHT | x) ∈ [0, 1]
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Threshold (default 0.5)    │  → class prediction: LEFT or RIGHT
│  or Youden optimal thresh   │
└─────────────────────────────┘
        │
        ▼
    Prediction + Confidence
```

---

## 4. Step 1 — Image Preprocessing and Bounding Box Crop

### Reading the Bounding Box

The YOLO coordinates are normalized, so we convert them to pixel coordinates:

```
x1 = (cx - width/2)  × image_width
y1 = (cy - height/2) × image_height
x2 = (cx + width/2)  × image_width
y2 = (cy + height/2) × image_height
```

### Preprocessing Steps

1. **Open as RGB** — PIL loads the image in full color
2. **Crop to bounding box** — `img.crop((x1, y1, x2, y2))` isolates the arrow
3. **Convert to grayscale** — color is not useful for classifying arrow direction; shape is. Grayscale reduces the data from 3 channels to 1
4. **Resize to 128×128** — ensures every image produces a feature vector of the same length regardless of original size
5. **Normalize to [0, 1]** — divide pixel values by 255 so all values are in the same range

### Why 128×128?

This size is a trade-off:
- Large enough to preserve the arrow's edge structure clearly
- Small enough to keep the HOG feature vector manageable (~8,100 values)
- Larger sizes (e.g. 256×256) would give ~32,000 features with diminishing returns

---

## 5. Step 2 — HOG Feature Extraction

### What HOG Measures

HOG (Histogram of Oriented Gradients) was introduced by Dalal and Triggs in 2005 for pedestrian detection. It describes an image by the **distribution of local edge directions**.

The key insight: the shape of an object can be described well by the local intensity gradients (edges) and their directions, regardless of exact color or illumination.

### How HOG Works (Step by Step)

**Step 1 — Compute gradients**  
For each pixel, compute the intensity change in the horizontal (Gx) and vertical (Gy) directions using derivative filters:

```
Gx = pixel(x+1, y) - pixel(x-1, y)
Gy = pixel(x, y+1) - pixel(x, y-1)
```

From these, compute the gradient magnitude and direction at each pixel:

```
magnitude = sqrt(Gx² + Gy²)
angle     = arctan(Gy / Gx)   →  mapped to 0°–180°
```

**Step 2 — Divide into cells**  
The 128×128 image is divided into a grid of small cells, each 8×8 pixels. This gives a 16×16 grid of cells.

**Step 3 — Build a histogram per cell**  
For each cell, accumulate the gradient magnitudes into 9 orientation bins (0°, 20°, 40°, ..., 160°). Each pixel votes for its orientation bin weighted by its magnitude.

The result is a 9-value histogram per cell that describes which edge directions are dominant in that region.

**Step 4 — Normalize across blocks**  
Cells are grouped into overlapping blocks of 2×2 cells (so 4 cells, 36 values per block). Each block is L2-normalized. This makes the descriptor robust to local lighting changes.

**Step 5 — Concatenate into one vector**  
All block descriptors are concatenated into a single flat vector.

### Feature Vector Size

With a 128×128 image and our parameters:
- Number of blocks = (16-1) × (16-1) = 225
- Values per block = 2 × 2 cells × 9 orientations = 36
- **Total = 225 × 36 = 8,100 values**

### Why HOG Works for Arrows

A LEFT arrow and a RIGHT arrow are mirror images. Their edge structures are symmetric but flipped:

```
LEFT arrow edges:          RIGHT arrow edges:
  \   /                        \   /
   \ /   → tip points left      \ /   → tip points right
   /|\                          /|\
  / | \                        / | \
```

HOG captures these directional patterns spatially. The gradient histograms on the left side of a LEFT arrow look completely different from those on the left side of a RIGHT arrow. This asymmetry is exactly what logistic regression learns to distinguish.

### HOG Parameters Used

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `orientations` | 9 | Number of angle bins (0°–180°, 20° each) |
| `pixels_per_cell` | (8, 8) | Each cell covers 8×8 pixels |
| `cells_per_block` | (2, 2) | Normalization block = 2×2 cells |

---

## 6. Step 3 — Feature Matrix

After processing all images, the dataset becomes two arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `X_train` | (2645, 8100) | One HOG vector per training image |
| `y_train` | (2645,) | Class labels (0=LEFT, 1=RIGHT) |
| `X_test` | (402, 8100) | One HOG vector per test image |
| `y_test` | (402,) | True class labels for test set |

At this point, **the model has no idea these came from images**. It simply sees a matrix of numbers. The semantic meaning (arrow direction) is implicitly encoded in the structure of those numbers.

---

## 7. Step 4 — Logistic Regression Training

### Standardization

Before training, features are standardized using `StandardScaler`:

```
x_scaled = (x - mean) / std
```

where `mean` and `std` are computed from the **training set only**, then applied to both train and test. This is critical — computing statistics from the test set would be data leakage.

Why standardize? Logistic regression uses gradient descent to optimize weights. If features have very different scales or variances, some dimensions dominate the optimization and convergence is slow or biased.

### How Logistic Regression Works

Logistic regression learns one **weight** per feature (8,100 weights total) plus one **bias** term. Given a feature vector x, it computes:

```
z = w₁·x₁ + w₂·x₂ + ... + w₈₁₀₀·x₈₁₀₀ + bias
```

This linear combination is passed through the **sigmoid function**:

```
P(RIGHT | x) = σ(z) = 1 / (1 + e⁻ᶻ)
```

The sigmoid maps any real number to a value between 0 and 1, which we interpret as a probability.

### Training Objective

During training, the model finds the weights that minimize the **binary cross-entropy loss** over all training samples:

```
Loss = -1/N × Σ [ yᵢ · log(p̂ᵢ) + (1 - yᵢ) · log(1 - p̂ᵢ) ]
```

where `yᵢ` is the true label and `p̂ᵢ` is the predicted probability. This loss is minimized using an optimization algorithm (by default, sklearn uses L-BFGS).

### What Each Weight Means

- A weight with a **large positive value** → that HOG feature is strongly associated with RIGHT arrows
- A weight with a **large negative value** → that HOG feature is strongly associated with LEFT arrows
- A weight **near zero** → that feature is not informative

After training, the top influential features (highest `|weight|`) correspond to HOG cells in the spatial regions where left and right arrows differ most — typically around the tip and the tail of the arrow.

### Training Results

| Metric | Value |
|--------|-------|
| Train accuracy | 100% |
| Feature vector size | 8,100 |
| Training samples | 2,645 |

The 100% train accuracy means the training data is **linearly separable** in the 8,100-dimensional HOG feature space. The bounding box crop was essential for this — without it, background noise would blur the separation.

---

## 8. Step 5 — Probability Scores

The model does not directly output a class label. It outputs a **probability**:

```python
proba = model.predict_proba(X)   # shape: (n_samples, 2)
# proba[:, 0] = P(LEFT  | x)
# proba[:, 1] = P(RIGHT | x)
```

These two values always sum to 1. `proba[:, 1]` is the score we use for ROC analysis — a number between 0 and 1 expressing how confident the model is that the image shows a RIGHT arrow.

### Ideal Distribution

On a well-trained model, the probability scores should form two non-overlapping distributions:

```
Density
  │     TRUE LEFT              TRUE RIGHT
  │     ████                   ████
  │    ██████                 ██████
  │   ████████               ████████
  └───────────────────────────────────────── P(RIGHT)
  0.0                 0.5                 1.0
```

In our case, the distributions are nearly perfectly separated: true LEFT images score near 0, true RIGHT images score near 1, with almost no overlap.

---

## 9. Step 6 — ROC Curve and AUC

### Why Not Just Use Accuracy?

Accuracy requires choosing a fixed threshold before evaluation. If the threshold is suboptimal, accuracy can be misleading. ROC analysis evaluates the model **at every possible threshold simultaneously**.

### Building the ROC Curve

We sweep the threshold `t` from 0 to 1. At each value of `t`:
- Predict RIGHT if `P(RIGHT | x) > t`, else predict LEFT
- Compute two rates from the resulting predictions:

```
TPR (True Positive Rate)  = TP / (TP + FN)
    "Of all actual RIGHT images, what fraction did we correctly predict as RIGHT?"
    Also called: Recall, Sensitivity

FPR (False Positive Rate) = FP / (FP + TN)
    "Of all actual LEFT images, what fraction did we wrongly predict as RIGHT?"
    Also called: 1 - Specificity
```

Plotting (FPR, TPR) for all thresholds gives the ROC curve.

### Reading the ROC Curve

```
TPR
1.0 │         ●────────────────────
    │        /
    │       /   ← model ROC curve
    │      /
    │     /  ← diagonal = random guessing
0.0 │────────────────────────────────
    0.0                            1.0  FPR
```

- **Top-left corner (FPR=0, TPR=1)** = perfect classifier: catches all positives with no false alarms
- **Diagonal line** = random guessing: no discriminative power
- The more the curve bows toward the top-left, the better the model

### AUC — Area Under the Curve

AUC summarizes the entire ROC curve as a single number (the area under it).

**Probabilistic interpretation:** AUC = the probability that a randomly chosen RIGHT image scores higher than a randomly chosen LEFT image.

| AUC | Interpretation |
|-----|---------------|
| 1.00 | Perfect classifier |
| 0.90–0.99 | Excellent |
| 0.80–0.89 | Good |
| 0.70–0.79 | Acceptable |
| 0.60–0.69 | Poor |
| 0.50 | Random guessing |
| < 0.50 | Worse than random |

**Our result: AUC = 1.000** — the two classes are perfectly separable in HOG feature space.

---

## 10. Step 7 — Threshold Selection (Youden Index)

### The Trade-off

Every threshold creates a trade-off:

| Threshold | Effect |
|-----------|--------|
| Low (e.g. 0.1) | Predict RIGHT very aggressively → high TPR, but also high FPR |
| High (e.g. 0.9) | Only predict RIGHT when very confident → low FPR, but miss many true RIGHTs |

### Youden Index

The Youden Index selects the threshold that maximizes the combined benefit:

```
J(t) = TPR(t) - FPR(t)
```

Geometrically, this is the threshold at the point on the ROC curve that is **farthest from the random diagonal**. It treats both types of error (false positives and false negatives) as equally costly.

```
best_threshold = argmax_t [ TPR(t) - FPR(t) ]
```

For our model, the default threshold of 0.5 already achieves perfect results because the probability scores are so well separated.

### Confusion Matrix

At the chosen threshold, we compute the confusion matrix:

```
                   Predicted LEFT    Predicted RIGHT
  Actual LEFT          TN                 FP
  Actual RIGHT         FN                 TP
```

From these four cells:
- **Precision** = TP / (TP + FP) — of all predicted RIGHTs, how many are actually RIGHT
- **Recall** = TP / (TP + FN) — of all actual RIGHTs, how many did we catch
- **F1 score** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean

---

## 11. Real-Time Inference with OpenCV

### The Challenge

During training, we knew exactly where each arrow was (from the YOLO label files). In real-time, there are no label files — we only have a live camera frame.

### Solution: Fixed ROI (Region of Interest)

A green rectangle is drawn on the live frame. The user places the arrow inside this rectangle. Every frame, we:

1. Crop the frame to the ROI rectangle
2. Run the crop through the full preprocessing pipeline (grayscale → resize → normalize)
3. Extract HOG features
4. Scale using the saved `scaler.pkl`
5. Run `model.predict_proba()` using the saved `model.pkl`
6. Display the result overlaid on the frame

### Latency

The entire inference pipeline takes approximately 2–5 ms per frame on a modern CPU:
- HOG extraction: ~1–3 ms
- StandardScaler transform: < 1 ms
- Logistic regression prediction: < 1 ms

This is well within the budget of a 30fps camera (33 ms per frame).

### Confidence Threshold

A display confidence threshold of 60% is used: if `max(P(LEFT), P(RIGHT)) < 0.6`, the label shows "uncertain" in gray instead of committing to a class.

---

## 12. What the Model Actually Learned

### The Model Has No Concept of "Arrow"

This is a crucial point. The model never learned what an arrow is. It learned:

> "Images where the 8,100-dimensional HOG vector has these specific statistical properties tend to have label 1 (RIGHT). Images with these other properties tend to have label 0 (LEFT)."

The model knows nothing about:
- The fact that the input is an image
- What an arrow looks like
- Spatial structure beyond what is encoded in the feature vector
- Why one arrow points left and another points right

### What It Does Know

Through training, the model learned that certain **gradient histogram values in certain spatial positions** are discriminative. Specifically:

- HOG cells on the **right side** of a RIGHT arrow contain strong **horizontal rightward gradients** (the tip)
- The same cells in a LEFT arrow contain strong **horizontal leftward gradients**
- The tail region shows the mirror pattern

The weights for these spatially informative HOG bins are large. Weights for HOG bins in uninformative regions (the center of the shaft, for example) are near zero.

### Why the Bounding Box Crop Was Essential

Without the crop:
- HOG would describe background content (walls, hands, ceiling)
- Many features would carry zero signal about arrow direction
- The decision boundary in 8,100-dimensional space would need to ignore most dimensions
- Training accuracy would be much lower, and AUC would likely fall below 0.9

With the crop:
- Every pixel in the feature vector belongs to the arrow
- Almost all 8,100 features carry useful signal
- The classes become **linearly separable** in HOG feature space
- AUC reaches 1.000

---

## 13. File Reference

### Scripts

| File | Purpose |
|------|---------|
| `step1_load.py` | Load images, apply bbox crop, display preprocessed samples |
| `step2_hog.py` | Visualize HOG features on one image per class |
| `step3_build_matrix.py` | Process all images → `X_train.npy`, `X_test.npy`, etc. |
| `step4_train.py` | Train logistic regression, save model |
| `step5_probabilities.py` | Compute and visualize probability scores on test set |
| `step6_roc.py` | Compute ROC curve, compute AUC, save plot |
| `step7_threshold.py` | Youden threshold, confusion matrix, final evaluation |
| `predict.py` | Run inference on any image or folder from command line |
| `realtime_predict.py` | Live camera inference using OpenCV fixed-ROI approach |
| `roc_logistic_regression.ipynb` | Full pipeline in a single Jupyter notebook with explanations |

### Saved Artifacts

| File | Contents | Required for inference? |
|------|----------|------------------------|
| `model.pkl` | Logistic regression weights (8,100 + bias) | Yes |
| `scaler.pkl` | StandardScaler mean and std per feature | Yes |
| `X_train.npy` | HOG feature matrix for all train images | No (only for retraining) |
| `y_train.npy` | Train labels | No |
| `X_test.npy` | HOG feature matrix for all test images | No |
| `y_test.npy` | Test labels | No |
| `proba_test.npy` | Saved probability scores on test set | No |
| `roc_fpr/tpr/thresholds.npy` | Saved ROC curve data | No |

### Environments

| Environment | Tools |
|-------------|-------|
| `ml_week2` | scikit-learn, scikit-image, matplotlib, numpy, Pillow, joblib |
| `opencv-env` | All of the above + OpenCV 4.12 (for real-time script) |

### How to Run

```bash
# Training pipeline (steps 1–7)
mamba_on && conda activate ml_week2
python step1_load.py
python step2_hog.py
python step3_build_matrix.py   # ~1 minute
python step4_train.py
python step5_probabilities.py
python step6_roc.py
python step7_threshold.py

# Inference on a single image
python predict.py archive/images/test/1641.jpg archive/labels/test/1641.txt

# Inference on a folder
python predict.py archive/images/test/

# Real-time inference
mamba_on && conda activate opencv-env
python realtime_predict.py
```
