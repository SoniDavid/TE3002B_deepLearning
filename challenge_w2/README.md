# Week 2 Challenge: Logistic Regression Arrow Classification

Binary image classification project for left/right arrow direction using classical computer vision + machine learning.

## Objective

Train a model that classifies arrow images into:

- `0` = LEFT
- `1` = RIGHT

The pipeline uses:

- YOLO-format labels only to crop the arrow ROI
- grayscale + Canny edge preprocessing
- column-pooled HOG features
- Logistic Regression for classification
- ROC/AUC analysis and threshold tuning (Youden index)

## Dataset Source

This repository does not include the full image dataset. Download it from Kaggle:

- https://www.kaggle.com/datasets/pranavunni/left-and-right-arrows-dataset-with-annotations

After downloading/unzipping, place the dataset under `challenge_w2/archive/`.

Expected local layout:

```
challenge_w2/
└── archive/
	├── images/
	│   ├── train/
	│   ├── test/
	│   └── validate/
	└── labels/
		├── train/
		├── test/
		└── validate/
```

Each image should have a corresponding YOLO label file with the same stem name:

- `images/train/1234.jpg` <-> `labels/train/1234.txt`

## Project Structure

```
challenge_w2/
├── data/
│   ├── features/          # X_train/X_test/y_train/y_test .npy
│   ├── model/             # model.pkl + scaler.pkl
│   └── roc/               # probabilities + ROC arrays
├── docs/
│   └── DOCUMENTATION.md   # extended technical explanation
├── plots/                 # generated figures from each step
├── src/
│   ├── step1_load.py
│   ├── step2_hog.py
│   ├── step3_build_matrix.py
│   ├── step4_train.py
│   ├── step5_probabilities.py
│   ├── step6_roc.py
│   ├── step7_threshold.py
│   ├── predict.py
│   └── realtime_predict.py
└── environment.yml
```

Note: `archive/` is expected locally after download and is not required to be versioned in this repository.

## Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ml_week2
```

Main dependencies include Python 3.11, NumPy, scikit-learn, scikit-image, OpenCV, Matplotlib, Pillow, and joblib.

## Full Pipeline (Run Order)

Run scripts from the `challenge_w2` root:

```bash
python src/step1_load.py
python src/step2_hog.py
python src/step3_build_matrix.py
python src/step4_train.py
python src/step5_probabilities.py
python src/step6_roc.py
python src/step7_threshold.py
```

## How Images and YOLO Labels Are Used

Each label file follows YOLO format (single object per image in this dataset):

```text
class_id cx cy bw bh
```

Where:

- `class_id`: `0` = LEFT, `1` = RIGHT
- `cx`, `cy`: normalized bbox center coordinates
- `bw`, `bh`: normalized bbox width and height

The scripts convert normalized coordinates to pixel coordinates:

```text
x1 = (cx - bw/2) * image_width
y1 = (cy - bh/2) * image_height
x2 = (cx + bw/2) * image_width
y2 = (cy + bh/2) * image_height
```

Then the pipeline:

1. Crops the image to the YOLO bounding box (arrow ROI).
2. Converts ROI to grayscale and resizes to 128x128.
3. Applies Canny edge detection.
4. Extracts HOG features.
5. Trains/evaluates Logistic Regression.

This means labels are used for two things:

- Class target (`0`/`1`) for supervision.
- Bounding box crop so features focus on the arrow rather than background.

### What each step does

1. `step1_load.py`
- Loads images across all splits.
- Reads YOLO labels and crops bounding boxes.
- Converts to grayscale, resizes to 128x128.
- Saves preview plot in `plots/step1_preview.png`.

2. `step2_hog.py`
- Demonstrates feature pipeline visually (grayscale -> Canny -> HOG).
- Uses column-pooled HOG configuration.
- Saves `plots/step2_hog_viz.png`.

3. `step3_build_matrix.py`
- Builds feature matrix with augmentation.
- Pipeline: crop -> grayscale -> normalize -> Canny -> HOG.
- Uses horizontal shift augmentation (`0`, `+15%`, `-15%`).
- Performs stratified random 80/20 split.
- Saves:
	- `data/features/X_train.npy`
	- `data/features/X_test.npy`
	- `data/features/y_train.npy`
	- `data/features/y_test.npy`

4. `step4_train.py`
- Standardizes features using `StandardScaler` (fit on train only).
- Trains Logistic Regression (`max_iter=1000`).
- Reports training metrics.
- Saves model artifacts:
	- `data/model/model.pkl`
	- `data/model/scaler.pkl`
- Saves coefficient visualization to `plots/step4_coefficients.png`.

5. `step5_probabilities.py`
- Loads model and test set.
- Computes `P(RIGHT | x)` with `predict_proba`.
- Saves `data/roc/proba_test.npy`.
- Saves `plots/step5_probabilities.png`.

6. `step6_roc.py`
- Builds ROC points and computes AUC (manual + sklearn cross-check).
- Saves:
	- `data/roc/roc_fpr.npy`
	- `data/roc/roc_tpr.npy`
	- `data/roc/roc_thresholds.npy`
- Saves `plots/step6_roc.png`.

7. `step7_threshold.py`
- Selects best threshold using Youden index $J = TPR - FPR$.
- Compares default threshold (`0.5`) vs tuned threshold.
- Prints classification reports + confusion matrices.
- Saves `plots/step7_threshold.png`.

## Inference Scripts

### Single image or folder prediction

```bash
python src/predict.py path/to/image.jpg
python src/predict.py path/to/image.jpg path/to/label.txt
python src/predict.py archive/images/test/
```

Notes:
- If a label path is provided, prediction uses the YOLO bbox crop.
- Folder mode auto-tries matching labels by replacing `images` -> `labels` in path.

### Real-time webcam prediction

```bash
python src/realtime_predict.py
```

Keyboard controls:

- `Q` / `ESC`: quit
- `R`: reset temporal memory (EMA)
- `M` / `m`: increase/decrease memory strength
- `+` / `-`: resize ROI box
- Arrow keys: move ROI box

## Key Methodology Notes

- This project does not use the dataset folder split directly for training/testing.
- In `step3_build_matrix.py`, all available images are pooled and then split randomly with stratification.
- HOG is configured to be vertically invariant via full-height cells and to preserve horizontal asymmetry for left/right discrimination.
- Canny preprocessing reduces noisy gradients from non-arrow background inside the ROI.

## Outputs Checklist

After running all steps, you should have:

- Feature arrays in `data/features/`
- Trained model/scaler in `data/model/`
- ROC arrays in `data/roc/`
- Diagnostic plots in `plots/`

## Extended Documentation

For deeper explanation of HOG, ROC table construction, AUC computation, and interpretation, see:

- `docs/DOCUMENTATION.md`

