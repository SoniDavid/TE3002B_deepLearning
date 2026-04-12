# TE3002B Deep Learning module

Repository for coursework and practical challenges in the TE3002B Intelligent Robotics Deep Learning module.

## Course Information

- Course: TE3002B - Intelligent Robotics (Deep Learning module)
- Professor: [Professor Name]
- Professor Number: [Professor Number]

### Students


| Name | Student ID |
|---|---|
| David Soni | [A01571777] |
| [Name] | [ID] |

## Featured Project: Week 2 Challenge

### Logistic Regression Arrow Classification

Goal: classify arrow direction as LEFT (`0`) or RIGHT (`1`) from annotated images.

Core approach:

- YOLO annotations are used to crop arrow regions of interest
- preprocessing with grayscale conversion + Canny edge extraction
- HOG feature encoding (column-pooled configuration)
- Logistic Regression for binary classification
- ROC/AUC analysis and decision-threshold tuning

Detailed documentation is available in:

- [challenge_w2/README.md](challenge_w2/README.md)
- [challenge_w2/docs/DOCUMENTATION.md](challenge_w2/docs/DOCUMENTATION.md)

## Week 2 Quick Start

### Clone Repository

```bash
git clone https://github.com/SoniDavid/TE3002B_deepLearning.git
cd TE3002B_deepLearning
```

### Install Dependencies

The project uses a Conda environments, so be sure to have it installed (or either mamba or miniforge)

Main dependencies:

- Python 3.11
- numpy
- matplotlib
- pillow
- joblib
- scikit-image
- scikit-learn
- opencv
- ipykernel

### Run Pipeline

```bash
cd challenge_w2
python src/step3_build_matrix.py
python src/step4_train.py
python src/step5_probabilities.py
python src/step6_roc.py
python src/step7_threshold.py
```



