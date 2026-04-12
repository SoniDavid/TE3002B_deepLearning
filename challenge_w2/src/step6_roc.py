"""
Step 6 — ROC Curve + AUC
Goal: Sweep thresholds over P(RIGHT) scores, compute TPR/FPR at each,
      build the full ROC table (as shown in the professor's TablaROC reference),
      compute AUC by the trapezoidal rule, and plot the curve.

Methodology follows:
  TablaROC-ClasificaciónBinaria-RegresiónLogística.pdf (Prof. Alberto Muñoz)
  - At each threshold t: count TP, FP, TN, FN
  - Derive TPR = TP/P, FPR = FP/N, Precision = TP/(TP+FP), NPV = TN/(TN+FN)
  - AUC = Σ (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2  (trapezoid rule)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # challenge_w2/
FEATURES_DIR = os.path.join(ROOT_DIR, "data", "features")
ROC_DIR      = os.path.join(ROOT_DIR, "data", "roc")
PLOTS_DIR    = os.path.join(ROOT_DIR, "plots")

os.makedirs(ROC_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading probability scores and test labels...")
proba_right = np.load(os.path.join(ROC_DIR, "proba_test.npy"))
y_test      = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

P = int((y_test == 1).sum())   # total positives  (RIGHT)
N = int((y_test == 0).sum())   # total negatives  (LEFT)

print(f"  proba_test : {proba_right.shape}")
print(f"  y_test     : {y_test.shape}   (P={P} RIGHT, N={N} LEFT)")


# ── Manual ROC table (professor's methodology) ────────────────────────────────

def compute_roc_table(y_true: np.ndarray, scores: np.ndarray):
    """
    Build the ROC table by sweeping every unique score as a threshold.

    Returns a list of dicts, one row per threshold, sorted from t=+∞ → t=-∞:
        t, TP, FP, TN, FN, TPR, FPR, Precision, NPV
    """
    # Origin point: t just above the maximum score — nothing classified positive
    rows = [dict(t=">max", TP=0, FP=0, TN=N, FN=P,
                 TPR=0.0, FPR=0.0, Precision=0.0, NPV=P/(P+N) if (P+N) > 0 else 0.0)]

    # All unique scores, sorted highest → lowest
    thresholds = sorted(np.unique(scores), reverse=True)

    for t in thresholds:
        pred     = (scores >= t).astype(int)
        TP = int(((pred == 1) & (y_true == 1)).sum())
        FP = int(((pred == 1) & (y_true == 0)).sum())
        TN = int(((pred == 0) & (y_true == 0)).sum())
        FN = int(((pred == 0) & (y_true == 1)).sum())

        TPR       = TP / P if P > 0 else 0.0
        FPR       = FP / N if N > 0 else 0.0
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        NPV       = TN / (TN + FN) if (TN + FN) > 0 else 0.0

        rows.append(dict(t=f"{t:.4f}", TP=TP, FP=FP, TN=TN, FN=FN,
                         TPR=TPR, FPR=FPR, Precision=Precision, NPV=NPV))

    return rows


print("\nBuilding ROC table (manual threshold sweep)...")
roc_table = compute_roc_table(y_test, proba_right)

# ── Print table ───────────────────────────────────────────────────────────────
header = (f"{'t':>8} | {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} | "
          f"{'TPR':>6} {'FPR':>6} {'Precision':>9} {'NPV':>6}")
sep = "-" * len(header)

print(f"\n{sep}")
print(header)
print(sep)

# Print first 5 rows, last 5 rows, and a mid-section summary to keep output readable
n_rows = len(roc_table)
show_indices = set(range(min(5, n_rows))) | set(range(max(0, n_rows-5), n_rows))
prev_skipped = False

for i, row in enumerate(roc_table):
    if i not in show_indices:
        if not prev_skipped:
            print(f"{'  ... (' + str(n_rows-10) + ' more rows) ...':^{len(header)}}")
            prev_skipped = True
        continue
    prev_skipped = False
    print(f"{row['t']:>8} | {row['TP']:>5} {row['FP']:>5} {row['TN']:>5} {row['FN']:>5} | "
          f"{row['TPR']:>6.3f} {row['FPR']:>6.3f} {row['Precision']:>9.3f} {row['NPV']:>6.3f}")

print(sep)
print(f"  P (total RIGHT) = {P}   N (total LEFT) = {N}   Total = {P+N}")


# ── Manual AUC (trapezoidal rule) ─────────────────────────────────────────────
# Collect (FPR, TPR) pairs from the table, sort by FPR to form the curve
fpr_manual = np.array([r['FPR'] for r in roc_table])
tpr_manual = np.array([r['TPR'] for r in roc_table])

# Sort by FPR ascending (needed for trapezoid)
order = np.argsort(fpr_manual)
fpr_sorted = fpr_manual[order]
tpr_sorted = tpr_manual[order]

auc_manual = float(np.trapezoid(tpr_sorted, fpr_sorted))


# ── sklearn ROC (for cross-check and plot) ────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, proba_right, pos_label=1)
auc_sklearn = roc_auc_score(y_test, proba_right)

# Save for Step 7
np.save(os.path.join(ROC_DIR, "roc_fpr.npy"),        fpr)
np.save(os.path.join(ROC_DIR, "roc_tpr.npy"),        tpr)
np.save(os.path.join(ROC_DIR, "roc_thresholds.npy"), thresholds)

print(f"\nAUC (trapezoidal rule, manual) : {auc_manual:.4f}")
print(f"AUC (sklearn roc_auc_score)    : {auc_sklearn:.4f}")
print(f"Difference                     : {abs(auc_manual - auc_sklearn):.6f}")
print()

if auc_sklearn >= 0.9:
    print("Interpretation: Excellent — model separates LEFT/RIGHT very well.")
elif auc_sklearn >= 0.8:
    print("Interpretation: Good — solid performance.")
elif auc_sklearn >= 0.7:
    print("Interpretation: Acceptable — some discriminative power.")
elif auc_sklearn >= 0.6:
    print("Interpretation: Poor — weak discrimination.")
else:
    print("Interpretation: Fail — model is no better than random guessing.")

# ── Find point closest to (0,1) ───────────────────────────────────────────────
distances = np.sqrt(fpr**2 + (1 - tpr)**2)
best_idx   = np.argmin(distances)
best_fpr   = fpr[best_idx]
best_tpr   = tpr[best_idx]
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

print(f"\nPoint closest to (0,1) — near-optimal operating point:")
print(f"  Threshold   : {best_thresh:.4f}")
print(f"  TPR (Recall): {best_tpr:.4f}")
print(f"  FPR         : {best_fpr:.4f}")
print(f"  Specificity : {1 - best_fpr:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 6 — ROC Curve", fontsize=12)

# Full ROC curve (sklearn points = smooth curve; manual points = discrete table)
ax = axes[0]
ax.plot(fpr, tpr, color="darkorange", linewidth=2,
        label=f"ROC curve (AUC = {auc_sklearn:.3f})")
ax.scatter(fpr_manual, tpr_manual, color="steelblue", s=10, alpha=0.5, zorder=3,
           label=f"Manual table points (AUC_trap = {auc_manual:.3f})")
ax.plot([0, 1], [0, 1], color="navy", linewidth=1,
        linestyle="--", label="Random baseline (AUC = 0.5)")
ax.scatter(best_fpr, best_tpr, color="red", zorder=5, s=80,
           label=f"Closest to (0,1)\nthresh={best_thresh:.2f}")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("ROC Curve")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

# TPR and FPR as a function of threshold
ax2 = axes[1]
t = thresholds
ax2.plot(t, tpr[1:] if len(tpr) > len(t) else tpr[:len(t)],
         color="tomato",    linewidth=1.5, label="TPR (Recall)")
ax2.plot(t, fpr[1:] if len(fpr) > len(t) else fpr[:len(t)],
         color="steelblue", linewidth=1.5, label="FPR")
ax2.axvline(best_thresh, color="gray", linestyle="--", linewidth=1,
            label=f"Best threshold ≈ {best_thresh:.2f}")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Rate")
ax2.set_title("TPR and FPR vs Threshold")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "step6_roc.png")
plt.savefig(out_path, dpi=100)
print(f"\nSaved ROC plot to {out_path}")
plt.show()
