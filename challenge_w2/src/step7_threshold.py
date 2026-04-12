"""
Step 7 — Best Threshold Selection & Final Evaluation
Goal: Use the Youden Index to pick the optimal threshold, then evaluate
      with a full confusion matrix and classification report.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # challenge_w2/
FEATURES_DIR = os.path.join(ROOT_DIR, "data", "features")
ROC_DIR      = os.path.join(ROOT_DIR, "data", "roc")
PLOTS_DIR    = os.path.join(ROOT_DIR, "plots")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading saved ROC data and probability scores...")
proba_right = np.load(os.path.join(ROC_DIR, "proba_test.npy"))
y_test      = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))
fpr         = np.load(os.path.join(ROC_DIR, "roc_fpr.npy"))
tpr         = np.load(os.path.join(ROC_DIR, "roc_tpr.npy"))
thresholds  = np.load(os.path.join(ROC_DIR, "roc_thresholds.npy"))
auc         = roc_auc_score(y_test, proba_right)

# ── Youden Index ──────────────────────────────────────────────────────────────
# J(t) = TPR(t) - FPR(t)
# Maximizing J finds the threshold that best balances sensitivity and specificity.
# Note: thresholds has one fewer element than fpr/tpr (sklearn convention)
tpr_trimmed = tpr[1:]   # align lengths
fpr_trimmed = fpr[1:]

J        = tpr_trimmed - fpr_trimmed
best_idx = np.argmax(J)
best_threshold = thresholds[best_idx]
best_tpr = tpr_trimmed[best_idx]
best_fpr = fpr_trimmed[best_idx]
best_J   = J[best_idx]

print(f"\nYouden Index results:")
print(f"  Best threshold : {best_threshold:.4f}  (default is 0.5)")
print(f"  Youden J       : {best_J:.4f}")
print(f"  TPR at best    : {best_tpr:.4f}")
print(f"  FPR at best    : {best_fpr:.4f}")
print(f"  Specificity    : {1 - best_fpr:.4f}  (1 - FPR)")

# ── Apply threshold ───────────────────────────────────────────────────────────
y_pred_default = (proba_right >= 0.5).astype(int)
y_pred_best    = (proba_right >= best_threshold).astype(int)

# ── Confusion matrices ────────────────────────────────────────────────────────
cm_default = confusion_matrix(y_test, y_pred_default)
cm_best    = confusion_matrix(y_test, y_pred_best)

print("\n" + "="*55)
print("EVALUATION WITH DEFAULT THRESHOLD (0.5)")
print("="*55)
print(classification_report(y_test, y_pred_default,
                             target_names=["LEFT (0)", "RIGHT (1)"]))

print("="*55)
print(f"EVALUATION WITH YOUDEN THRESHOLD ({best_threshold:.3f})")
print("="*55)
print(classification_report(y_test, y_pred_best,
                             target_names=["LEFT (0)", "RIGHT (1)"]))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Step 7 — Threshold Selection & Final Evaluation", fontsize=12)

# ROC curve with both thresholds marked
axes[0].plot(fpr, tpr, color="darkorange", linewidth=2,
             label=f"ROC (AUC={auc:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
# Youden threshold
axes[0].scatter(best_fpr, best_tpr, color="red", zorder=5, s=100,
                label=f"Youden thresh={best_threshold:.2f}\n(TPR={best_tpr:.2f}, FPR={best_fpr:.2f})")
# Default 0.5 threshold — find nearest point
diff = np.abs(thresholds - 0.5)
idx_05 = np.argmin(diff)
axes[0].scatter(fpr_trimmed[idx_05], tpr_trimmed[idx_05], color="blue",
                zorder=5, s=80, marker="s",
                label=f"Default thresh=0.5\n(TPR={tpr_trimmed[idx_05]:.2f}, FPR={fpr_trimmed[idx_05]:.2f})")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve — Both Thresholds")
axes[0].legend(fontsize=7, loc="lower right")
axes[0].grid(alpha=0.3)

# Confusion matrix — default threshold
disp1 = ConfusionMatrixDisplay(cm_default, display_labels=["LEFT", "RIGHT"])
disp1.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix\nDefault threshold = 0.5")

# Confusion matrix — Youden threshold
disp2 = ConfusionMatrixDisplay(cm_best, display_labels=["LEFT", "RIGHT"])
disp2.plot(ax=axes[2], colorbar=False, cmap="Oranges")
axes[2].set_title(f"Confusion Matrix\nYouden threshold = {best_threshold:.3f}")

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "step7_threshold.png")
plt.savefig(out_path, dpi=100)
print(f"\nSaved final evaluation plot to {out_path}")
plt.show()

# ── Closing summary ───────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
tn0, fp0, fn0, tp0 = cm_default.ravel()
tn1, fp1, fn1, tp1 = cm_best.ravel()
print(f"  {'':30s}  {'Default (0.5)':>14}  {'Youden':>8}")
print(f"  {'True Positives (RIGHT→RIGHT)':30s}  {tp0:>14}  {tp1:>8}")
print(f"  {'False Positives (LEFT→RIGHT)':30s}  {fp0:>14}  {fp1:>8}")
print(f"  {'True Negatives (LEFT→LEFT)':30s}  {tn0:>14}  {tn1:>8}")
print(f"  {'False Negatives (RIGHT→LEFT)':30s}  {fn0:>14}  {fn1:>8}")
