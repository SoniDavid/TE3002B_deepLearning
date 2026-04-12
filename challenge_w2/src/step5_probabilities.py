"""
Step 5 — Probability Scores on Test Set
Goal: Use predict_proba to get P(RIGHT | x) for every test image.
      Inspect the distribution to confirm the model discriminates between classes.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # challenge_w2/
FEATURES_DIR = os.path.join(ROOT_DIR, "data", "features")
MODEL_DIR    = os.path.join(ROOT_DIR, "data", "model")
ROC_DIR      = os.path.join(ROOT_DIR, "data", "roc")
PLOTS_DIR    = os.path.join(ROOT_DIR, "plots")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading model, scaler, and test data...")
model  = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

print(f"  X_test : {X_test.shape}")
print(f"  y_test : {y_test.shape}  (class 0=LEFT, 1=RIGHT)")

# ── Scale (using TRAIN scaler — never refit on test) ──────────────────────────
X_test_scaled = scaler.transform(X_test)

# ── Predict probabilities ─────────────────────────────────────────────────────
# predict_proba returns [[P(LEFT), P(RIGHT)], ...]
# We take column 1: P(RIGHT | x)
proba_all   = model.predict_proba(X_test_scaled)
proba_right = proba_all[:, 1]   # probability of being class 1 (RIGHT)

# Save for Step 6
np.save(os.path.join(ROC_DIR, "proba_test.npy"), proba_right)
print(f"\nSaved probability scores to {ROC_DIR}/proba_test.npy  shape={proba_right.shape}")

# ── Print sample rows ─────────────────────────────────────────────────────────
print("\nSample predictions (first 15):")
print(f"  {'Index':>6}  {'True label':>12}  {'P(RIGHT)':>10}  {'Verdict'}")
print("  " + "-" * 45)
for i in range(15):
    true = "RIGHT" if y_test[i] == 1 else "LEFT"
    prob = proba_right[i]
    verdict = "✓" if (prob >= 0.5) == (y_test[i] == 1) else "✗"
    print(f"  {i:>6}  {true:>12}  {prob:>10.4f}  {verdict}")

# ── Statistics ────────────────────────────────────────────────────────────────
left_probs  = proba_right[y_test == 0]
right_probs = proba_right[y_test == 1]

print(f"\nP(RIGHT) statistics:")
print(f"  True LEFT  samples — mean={left_probs.mean():.4f}  std={left_probs.std():.4f}")
print(f"  True RIGHT samples — mean={right_probs.mean():.4f}  std={right_probs.std():.4f}")
print()
print("Interpretation:")
print("  LEFT  samples should cluster near 0  (model confident they are NOT right)")
print("  RIGHT samples should cluster near 1  (model confident they ARE right)")
print("  Overlap in the middle = uncertainty zone")

# ── Plot: histogram of P(RIGHT) per class ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Step 5 — Distribution of P(RIGHT | x) on test set", fontsize=12)

# Overlapping histogram
axes[0].hist(left_probs,  bins=30, alpha=0.6, color="steelblue",  label="True LEFT (0)",  density=True)
axes[0].hist(right_probs, bins=30, alpha=0.6, color="tomato",     label="True RIGHT (1)", density=True)
axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold=0.5")
axes[0].set_xlabel("P(RIGHT | x)")
axes[0].set_ylabel("Density")
axes[0].set_title("Overlapping distributions")
axes[0].legend()

# Stacked side-by-side
axes[1].boxplot([left_probs, right_probs], tick_labels=["True LEFT", "True RIGHT"],
                patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.5))
axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1, label="threshold=0.5")
axes[1].set_ylabel("P(RIGHT | x)")
axes[1].set_title("Boxplot per class")
axes[1].legend()

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "step5_probabilities.png")
plt.savefig(out_path, dpi=100)
print(f"Saved probability plot to {out_path}")
plt.show()
