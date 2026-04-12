"""
Step 4 — Train Logistic Regression
Goal: Fit logistic regression on HOG features, check training performance,
      save model + scaler for later steps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # challenge_w2/
FEATURES_DIR = os.path.join(ROOT_DIR, "data", "features")
MODEL_DIR    = os.path.join(ROOT_DIR, "data", "model")
PLOTS_DIR    = os.path.join(ROOT_DIR, "plots")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading feature matrices...")
X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))

print(f"  X_train : {X_train.shape}  (samples × features)")
print(f"  y_train : {y_train.shape}")

# ── Standardize ───────────────────────────────────────────────────────────────
# IMPORTANT: fit the scaler ONLY on train data, never on test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"\nAfter scaling:")
print(f"  Feature mean ≈ {X_train_scaled.mean():.4f}  (should be ~0)")
print(f"  Feature std  ≈ {X_train_scaled.std():.4f}   (should be ~1)")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
print("  Done.")

# ── Train accuracy ────────────────────────────────────────────────────────────
y_pred_train = model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, y_pred_train)
print(f"\nTrain accuracy: {train_acc*100:.2f}%")
print("\nTrain classification report:")
print(classification_report(y_train, y_pred_train, target_names=["LEFT (0)", "RIGHT (1)"]))

# ── Most influential features ─────────────────────────────────────────────────
# Logistic regression learns one coefficient per feature.
# Large |coeff| = high influence on the decision boundary.
coef = model.coef_[0]   # shape: (n_features,)
n_top = 10
top_idx    = np.argsort(np.abs(coef))[::-1][:n_top]
top_values = coef[top_idx]

print(f"Top {n_top} most influential HOG features (by |coefficient|):")
for rank, (idx, val) in enumerate(zip(top_idx, top_values), 1):
    direction = "→ RIGHT" if val > 0 else "← LEFT"
    print(f"  #{rank:2d}  feature[{idx:4d}]  coeff={val:+.4f}  {direction}")

# ── Plot: coefficient distribution ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Step 4 — Logistic Regression Coefficients", fontsize=12)

axes[0].hist(coef, bins=60, color="steelblue", edgecolor="white", linewidth=0.2)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1, label="0")
axes[0].set_title("Distribution of all coefficients")
axes[0].set_xlabel("Coefficient value")
axes[0].set_ylabel("Count")
axes[0].legend()

axes[1].barh(range(n_top), top_values[::-1],
             color=["tomato" if v > 0 else "steelblue" for v in top_values[::-1]])
axes[1].set_yticks(range(n_top))
axes[1].set_yticklabels([f"feat[{i}]" for i in top_idx[::-1]], fontsize=8)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_title(f"Top {n_top} features (red=RIGHT, blue=LEFT)")
axes[1].set_xlabel("Coefficient value")

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "step4_coefficients.png")
plt.savefig(out_path, dpi=100)
print(f"\nSaved coefficient plot to {out_path}")
plt.show()

# ── Save model + scaler ───────────────────────────────────────────────────────
joblib.dump(model,  os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"Saved model.pkl and scaler.pkl to {MODEL_DIR}")
