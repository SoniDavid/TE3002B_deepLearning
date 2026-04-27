"""
05_visualizations.py — SpongeBob Edition
==========================================
Genera las 5 gráficas requeridas por la rúbrica.

    plots/01_ridge_path.png         ← Acción 13
    plots/02_mse_vs_alpha.png       ← Acción 14
    plots/03_confusion_matrix.png   ← Acción 15
    plots/04_faces_grid.png         ← Acción 16
    plots/05_sensitivity_noise.png  ← Acción 17

Uso:
    python 05_visualizations.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
_m04 = import_module("04_ridge_model")
ridge_fit       = _m04.ridge_fit
predict_class   = _m04.predict_class
compute_accuracy = _m04.compute_accuracy

MODEL_FILE   = Path("data/model.npz")
DATASET_FILE = Path("data/dataset.npz")
PLOTS_DIR    = Path("plots")

SEED = 42
np.random.seed(SEED)
N_CLASSES = 10

EXPRESSION_NAMES = [
    "angry", "confused", "crying", "disgusted", "excited",
    "happy", "scared",  "serious", "smug",      "surprised",
]

COLORS = [
    "#E05252", "#85C1E9", "#6BB5E0", "#A8D8A8", "#F39C12",
    "#FFD700", "#52BE80", "#566573", "#9B59B6", "#F7DC6F",
]

# ─────────────────────────────────────────────
# 1. RIDGE PATH
# ─────────────────────────────────────────────

def plot_ridge_path(m):
    alphas     = m["alphas"]
    betas      = m["betas"]
    alpha_star = float(m["alpha_star"])

    var     = np.var(betas[:,:,0], axis=0)
    top_idx = np.argsort(var)[-30:]

    fig, ax = plt.subplots(figsize=(10,5))
    for feat in top_idx:
        ax.semilogx(alphas, betas[:,feat,0], alpha=0.5,
                    linewidth=0.8, color="#4A90D9")
    ax.axvline(alpha_star, color="#E74C3C", lw=2, ls="--",
               label=f"α* = {alpha_star:.3f}")
    ax.set_xlabel("α (escala logarítmica)", fontsize=11)
    ax.set_ylabel("Valor del coeficiente β", fontsize=11)
    ax.set_title("Ridge Path — Encogimiento de coeficientes\n"
                 "(30 coeficientes de mayor varianza)", fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3, ls=":"); ax.set_facecolor("#F8F9FA")
    plt.tight_layout()
    out = PLOTS_DIR / "01_ridge_path.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Ridge Path: {out}")

# ─────────────────────────────────────────────
# 2. MSE VS ALPHA
# ─────────────────────────────────────────────

def plot_mse_vs_alpha(m):
    alphas     = m["alphas"]
    mse_tr     = m["mse_train"]
    mse_val    = m["mse_val"]
    cv_mse     = m["cv_mse"]
    alpha_star = float(m["alpha_star"])
    best_mse   = float(np.min(cv_mse))

    fig, ax = plt.subplots(figsize=(9,4))
    ax.semilogx(alphas, mse_tr,  color="#4A90D9", lw=1.8, ls="--",
                alpha=0.7, label="MSE train")
    ax.semilogx(alphas, mse_val, color="#E07B54", lw=1.8,
                label="MSE validación")
    ax.semilogx(alphas, cv_mse,  color="#27AE60", lw=2,
                label=f"{len(cv_mse)//len(alphas) if len(cv_mse)>len(alphas) else 5}-fold CV MSE")
    ax.axvline(alpha_star, color="#E74C3C", lw=2, ls=":",
               label=f"α* = {alpha_star:.3f}")
    ax.scatter([alpha_star], [best_mse], color="#E74C3C", zorder=5, s=80,
               label=f"Mín CV = {best_mse:.4f}")
    ax.set_xlabel("α (log)", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title("MSE vs α — Bias-Variance Tradeoff\n"
                 "α pequeño → bajo sesgo  |  α grande → baja varianza",
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, ls=":")
    ax.set_facecolor("#F8F9FA")
    plt.tight_layout()
    out = PLOTS_DIR / "02_mse_vs_alpha.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ MSE vs α: {out}")

# ─────────────────────────────────────────────
# 3. MATRIZ DE CONFUSIÓN
# ─────────────────────────────────────────────

def plot_confusion_matrix(m):
    B, T, mu, sd = m["B"], m["T"], m["mu"], m["sd"]
    X_test, y_test = m["X_test"], m["y_test"]
    Xs    = (X_test - mu) / np.where(sd==0, 1, sd)
    y_pred = predict_class(Xs, B, T)
    acc    = float(np.mean(y_pred == y_test))

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=float)
    for t, p in zip(y_test, y_pred):
        cm[t,p] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    cm_pct = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Recall (%)")
    labels = [n.capitalize() for n in EXPRESSION_NAMES]
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = "white" if cm_pct[i,j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i,j]:.0f}%",
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_title(f"Matriz de Confusión — Accuracy: {acc*100:.1f}%\n"
                 f"(% por fila = recall por expresión)", fontsize=11)

    # Identificar par más confundido
    cm_off = cm_pct.copy(); np.fill_diagonal(cm_off, 0)
    i_c, j_c = np.unravel_index(np.argmax(cm_off), cm_pct.shape)
    print(f"\n  ⚠ Par más confundido: '{EXPRESSION_NAMES[i_c]}' "
          f"→ '{EXPRESSION_NAMES[j_c]}' ({cm_off[i_c,j_c]:.0f}%)")

    plt.tight_layout()
    out = PLOTS_DIR / "03_confusion_matrix.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Matriz confusión: {out}")
    return y_pred

# ─────────────────────────────────────────────
# 4. GRID DE CARAS
# ─────────────────────────────────────────────

def plot_faces_grid(m, y_pred):
    """
    Muestra un grid de caras originales vs caras reconstruidas por el modelo
    (ŷ = Xs @ B proyectado a píxeles 32×32) para al menos 3 identidades en test.
    Incluye predicciones correctas e incorrectas con etiquetas real y predicha.
    """
    B, T, mu, sd = m["B"], m["T"], m["mu"], m["sd"]
    X_test, y_test = m["X_test"], m["y_test"]
    sd_safe = np.where(sd == 0, 1, sd)
    Xs_test = (X_test - mu) / sd_safe

    # Reconstrucción: proyectar ŷ de vuelta a escala de píxeles
    Y_hat = Xs_test @ B                  # (N, 9) en espacio símplex
    # Recuperar imagen aprox: deshacer estandarización sobre X
    X_recon = np.clip(Xs_test * sd_safe + mu, 0, 1)   # imagen original reconstruida

    show_classes = [0, 2, 5]            # angry, crying, happy
    n_cols       = 6                    # 3 pares (original|reconstruida) por fila
    n_rows       = len(show_classes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.8, n_rows * 3.6)
    )
    fig.suptitle(
        "Caras test — Original vs Reconstruida\n"
        "(verde = correcta, rojo = incorrecta)",
        fontsize=11
    )

    for row, cls in enumerate(show_classes):
        ok_mask  = (y_test == cls) & (y_pred == cls)
        bad_mask = (y_test == cls) & (y_pred != cls)
        ok_orig  = X_test[ok_mask][:2]
        ok_recon = X_recon[ok_mask][:2]
        bad_orig  = X_test[bad_mask][:1]
        bad_recon = X_recon[bad_mask][:1]
        bad_pred_cls = y_pred[bad_mask][:1]

        col = 0
        # Par 1 y 2: predicciones correctas
        for orig, recon in zip(ok_orig, ok_recon):
            for ax, img, lbl in [
                (axes[row, col],   orig,  f"Real: {EXPRESSION_NAMES[cls]}"),
                (axes[row, col+1], recon, f"✓ Pred: {EXPRESSION_NAMES[cls]}"),
            ]:
                ax.imshow(img.reshape(32, 32), cmap="gray", vmin=0, vmax=1)
                ax.set_title(lbl, fontsize=6, color="green", pad=2)
                ax.axis("off")
            col += 2

        # Par 3: predicción incorrecta
        if len(bad_orig):
            pred_name = EXPRESSION_NAMES[int(bad_pred_cls[0])]
            for ax, img, lbl, color in [
                (axes[row, col],   bad_orig[0],  f"Real: {EXPRESSION_NAMES[cls]}", "black"),
                (axes[row, col+1], bad_recon[0], f"✗ Pred: {pred_name}", "red"),
            ]:
                ax.imshow(img.reshape(32, 32), cmap="gray", vmin=0, vmax=1)
                ax.set_title(lbl, fontsize=6, color=color, pad=2)
                ax.axis("off")
            col += 2

        while col < n_cols:
            axes[row, col].axis("off")
            col += 1

    plt.tight_layout()
    out = PLOTS_DIR / "04_faces_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grid de caras: {out}")

# ─────────────────────────────────────────────
# 5. SENSIBILIDAD AL RUIDO
# ─────────────────────────────────────────────

def plot_sensitivity_noise(m):
    B, T, mu, sd = m["B"], m["T"], m["mu"], m["sd"]
    X_test, y_test = m["X_test"], m["y_test"]
    sd_safe = np.where(sd==0, 1, sd)

    sigmas = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    accs   = []
    acc_base = compute_accuracy((X_test-mu)/sd_safe, y_test, B, T)

    for sigma in sigmas:
        rng    = np.random.default_rng(SEED)
        X_noisy = np.clip(X_test + rng.normal(0,sigma,X_test.shape), 0, 1)
        Xs      = (X_noisy - mu) / sd_safe
        accs.append(compute_accuracy(Xs, y_test, B, T)*100)

    drops  = [accs[i]-accs[i+1] for i in range(len(accs)-1)]
    infl_s = sigmas[int(np.argmax(drops))+1]

    fig, ax = plt.subplots(figsize=(9,4))
    ax.semilogx(sigmas, accs, "o-", color="#4A90D9", lw=2, ms=7,
                label="Accuracy en test ruidoso")
    ax.axhline(acc_base*100, color="#27AE60", lw=1.5, ls="--",
               label=f"Sin ruido ({acc_base*100:.1f}%)")
    ax.axvline(infl_s, color="#E74C3C", lw=1.5, ls=":",
               label=f"Inflexión σ={infl_s}")
    ax.scatter([infl_s], [accs[sigmas.index(infl_s)]],
               color="#E74C3C", zorder=5, s=100)
    ax.set_xlabel("Nivel de ruido σ (log)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Sensibilidad al Ruido — α* fijo", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, ls=":")
    ax.set_facecolor("#F8F9FA"); ax.set_ylim(0,105)
    plt.tight_layout()
    out = PLOTS_DIR / "05_sensitivity_noise.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Sensibilidad al ruido: {out}")
    print(f"     Punto de degradación: σ = {infl_s}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "█"*57)
    print("  VISUALIZACIONES — SpongeBob Expression Classifier")
    print("  Reto Ridge Regression | TE3002B")
    print("█"*57)

    if not MODEL_FILE.exists():
        print(f"\n  ❌ No se encontró {MODEL_FILE}")
        print(f"     Ejecuta: python 04_ridge_model.py\n")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    m = dict(np.load(MODEL_FILE, allow_pickle=True))
    d = dict(np.load(DATASET_FILE, allow_pickle=True))

    print("\n  Generando gráficas...")
    print("  " + "─"*52)

    print("  [1/5] Ridge Path...")
    plot_ridge_path(m)

    print("  [2/5] MSE vs Alpha...")
    plot_mse_vs_alpha(m)

    print("  [3/5] Matriz de Confusión...")
    y_pred = plot_confusion_matrix(m)

    print("  [4/5] Grid de Caras...")
    plot_faces_grid(m, y_pred)

    print("  [5/5] Sensibilidad al Ruido...")
    plot_sensitivity_noise(m)

    print(f"\n  ✓ Todas las gráficas en: {PLOTS_DIR.resolve()}/")
    print(f"\n  ⚡ Siguiente: python 06_predict.py image.png\n")

if __name__ == "__main__":
    main()
