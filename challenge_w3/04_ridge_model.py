"""
04_ridge_model.py — SpongeBob Edition
=======================================
Ridge Regression manual para clasificación de expresiones de SpongeBob.
Sin sklearn.Ridge — implementación directa con np.linalg.solve.

Uso:
    python 04_ridge_model.py
"""

import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

SEED         = 42
np.random.seed(SEED)

DATASET_FILE = Path("data/dataset.npz")
MODEL_FILE   = Path("data/model.npz")

N_CLASSES  = 5
N_ALPHAS   = 40
ALPHA_MIN  = 1e-4
ALPHA_MAX = 1e1   # máximo 1.0
N_CV_FOLDS = 5

EXPRESSION_NAMES = [
    "happy", "angry", "crying", "surprised", "confused"
]

# ─────────────────────────────────────────────
# SÍMPLEX REGULAR EN R^9
# ─────────────────────────────────────────────

def build_simplex_vertices(n_classes: int) -> np.ndarray:
    """
    Construye vértices de un símplex regular en R^(n-1).
    Garantiza distancia euclídea igual entre todos los pares.
    Ventaja sobre one-hot: máximo balance entre clases.
    """
    dim = n_classes - 1
    T   = np.zeros((n_classes, dim))
    for i in range(n_classes):
        for j in range(dim):
            if j < i:
                T[i,j] = -1.0 / np.sqrt((n_classes-j)*(n_classes-j-1))
            elif j == i:
                T[i,j] = np.sqrt((n_classes-j-1)/(n_classes-j))

    dists = [np.linalg.norm(T[i]-T[j])
             for i in range(n_classes)
             for j in range(i+1, n_classes)]
    dists = np.array(dists)
    assert np.allclose(dists, dists[0], atol=1e-8), "Símplex NO equidistante"
    print(f"  ✓ Símplex R^{dim}: dist entre vértices = {dists[0]:.4f}")
    return T

# ─────────────────────────────────────────────
# PARTICIÓN ESTRATIFICADA
# ─────────────────────────────────────────────

def train_test_split(X, y, test_ratio=0.20, seed=SEED):
    """Partición 80/20 estratificada con semilla reproducible."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx)*test_ratio))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ─────────────────────────────────────────────
# ESTANDARIZACIÓN
# ─────────────────────────────────────────────

def standardize(X_train, X_test):
    """
    Estandariza con mu/sd del TRAIN únicamente.
    Aplicar a test sin recalcular evita data leakage.
    """
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train-mu)/sd, (X_test-mu)/sd, mu, sd

# ─────────────────────────────────────────────
# RIDGE MANUAL
# ─────────────────────────────────────────────

def ridge_fit(Xs, Y, alpha):
    """
    β̂_Ridge = (XᵀX + αI)⁻¹ Xᵀy
    Usando np.linalg.solve por estabilidad numérica.
    """
    p = Xs.shape[1]
    A = Xs.T @ Xs + alpha * np.eye(p)
    return np.linalg.solve(A, Xs.T @ Y)

def ridge_predict(Xs, B):
    return Xs @ B

def predict_class(Xs, B, T):
    """
    Clase = argmin_i ||ŷ - t_i||²
    Regla de Bayes óptima bajo gaussianas esféricas.
    """
    Y_pred = ridge_predict(Xs, B)
    dists  = np.array([
        np.sum((Y_pred - T[i])**2, axis=1)
        for i in range(len(T))
    ]).T
    return np.argmin(dists, axis=1)

def compute_mse(Xs, Y_true, B):
    return float(np.mean((ridge_predict(Xs,B) - Y_true)**2))

def compute_accuracy(Xs, y_true, B, T):
    return float(np.mean(predict_class(Xs,B,T) == y_true))

# ─────────────────────────────────────────────
# RIDGE PATH
# ─────────────────────────────────────────────

def ridge_path(Xs_tr, Y_tr, Xs_val, Y_val, alphas):
    betas, mse_tr, mse_val = [], [], []
    for alpha in alphas:
        B = ridge_fit(Xs_tr, Y_tr, alpha)
        betas.append(B)
        mse_tr.append(compute_mse(Xs_tr, Y_tr, B))
        mse_val.append(compute_mse(Xs_val, Y_val, B))
    return np.array(betas), np.array(mse_tr), np.array(mse_val)

def cross_validate(Xs, Y, alphas, n_folds=N_CV_FOLDS):
    """k-fold CV para selección de α*."""
    n       = len(Xs)
    fold_sz = n // n_folds
    cv_mse  = np.zeros(len(alphas))
    for fold in range(n_folds):
        vs = fold * fold_sz
        ve = vs + fold_sz if fold < n_folds-1 else n
        vi = np.arange(vs, ve)
        ti = np.concatenate([np.arange(0,vs), np.arange(ve,n)])
        for i, alpha in enumerate(alphas):
            B = ridge_fit(Xs[ti], Y[ti], alpha)
            cv_mse[i] += compute_mse(Xs[vi], Y[vi], B) / n_folds
    return cv_mse

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "█"*57)
    print("  RIDGE MODEL — SpongeBob Expression Classifier")
    print("  Reto Ridge Regression | TE3002B")
    print("█"*57)

    if not DATASET_FILE.exists():
        print(f"\n  ❌ No se encontró {DATASET_FILE}")
        print(f"     Ejecuta: python 03_preprocessing.py\n")
        return

    data = np.load(DATASET_FILE, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"\n  Dataset: X={X.shape}, y={y.shape}")
    print(f"  Clases : {dict(zip(*np.unique(y, return_counts=True)))}")

    # Símplex
    T = build_simplex_vertices(N_CLASSES)
    Y = T[y]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    Y_train, Y_test = T[y_train], T[y_test]
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Estandarizar
    Xs_train, Xs_test, mu, sd = standardize(X_train, X_test)
    print(f"  Estandarización: mu[:3]={mu[:3].round(4)}, sd[:3]={sd[:3].round(4)}")

    # Alphas
    alphas = np.logspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), N_ALPHAS)
    print(f"\n  Ridge path: {N_ALPHAS} alphas [{ALPHA_MIN:.0e}, {ALPHA_MAX:.0e}]")

    # Cross-validation
    print(f"  {N_CV_FOLDS}-fold CV para α*...")
    cv_mse     = cross_validate(Xs_train, Y_train, alphas)
    best_idx   = int(np.argmin(cv_mse))
    alpha_star = alphas[best_idx]
    print(f"  ✓ α* = {alpha_star:.4f}  (MSE_cv = {cv_mse[best_idx]:.6f})")

    # Modelo final
    B_final = ridge_fit(Xs_train, Y_train, alpha_star)

    # Ridge path completo (para visualizaciones)
    n_val    = int(len(Xs_train)*0.2)
    betas, mse_tr, mse_val = ridge_path(
        Xs_train[n_val:], Y_train[n_val:],
        Xs_train[:n_val], Y_train[:n_val],
        alphas
    )

    # Evaluación
    acc_train = compute_accuracy(Xs_train, y_train, B_final, T)
    acc_test  = compute_accuracy(Xs_test,  y_test,  B_final, T)
    mse_test  = compute_mse(Xs_test, Y_test, B_final)

    print(f"\n  Resultados con α* = {alpha_star:.4f}:")
    print(f"     Accuracy train : {acc_train*100:.1f}%")
    print(f"     Accuracy test  : {acc_test*100:.1f}%")
    print(f"     MSE test       : {mse_test:.6f}")

    # Guardar
    np.savez(MODEL_FILE,
             B=B_final, T=T, mu=mu, sd=sd,
             alpha_star=alpha_star, alphas=alphas,
             betas=betas, mse_train=mse_tr, mse_val=mse_val,
             cv_mse=cv_mse, y_train=y_train, y_test=y_test,
             X_test=X_test, expression_names=EXPRESSION_NAMES)

    print(f"\n  ✓ Modelo guardado: {MODEL_FILE}")
    print(f"\n  ⚡ Siguiente: python 05_visualizations.py\n")

if __name__ == "__main__":
    main()
