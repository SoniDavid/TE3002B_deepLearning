"""
03_preprocessing.py — SpongeBob Edition
=========================================
Preprocesamiento de imágenes de SpongeBob para el clasificador Ridge.

Pipeline por imagen:
    1. Cargar imagen
    2. Convertir a escala de grises
    3. Redimensionar a 32×32 px
    4. Normalizar a [0, 1]
    5. Aplanar a vector de 1,024 features

Salida:
    data/processed/<expresion>/  — imágenes 32x32
    data/dataset.npz             — X (N,1024) e y (N,)
    data/histogramas/            — visualizaciones

Uso:
    python 03_preprocessing.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)

CLEAN_DIR     = Path("data/spongebob")   # carpeta con las imágenes limpias
PROCESSED_DIR = Path("data/processed")
DATASET_FILE  = Path("data/dataset.npz")
DISCARD_LOG   = Path("data/limpieza_log.csv")
HIST_DIR      = Path("data/histogramas")
32
IMG_SIZE = (64, 64)  # tamaño para redimensionar (32x32 es muy pequeño)

EXPRESSION_LABELS = {
    "happy":     0,
    "angry":     1,
    "crying":    2,
    "surprised": 3,
    "confused":  4,
}
LABEL_NAMES = {v: k for k, v in EXPRESSION_LABELS.items()}
VALID_EXT   = {".jpg", ".jpeg", ".png", ".webp"}

# ─────────────────────────────────────────────
# PROCESAR UNA IMAGEN
# ─────────────────────────────────────────────

def process_image(img_path: Path, discard_writer) -> np.ndarray | None:
    """
    Convierte imagen a vector normalizado 32x32 en escala de grises.
    Descarta imágenes con histograma saturado.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        img = Image.open(img_path).convert("L")  # directo a grises
    except Exception as e:
        discard_writer.writerow({
            "nombre_archivo": img_path.name,
            "expresion":      img_path.parent.name,
            "razon":          f"error_carga: {str(e)[:50]}",
            "fecha_captura":  timestamp,
        })
        return None

    img_resized = img.resize(IMG_SIZE, Image.LANCZOS)
    arr         = np.array(img_resized, dtype=np.float32) / 255.0

    # Verificar histograma — descartar si >60% en extremos
    hist, _       = np.histogram(arr, bins=10, range=(0,1))
    extreme_ratio = (hist[0] + hist[-1]) / arr.size
    if extreme_ratio > 0.60:
        discard_writer.writerow({
            "nombre_archivo": img_path.name,
            "expresion":      img_path.parent.name,
            "razon":          f"histograma_saturado_{extreme_ratio:.2f}",
            "fecha_captura":  timestamp,
        })
        return None

    return arr.flatten()

# ─────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────

def plot_histogram(X: np.ndarray, sample_size: int = 10):
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    sample = X[:sample_size].flatten()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist((sample*255).astype(int), bins=50,
                 color="#e07b54", edgecolor="none")
    axes[0].set_title("Antes (escala 0-255)")
    axes[0].set_xlabel("Valor de píxel")
    axes[0].set_ylabel("Frecuencia")
    axes[0].text(0.98, 0.95,
                 f"μ={np.mean(sample*255):.1f}\nσ={np.std(sample*255):.1f}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=8)

    axes[1].hist(sample, bins=50, color="#5497d4", edgecolor="none")
    axes[1].set_title("Después (normalizado [0,1])")
    axes[1].set_xlabel("Valor de píxel")
    axes[1].text(0.98, 0.95,
                 f"μ={np.mean(sample):.3f}\nσ={np.std(sample):.3f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8)

    fig.suptitle(f"Histograma de píxeles — lote de {sample_size} imágenes", fontsize=11)
    plt.tight_layout()
    out = HIST_DIR / "histograma_normalizacion.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  ✓ Histograma guardado: {out}")


def plot_sample_faces(X: np.ndarray, y: np.ndarray, n_per_class: int = 5):
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    n_classes = len(EXPRESSION_LABELS)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class*1.8, n_classes*1.8))
    fig.suptitle("SpongeBob — Muestras procesadas (64×64 px)", fontsize=11)

    for cls_idx in range(n_classes):
        mask    = (y == cls_idx)
        samples = X[mask][:n_per_class]
        label   = LABEL_NAMES[cls_idx]
        for j in range(n_per_class):
            ax = axes[cls_idx, j]
            if j < len(samples):
                ax.imshow(samples[j].reshape(64,64), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0,
                              labelpad=50, va="center")

    plt.tight_layout()
    out = HIST_DIR / "grid_caras_procesadas.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Grid de muestras: {out}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "█"*57)
    print("  PREPROCESAMIENTO — SpongeBob Expression Dataset")
    print("  Reto Ridge Regression | TE3002B")
    print("█"*57)

    if not CLEAN_DIR.exists():
        print(f"\n  ❌ No se encontró {CLEAN_DIR}/")
        print(f"     Ejecuta primero 02_labeler.py\n")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)

    X_all, y_all = [], []
    resumen      = {}

    with open(DISCARD_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "nombre_archivo", "expresion", "razon", "fecha_captura"
        ])
        writer.writeheader()

        for expression, label_idx in EXPRESSION_LABELS.items():
            expr_dir = CLEAN_DIR / expression
            if not expr_dir.exists():
                print(f"  ⚠ Sin carpeta: {expr_dir}")
                resumen[expression] = (0, 0)
                continue

            images = [p for p in expr_dir.glob("*")
                      if p.suffix.lower() in VALID_EXT]
            out_dir = PROCESSED_DIR / expression
            out_dir.mkdir(parents=True, exist_ok=True)

            ok = 0
            for path in images:
                vec = process_image(path, writer)
                if vec is not None:
                    X_all.append(vec)
                    y_all.append(label_idx)
                    ok += 1
                    img_out = Image.fromarray(
                        (vec.reshape(64,64)*255).astype(np.uint8))
                    img_out.save(out_dir / f"{expression}_{ok:04d}.png")

            discarded = len(images) - ok
            resumen[expression] = (ok, discarded)
            print(f"  ✓ {expression:<13} {ok:>4} OK  |  {discarded:>3} descartadas")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)

    np.savez(DATASET_FILE, X=X, y=y,
             label_names=list(EXPRESSION_LABELS.keys()))
    print(f"\n  ✓ Dataset: {DATASET_FILE}  →  X={X.shape}, y={y.shape}")

    if len(X) > 0:
        plot_histogram(X, min(10, len(X)))
        plot_sample_faces(X, y)

    print("\n  RESUMEN:")
    total_ok = total_desc = 0
    for expr, (ok, desc) in resumen.items():
        status = "✓" if ok >= 80 else "⚠"
        print(f"  {status} {expr:<13} {ok:>4} procesadas  |  {desc:>3} descartadas")
        total_ok += ok; total_desc += desc
    print(f"\n  TOTAL: {total_ok} listas  |  {total_desc} descartadas")
    print(f"\n  ⚡ Siguiente: python 04_ridge_model.py\n")


if __name__ == "__main__":
    main()
