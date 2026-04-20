"""
06_predict.py — SpongeBob Edition
====================================
Clasifica la expresión de SpongeBob en una imagen.

Uso:
    python 06_predict.py image.png
    python 06_predict.py ruta/cara_spongebob.jpg
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

MODEL_FILE = Path("data/model.npz")
IMG_SIZE   = (64, 64)

EXPRESSION_NAMES = [
    "happy", "angry", "crying", "surprised", "confused"
]

EXPRESSION_EMOJI = {
    "happy":     "😄", "angry":     "😠",
    "crying":    "😢", "surprised": "😲",
    "confused":  "😕",
}

# Rango amarillo de SpongeBob
YELLOW_LOW  = np.array([18, 150, 150])
YELLOW_HIGH = np.array([38, 255, 255])


def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"\n  ❌ Modelo no encontrado: {path}\n"
            f"     Ejecuta: python 04_ridge_model.py\n"
        )
    return dict(np.load(path, allow_pickle=True))


def detect_and_crop_spongebob(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detecta SpongeBob por color amarillo y recorta su región.
    Si no detecta, usa recorte centrado como fallback.
    """
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    total = img_bgr.shape[0] * img_bgr.shape[1]
    ratio = np.sum(mask>0) / total

    if ratio >= 0.05:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            ih,iw   = img_bgr.shape[:2]
            mx = int(w*0.25); my = int(h*0.25)
            side = max(w+2*mx, h+2*my)
            cx,cy = x+w//2, y+h//2
            x1=max(0,cx-side//2); y1=max(0,cy-side//2)
            x2=min(iw,x1+side);   y2=min(ih,y1+side)
            x1=max(0,x2-side);    y1=max(0,y2-side)
            return img_bgr[y1:y2, x1:x2]

    # Fallback: recorte centrado cuadrado
    h,w = img_bgr.shape[:2]
    side = min(w,h)
    x1 = (w-side)//2; y1 = (h-side)//2
    return img_bgr[y1:y1+side, x1:x1+side]


def preprocess(img_path: Path, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    img_bgr  = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo abrir: {img_path}")

    crop     = detect_and_crop_spongebob(img_bgr)
    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized  = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    arr      = resized.astype(np.float32) / 255.0
    flat     = arr.flatten()
    sd_safe  = np.where(sd==0, 1.0, sd)
    return ((flat - mu) / sd_safe).reshape(1,-1)


def predict(Xs, B, T):
    Y_pred = Xs @ B
    dists  = np.array([float(np.sum((Y_pred-T[i])**2))
                       for i in range(len(T))])
    cls    = int(np.argmin(dists))
    return cls, dists


def confidence(dists):
    inv  = 1.0 / (dists + 1e-8)
    soft = np.exp(inv - np.max(inv))
    soft /= soft.sum()
    return float(soft[np.argmin(dists)]*100)


def print_result(img_path, cls, dists, conf):
    name  = EXPRESSION_NAMES[cls].upper()
    emoji = EXPRESSION_EMOJI[EXPRESSION_NAMES[cls]]
    top3  = np.argsort(dists)[:3]
    sep   = "─" * 48

    print(f"\n  ┌{sep}┐")
    print(f"  │  Imagen   : {img_path.name:<33}│")
    print(f"  ├{sep}┤")
    print(f"  │  Expresión : {emoji} {name:<31}│")
    print(f"  │  Confianza : {conf:>5.1f}%{'':<29}│")
    print(f"  ├{sep}┤")
    print(f"  │  Top-3:{'':>39}│")
    for rank, idx in enumerate(top3, 1):
        n = EXPRESSION_NAMES[idx]
        d = f"{dists[idx]:.4f}"
        print(f"  │    {rank}. {n:<16} dist: {d:<16}│")
    print(f"  └{sep}┘\n")


def main():
    if len(sys.argv) < 2:
        print("\n  Uso: python 06_predict.py <imagen>")
        print("  Ejemplo: python 06_predict.py spongebob_test.jpg\n")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"\n  ❌ Imagen no encontrada: {img_path}\n")
        sys.exit(1)

    model = load_model(MODEL_FILE)
    B, T, mu, sd = model["B"], model["T"], model["mu"], model["sd"]

    print(f"\n  Modelo cargado: α* = {float(model['alpha_star']):.4f}")
    print(f"  Procesando: {img_path}...")

    Xs          = preprocess(img_path, mu, sd)
    cls, dists  = predict(Xs, B, T)
    conf        = confidence(dists)
    print_result(img_path, cls, dists, conf)


if __name__ == "__main__":
    main()
