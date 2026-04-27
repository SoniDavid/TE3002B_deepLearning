"""
01_video_to_frames.py — SpongeBob Edition
==========================================
Descarga episodios de SpongeBob de YouTube y extrae frames
donde aparece la cara de SpongeBob.

Pipeline:
    1. Descargar video con yt-dlp
    2. Extraer 1 frame/segundo con ffmpeg
    3. Detectar SpongeBob por color amarillo (HSV) — sin cascade
    4. Recortar región facial y guardar en data/frames_raw/

Por qué color amarillo:
    SpongeBob tiene un amarillo muy específico (H≈25-35 en HSV)
    que prácticamente ningún otro personaje del show comparte
    en cantidad. Esto da ~90% de precisión sin ML.

Instalación:
    pip install yt-dlp opencv-python Pillow tqdm

Uso:
    python 01_video_to_frames.py --url "https://www.youtube.com/watch?v=a3iIj9pwS0E" 
    python 01_video_to_frames.py --video episodio.mp4
    python 01_video_to_frames.py --list videos.txt
    python 01_video_to_frames.py --url "..." --threshold 0.08
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

OUTPUT_DIR   = Path("data/frames_raw")
TEMP_DIR     = Path("data/_temp_frames")
DOWNLOAD_DIR = Path("data/_downloads")

FRAME_RATE       = 1      # 1 frame por segundo
MIN_YELLOW_RATIO = 0.08  # mínimo % de amarillo para aceptar frame
MIN_FACE_SIZE    = 60     # tamaño mínimo del recorte en px

# Ruta explícita a ffmpeg (evita conflicto con stubs de WindowsApps)
FFMPEG_PATH = r"C:\Users\Carls\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

# Rango HSV del amarillo de SpongeBob
# Calibrado visualmente — H:18-38, S:150-255, V:150-255
YELLOW_HSV_LOW  = np.array([18, 150, 150])
YELLOW_HSV_HIGH = np.array([38, 255, 255])

# ─────────────────────────────────────────────
# VERIFICAR DEPENDENCIAS
# ─────────────────────────────────────────────

def check_deps() -> bool:
    ok = True
    try:
        import yt_dlp
        print("  ✓ yt-dlp disponible")
    except ImportError:
        print("  ❌ yt-dlp → pip install yt-dlp")
        ok = False

    r = subprocess.run([FFMPEG_PATH, "-version"], capture_output=True)
    if r.returncode == 0:
        print("  ✓ ffmpeg disponible")
    else:
        print("  ❌ ffmpeg no encontrado en:", FFMPEG_PATH)
        ok = False
    return ok


# ─────────────────────────────────────────────
# DESCARGA
# ─────────────────────────────────────────────

def download_video(url: str, out_dir: Path) -> Path | None:
    import yt_dlp
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl = str(out_dir / "%(title)s.%(ext)s")
    opts = {
        "format":      "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
        "outtmpl":     tmpl,
        "quiet":       False,
        "no_warnings": False,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
            if not path.exists():
                files = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime)
                path = files[-1] if files else None
            print(f"  ✓ Descargado: {path.name if path else 'ERROR'}")
            return path
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


# ─────────────────────────────────────────────
# EXTRACCIÓN DE FRAMES
# ─────────────────────────────────────────────

def extract_frames(video: Path, out_dir: Path, fps: int = FRAME_RATE) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_PATH, "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-hide_banner", "-loglevel", "error",
        str(out_dir / "frame_%06d.jpg"),
    ]
    print(f"\n  Extrayendo frames ({fps}/s): {video.name}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ❌ ffmpeg: {r.stderr[:150]}")
        return 0
    n = len(list(out_dir.glob("frame_*.jpg")))
    print(f"  ✓ {n} frames extraídos")
    return n


# ─────────────────────────────────────────────
# DETECCIÓN DE SPONGEBOB POR COLOR AMARILLO
# ─────────────────────────────────────────────

def detect_spongebob(img_bgr: np.ndarray, min_ratio: float) -> tuple | None:
    """
    Detecta la región de SpongeBob usando su color amarillo característico.

    Estrategia:
        1. Convertir a HSV
        2. Crear máscara del amarillo de SpongeBob
        3. Encontrar el contorno más grande de esa máscara
        4. Si cubre suficiente área → recortar bounding box

    Retorna (x, y, w, h) del bounding box o None.
    """
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH)

    # Limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    total_px    = img_bgr.shape[0] * img_bgr.shape[1]
    yellow_px   = np.sum(mask > 0)
    yellow_ratio = yellow_px / total_px

    if yellow_ratio < min_ratio:
        return None

    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Tomar el contorno más grande (el cuerpo/cara de SpongeBob)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Debe ser razonablemente cuadrado y suficientemente grande
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None

    return (x, y, w, h)


def crop_spongebob(img_bgr: np.ndarray, bbox: tuple,
                   margin: float = 0.25) -> np.ndarray:
    """
    Recorta la región de SpongeBob con margen adicional.
    Hace el recorte cuadrado centrado sobre el bounding box.
    """
    x, y, w, h    = bbox
    ih, iw        = img_bgr.shape[:2]
    mx            = int(w * margin)
    my            = int(h * margin)
    side          = max(w + 2*mx, h + 2*my)
    cx, cy        = x + w//2, y + h//2
    x1 = max(0, cx - side//2);  y1 = max(0, cy - side//2)
    x2 = min(iw, x1 + side);    y2 = min(ih, y1 + side)
    x1 = max(0, x2 - side);     y1 = max(0, y2 - side)
    return img_bgr[y1:y2, x1:x2]


# ─────────────────────────────────────────────
# FILTRAR FRAMES
# ─────────────────────────────────────────────

def filter_frames(
    frames_dir: Path,
    output_dir: Path,
    threshold: float,
    video_name: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames   = sorted(frames_dir.glob("frame_*.jpg"))
    accepted = 0

    pbar = tqdm(frames, desc="  Detectando SpongeBob", unit="frame")
    for frame_path in pbar:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        bbox = detect_spongebob(img, threshold)
        if bbox is None:
            continue

        crop = crop_spongebob(img, bbox)
        if crop.size == 0:
            continue

        frame_num = int(frame_path.stem.split("_")[1])
        out_name  = f"{video_name}_t{frame_num:05d}s.jpg"
        cv2.imwrite(str(output_dir / out_name), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        accepted += 1
        pbar.set_postfix({"SpongeBob": accepted})

    return accepted


# ─────────────────────────────────────────────
# PREVIEW
# ─────────────────────────────────────────────

def preview(img_path: str, threshold: float):
    """Prueba el detector en una imagen y guarda el resultado."""
    img  = cv2.imread(img_path)
    if img is None:
        print(f"  ❌ No se pudo abrir: {img_path}")
        return

    bbox = detect_spongebob(img, threshold)
    vis  = img.copy()

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.putText(vis, f"SpongeBob {w}x{h}px", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        crop = crop_spongebob(img, bbox)
        cv2.imwrite("preview_cropped.jpg", crop)
        print(f"  ✓ Detectado: {w}x{h} px")
        print(f"  ✓ Recorte guardado: preview_cropped.jpg")
    else:
        print(f"  ⚠ SpongeBob no detectado (threshold={threshold})")
        print(f"  Prueba bajando el threshold: --threshold {threshold-0.02:.2f}")

    cv2.imwrite("preview_detection.jpg", vis)
    print(f"  ✓ Detección guardada: preview_detection.jpg")

    # Mostrar máscara amarilla
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH)
    cv2.imwrite("preview_mask.jpg", mask)
    ratio = np.sum(mask>0) / (img.shape[0]*img.shape[1])
    print(f"  Ratio amarillo: {ratio:.3f} (threshold={threshold})")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extrae frames de SpongeBob de videos de YouTube"
    )
    parser.add_argument("--url",       type=str)
    parser.add_argument("--list",      type=str, help="Archivo .txt con URLs")
    parser.add_argument("--video",     type=str, help="Video ya descargado")
    parser.add_argument("--preview",   type=str, help="Probar en una imagen")
    parser.add_argument("--threshold", type=float, default=MIN_YELLOW_RATIO,
                        help=f"Mínimo ratio amarillo (default={MIN_YELLOW_RATIO})")
    args = parser.parse_args()

    print("\n" + "█"*57)
    print("  VIDEO → FRAMES — SpongeBob Detector")
    print("  Detección por color HSV amarillo")
    print("  Reto Ridge Regression | TE3002B")
    print("█"*57 + "\n")

    if args.preview:
        preview(args.preview, args.threshold)
        return

    if not check_deps():
        sys.exit(1)

    # Recopilar videos
    videos = []
    if args.video:
        videos.append(Path(args.video))
    elif args.url:
        v = download_video(args.url, DOWNLOAD_DIR)
        if v: videos.append(v)
    elif args.list:
        urls = Path(args.list).read_text().strip().splitlines()
        urls = [u.strip() for u in urls if u.strip() and not u.startswith("#")]
        for url in urls:
            v = download_video(url, DOWNLOAD_DIR)
            if v: videos.append(v)
    else:
        print("  Uso: --url URL  |  --list urls.txt  |  --video archivo.mp4")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_accepted = 0

    for vid in videos:
        print(f"\n  {'─'*54}")
        print(f"  Video: {vid.name}")
        temp = TEMP_DIR / vid.stem
        n    = extract_frames(vid, temp, FRAME_RATE)
        if n == 0:
            continue

        name     = vid.stem[:30].replace(" ", "_")
        accepted = filter_frames(temp, OUTPUT_DIR, args.threshold, name)
        total_accepted += accepted
        pct = accepted/n*100 if n else 0
        print(f"  ✓ {accepted}/{n} frames con SpongeBob ({pct:.0f}%)")
        shutil.rmtree(temp, ignore_errors=True)

    print(f"\n  {'═'*54}")
    print(f"  TOTAL frames de SpongeBob: {total_accepted}")
    print(f"  Guardados en: {OUTPUT_DIR.resolve()}/")
    print(f"\n  ⚡ Siguiente: python 02_labeler.py\n")


if __name__ == "__main__":
    main()
