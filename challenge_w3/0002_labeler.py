"""
02_labeler.py — SpongeBob Heuristic Auto-Labeler
==================================================
Clasifica automáticamente expresiones de SpongeBob usando
análisis de color y forma con OpenCV. Sin API, sin costo.

Heurísticas usadas:
    - Lágrimas: píxeles azules/transparentes bajo los ojos → crying
    - Boca muy abierta + grande: área oscura grande → surprised/excited
    - Región de cejas oscura y baja: → angry
    - Mejillas rojas/rosadas: → embarrassed/excited
    - Boca pequeña o cerrada: → serious/confused/smug
    - Ojos muy grandes respecto a la cara: → surprised/scared
    - Brillo general alto (cara muy iluminada): → happy
    - Combinaciones de señales → clasificación final

Precisión esperada: ~60-65%
Tú corriges el resto con la UI manual (Enter=ok, 1-0=cambiar)

Uso:
    python 02_labeler.py --auto          # solo heurística
    python 02_labeler.py --manual        # solo UI manual
    python 02_labeler.py                 # auto + manual juntos
    python 02_labeler.py --stats
    python 02_labeler.py --manual --expression angry
"""

import argparse, json, shutil, sys, tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

import cv2
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

FRAMES_DIR   = Path("data/frames_raw")
CLEAN_DIR    = Path("data/clean")
SKIP_DIR     = Path("data/skipped")
DELETE_DIR   = Path("data/deleted")
PROGRESS     = Path("data/label_progress.json")
AUTO_LOG     = Path("data/auto_label_log.json")
DISPLAY_SIZE = 320

EXPRESSIONS = {
    "1":"happy",    "2":"angry",     "3":"crying",
    "4":"smug",     "5":"surprised", "6":"scared",
    "7":"serious",  "8":"confused",  "9":"disgusted",
    "0":"excited",
}
EXPR_COLORS = {
    "happy":"#FFD700",    "angry":"#E05252",
    "crying":"#6BB5E0",   "smug":"#9B59B6",
    "surprised":"#F7DC6F","scared":"#52BE80",
    "serious":"#566573",  "confused":"#85C1E9",
    "disgusted":"#A8D8A8","excited":"#F39C12",
    "skip":"#95A5A6",     "delete":"#E74C3C",
}
EXPR_EMOJI = {
    "happy":"😄","angry":"😠","crying":"😢","smug":"😏",
    "surprised":"😲","scared":"😨","serious":"😐","confused":"😕",
    "disgusted":"🤢","excited":"🤩","skip":"⏭","delete":"🗑",
}
VALID_EXT = {".jpg",".jpeg",".png",".webp"}

# ─────────────────────────────────────────────
# PROGRESO
# ─────────────────────────────────────────────

def load_progress():
    if PROGRESS.exists(): return json.loads(PROGRESS.read_text())
    return {"labeled":{}, "history":[]}

def save_progress(p):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(p, indent=2))

def load_auto_log():
    if AUTO_LOG.exists(): return json.loads(AUTO_LOG.read_text())
    return {}

def save_auto_log(l):
    AUTO_LOG.parent.mkdir(parents=True, exist_ok=True)
    AUTO_LOG.write_text(json.dumps(l, indent=2))

def print_stats():
    prog = load_progress()
    counts = {}
    for lbl in prog.get("labeled",{}).values():
        counts[lbl] = counts.get(lbl,0)+1
    total = sum(counts.values())
    print(f"\n  {'─'*48}")
    print(f"  ESTADÍSTICAS — SpongeBob Expression Labeler")
    print(f"  {'─'*48}  Total: {total}\n")
    for expr in list(EXPRESSIONS.values())+["skip","delete"]:
        n = counts.get(expr,0)
        bar = "█" * min(n//2, 22)
        print(f"  {EXPR_EMOJI.get(expr,'')} {expr:<12} {bar:<22} {n:>4}")
    print(f"  {'─'*48}")
    if CLEAN_DIR.exists():
        print(f"\n  Archivos en data/clean/:")
        for d in sorted(CLEAN_DIR.iterdir()):
            if d.is_dir():
                n = len(list(d.glob("*")))
                print(f"    {d.name:<15} {n:>4}")
    print()

# ─────────────────────────────────────────────
# HEURÍSTICAS DE EXPRESIÓN
# ─────────────────────────────────────────────

def extract_features(img_bgr: np.ndarray) -> dict:
    """
    Extrae features visuales de la imagen de SpongeBob.
    Todas las medidas son ratios (0-1) para ser independientes del tamaño.
    """
    h, w = img_bgr.shape[:2]
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Zonas de interés ────────────────────────────────────────────
    # Superior (cejas/ojos): filas 15-50% de la imagen
    # Inferior (boca): filas 55-90% de la imagen
    eye_zone   = img_bgr[int(h*0.15):int(h*0.50), :]
    mouth_zone = img_bgr[int(h*0.55):int(h*0.90), :]
    cheek_zone = img_bgr[int(h*0.35):int(h*0.65), :]
    lower_eye  = img_bgr[int(h*0.35):int(h*0.55), :]  # zona bajo ojos (lágrimas)

    # ── 1. LÁGRIMAS — píxeles azules/cyan en zona bajo ojos ────────
    # SpongeBob llora con lágrimas azules brillantes
    lower_hsv = cv2.cvtColor(lower_eye, cv2.COLOR_BGR2HSV)
    tear_mask = cv2.inRange(lower_hsv, (90,50,100), (130,255,255))
    tear_ratio = np.sum(tear_mask>0) / (lower_eye.shape[0]*lower_eye.shape[1]+1)

    # ── 2. BOCA ABIERTA — zona oscura grande en área de boca ───────
    mouth_gray = cv2.cvtColor(mouth_zone, cv2.COLOR_BGR2GRAY)
    _, mouth_dark = cv2.threshold(mouth_gray, 60, 255, cv2.THRESH_BINARY_INV)
    mouth_open_ratio = np.sum(mouth_dark>0) / (mouth_zone.shape[0]*mouth_zone.shape[1]+1)

    # ── 3. CEJAS BAJAS/FRUNCIDAS — oscuro en zona superior ─────────
    eye_gray = cv2.cvtColor(eye_zone, cv2.COLOR_BGR2GRAY)
    # Píxeles muy oscuros en la zona de cejas = cejas fruncidas
    brow_dark = np.sum(eye_gray[:int(eye_zone.shape[0]*0.35)] < 60)
    brow_dark_ratio = brow_dark / (eye_zone.shape[1] * int(eye_zone.shape[0]*0.35) + 1)

    # ── 4. MEJILLAS ROJAS/ROSADAS — excited/embarrassed ────────────
    cheek_hsv = cv2.cvtColor(cheek_zone, cv2.COLOR_BGR2HSV)
    # Rosa/rojo suave (mejillas de SpongeBob cuando está emocionado)
    blush_mask = cv2.inRange(cheek_hsv, (0,30,150), (15,180,255))
    blush_ratio = np.sum(blush_mask>0) / (cheek_zone.shape[0]*cheek_zone.shape[1]+1)

    # ── 5. BRILLO GENERAL — cara muy iluminada = happy ─────────────
    brightness = float(np.mean(gray)) / 255.0

    # ── 6. ÁREA OSCURA EN BOCA PEQUEÑA — serious/smug ──────────────
    # Boca pequeña: poca área oscura pero centrada
    mouth_center = mouth_zone[:,int(w*0.25):int(w*0.75)]
    mc_gray = cv2.cvtColor(mouth_center, cv2.COLOR_BGR2GRAY)
    _, mc_dark = cv2.threshold(mc_gray, 60, 255, cv2.THRESH_BINARY_INV)
    mouth_center_ratio = np.sum(mc_dark>0) / (mc_dark.size+1)

    # ── 7. OJOS MUY ABIERTOS — surprised/scared ────────────────────
    # Blancos de los ojos: píxeles muy claros en zona de ojos
    eye_white = np.sum(eye_gray > 200)
    eye_white_ratio = eye_white / (eye_zone.shape[0]*eye_zone.shape[1]+1)

    # ── 8. ASIMETRÍA — confused ──────────────────────────────────
    left_half  = gray[:, :w//2]
    right_half = gray[:, w//2:]
    asymmetry  = float(abs(np.mean(left_half) - np.mean(right_half))) / 255.0

    # ── 9. CONTRASTE GENERAL — expresiones exageradas ──────────────
    contrast = float(np.std(gray)) / 128.0

    return {
        "tear_ratio":          tear_ratio,
        "mouth_open_ratio":    mouth_open_ratio,
        "brow_dark_ratio":     brow_dark_ratio,
        "blush_ratio":         blush_ratio,
        "brightness":          brightness,
        "mouth_center_ratio":  mouth_center_ratio,
        "eye_white_ratio":     eye_white_ratio,
        "asymmetry":           asymmetry,
        "contrast":            contrast,
    }


def classify_expression(features: dict) -> tuple[str, float]:
    """
    Clasifica la expresión basándose en las features extraídas.
    Retorna (expresión, confianza 0-1).

    Árbol de decisión calibrado para SpongeBob:
        1. Lágrimas → crying
        2. Cejas muy fruncidas + boca abierta → angry
        3. Boca muy abierta + ojos muy abiertos → surprised
        4. Mejillas + boca muy abierta → excited
        5. Muy brillante + boca abierta moderada → happy
        6. Asimetría alta → confused
        7. Boca pequeña + sin cejas fruncidas → smug/serious
        8. Boca abierta sin cejas fruncidas → scared
        9. Contraste bajo + boca cerrada → disgusted
        10. Default → serious
    """
    t  = features["tear_ratio"]
    mo = features["mouth_open_ratio"]
    bd = features["brow_dark_ratio"]
    bl = features["blush_ratio"]
    br = features["brightness"]
    mc = features["mouth_center_ratio"]
    ew = features["eye_white_ratio"]
    ay = features["asymmetry"]
    ct = features["contrast"]

    # ── Árbol de decisión ───────────────────────────────────────────

    # 1. CRYING — lágrimas azules son muy distintivas
    if t > 0.04:
        conf = min(1.0, t / 0.08)
        return "crying", conf

    # 2. ANGRY — cejas muy fruncidas + boca abierta o tensa
    if bd > 0.15 and mo > 0.12:
        conf = min(1.0, (bd + mo) / 0.5)
        return "angry", conf

    if bd > 0.20:
        conf = min(1.0, bd / 0.30)
        return "angry", conf

    # 3. SURPRISED — boca MUY abierta + ojos muy blancos
    if mo > 0.35 and ew > 0.20:
        conf = min(1.0, (mo + ew) / 0.7)
        return "surprised", conf

    # 4. EXCITED — mejillas + boca abierta + brillo alto
    if bl > 0.06 and mo > 0.20 and br > 0.55:
        conf = min(1.0, (bl*5 + mo + br) / 2.5)
        return "excited", conf

    # 5. SCARED — boca muy abierta pero sin cejas fruncidas
    if mo > 0.30 and bd < 0.10:
        conf = min(1.0, mo / 0.40)
        return "scared", conf

    # 6. HAPPY — buena iluminación, boca moderadamente abierta
    if br > 0.60 and mo > 0.15 and bd < 0.12:
        conf = min(1.0, (br + mo) / 1.0)
        return "happy", conf

    # 7. SURPRISED (moderado) — boca bastante abierta
    if mo > 0.25:
        conf = min(1.0, mo / 0.35)
        return "surprised", conf

    # 8. CONFUSED — alta asimetría entre lados
    if ay > 0.08 and ct > 0.40:
        conf = min(1.0, ay / 0.15)
        return "confused", conf

    # 9. SMUG — boca pequeña asimétrica, contraste medio
    if mc > 0.05 and mc < 0.18 and mo < 0.15 and ay > 0.04:
        conf = 0.55
        return "smug", conf

    # 10. DISGUSTED — bajo contraste, boca cerrada/tensa
    if ct < 0.35 and mo < 0.15 and bd > 0.08:
        conf = 0.50
        return "disgusted", conf

    # 11. SERIOUS — boca cerrada, cara simétrica
    if mo < 0.12 and ay < 0.05:
        conf = 0.55
        return "serious", conf

    # 12. HAPPY (default con boca abierta)
    if mo > 0.12:
        conf = 0.45
        return "happy", conf

    # 13. Default
    return "serious", 0.40


def auto_classify_image(img_path: Path) -> tuple[str, float]:
    """Clasifica una imagen. Retorna (expresión, confianza)."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return "delete", 0.0

        h, w = img.shape[:2]
        if h < 40 or w < 40:
            return "delete", 0.0

        # Redimensionar para consistencia
        img_resized = cv2.resize(img, (128, 128))
        features    = extract_features(img_resized)
        return classify_expression(features)

    except Exception:
        return "skip", 0.0


# ─────────────────────────────────────────────
# AUTO-ETIQUETADO
# ─────────────────────────────────────────────

def run_auto_label(input_dir: Path, force: bool = False):
    """Clasifica todas las imágenes con heurísticas."""
    from tqdm import tqdm

    images = sorted([p for p in input_dir.rglob("*")
                     if p.suffix.lower() in VALID_EXT])
    if not images:
        print(f"  ❌ Sin imágenes en {input_dir}"); return

    # Crear directorios
    all_exprs = list(EXPRESSIONS.values()) + ["skip", "delete"]
    for expr in all_exprs:
        (CLEAN_DIR / expr).mkdir(parents=True, exist_ok=True)
    SKIP_DIR.mkdir(parents=True, exist_ok=True)
    DELETE_DIR.mkdir(parents=True, exist_ok=True)

    auto_log = load_auto_log()
    progress = load_progress()
    labeled  = progress.setdefault("labeled", {})

    if not force:
        images = [p for p in images if p.name not in auto_log]

    if not images:
        print("  ✓ Todo ya clasificado. Usa --force para re-clasificar.\n")
        return

    print(f"\n  {'█'*54}")
    print(f"  AUTO-ETIQUETADO — Heurística OpenCV 🎨")
    print(f"  {'█'*54}")
    print(f"  Imágenes a clasificar: {len(images)}")
    print(f"  Precisión esperada   : ~60-65%")
    print(f"  (Corregirás errores con --manual)\n")

    counts = {e: 0 for e in all_exprs}
    pbar   = tqdm(images, desc="  Clasificando", unit="img")

    for img_path in pbar:
        expr, conf = auto_classify_image(img_path)

        # Copiar al directorio correspondiente
        dest = (DELETE_DIR / img_path.name if expr == "delete"
                else SKIP_DIR / img_path.name if expr == "skip"
                else CLEAN_DIR / expr / img_path.name)
        try:
            shutil.copy2(str(img_path), str(dest))
        except Exception:
            pass

        auto_log[img_path.name] = {"expression": expr, "confidence": round(conf, 3)}
        labeled[img_path.name]  = expr
        counts[expr] = counts.get(expr, 0) + 1
        pbar.set_postfix({
            "happy": counts.get("happy",0),
            "angry": counts.get("angry",0),
            "cry":   counts.get("crying",0),
        })

    save_auto_log(auto_log)
    save_progress(progress)

    # Resumen
    print(f"\n  {'═'*54}")
    print(f"  RESULTADO HEURÍSTICA")
    print(f"  {'═'*54}")
    total = sum(counts.values())
    for expr in list(EXPRESSIONS.values()) + ["skip", "delete"]:
        n = counts.get(expr, 0)
        if n > 0:
            bar = "█" * min(n//3, 20)
            print(f"  {EXPR_EMOJI.get(expr,'')} {expr:<12} {bar:<20} {n:>4}")
    print(f"  {'─'*54}  TOTAL: {total}")
    print(f"\n  ✓ Auto-clasificación completa")
    print(f"  ⚡ Ahora corre: python 02_labeler.py --manual\n")


# ─────────────────────────────────────────────
# UI MANUAL
# ─────────────────────────────────────────────

class ManualReviewer:
    def __init__(self, root, images, progress, auto_log):
        self.root     = root
        self.images   = images
        self.progress = progress
        self.auto_log = auto_log
        self.idx      = 0
        self.history  = progress.get("history", [])

        all_exprs = list(EXPRESSIONS.values()) + ["skip", "delete"]
        for expr in all_exprs:
            (CLEAN_DIR / expr).mkdir(parents=True, exist_ok=True)
        SKIP_DIR.mkdir(parents=True, exist_ok=True)
        DELETE_DIR.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._load_image()

    def _build_ui(self):
        self.root.title("SpongeBob Labeler — Revisión Manual 🧽")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # Imagen
        self.img_frame = tk.Frame(self.root, bg="#16213e",
                                  width=DISPLAY_SIZE+20,
                                  height=DISPLAY_SIZE+20)
        self.img_frame.pack(pady=(15,5), padx=20)
        self.img_frame.pack_propagate(False)
        self.img_label = tk.Label(self.img_frame, bg="#16213e")
        self.img_label.pack(expand=True)

        # Etiqueta auto
        self.auto_label = tk.Label(self.root, text="",
            fg="#F39C12", bg="#1a1a2e", font=("Consolas", 10, "bold"))
        self.auto_label.pack()

        # Info
        self.info_label = tk.Label(self.root, text="",
            fg="#a0a0b0", bg="#1a1a2e", font=("Consolas", 9))
        self.info_label.pack()

        # Acción
        self.action_label = tk.Label(self.root, text="",
            fg="white", bg="#1a1a2e",
            font=("Consolas", 13, "bold"), width=34)
        self.action_label.pack(pady=3)

        # Progreso
        self.progress_label = tk.Label(self.root, text="",
            fg="#6a6a8a", bg="#1a1a2e", font=("Consolas", 9))
        self.progress_label.pack()

        # Leyenda de teclas
        legend = tk.Frame(self.root, bg="#1a1a2e")
        legend.pack(pady=(8,5), padx=15)
        keys = [
            ("1 happy",     "#FFD700"), ("2 angry",     "#E05252"),
            ("3 crying",    "#6BB5E0"), ("4 smug",      "#9B59B6"),
            ("5 surprised", "#F7DC6F"), ("6 scared",    "#52BE80"),
            ("7 serious",   "#566573"), ("8 confused",  "#85C1E9"),
            ("9 disgusted", "#A8D8A8"), ("0 excited",   "#F39C12"),
        ]
        for i, (text, color) in enumerate(keys):
            r, c = divmod(i, 5)
            tk.Label(legend, text=text, fg="white", bg=color,
                     font=("Consolas", 8, "bold"), width=10, pady=3
                     ).grid(row=r, column=c, padx=2, pady=2)

        # Teclas especiales
        sp = tk.Frame(self.root, bg="#1a1a2e")
        sp.pack(pady=(0, 10))
        for text, color, w in [
            ("Enter confirmar", "#27AE60", 14),
            ("s skip",          "#95A5A6",  8),
            ("d delete",        "#E74C3C",  9),
            ("z undo",          "#3498DB",  7),
            ("q salir",         "#566573",  7),
        ]:
            tk.Label(sp, text=text, fg="white", bg=color,
                     font=("Consolas", 8, "bold"), width=w, pady=3
                     ).pack(side=tk.LEFT, padx=2)

        # Bindings
        for key, expr in EXPRESSIONS.items():
            self.root.bind(key, lambda e, x=expr: self._label(x))
        self.root.bind("<Return>", lambda e: self._keep())
        self.root.bind("s",        lambda e: self._label("skip"))
        self.root.bind("d",        lambda e: self._label("delete"))
        self.root.bind("z",        lambda e: self._undo())
        self.root.bind("q",        lambda e: self._quit())
        self.root.bind("<Escape>", lambda e: self._quit())

    def _load_image(self):
        if self.idx >= len(self.images):
            self._finish(); return

        path = self.images[self.idx]
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
            bg  = Image.new("RGB", (DISPLAY_SIZE, DISPLAY_SIZE), (22,33,62))
            bg.paste(img, ((DISPLAY_SIZE-img.width)//2,
                           (DISPLAY_SIZE-img.height)//2))
            self.tk_img = ImageTk.PhotoImage(bg)
            self.img_label.configure(image=self.tk_img)
        except Exception as e:
            self.action_label.configure(text=f"Error: {e}", fg="#E74C3C")
            self.idx += 1; self.root.after(300, self._load_image); return

        # Mostrar sugerencia automática
        log_entry  = self.auto_log.get(path.name, {})
        auto_expr  = (log_entry.get("expression", "?")
                      if isinstance(log_entry, dict)
                      else log_entry)
        auto_conf  = (log_entry.get("confidence", 0.0)
                      if isinstance(log_entry, dict)
                      else 0.0)
        color = EXPR_COLORS.get(auto_expr, "#ffffff")
        emoji = EXPR_EMOJI.get(auto_expr, "")
        self.auto_label.configure(
            text=f"🎨 Auto: {emoji} {auto_expr.upper()} "
                 f"({auto_conf*100:.0f}%)  ← Enter para confirmar",
            fg=color)

        labeled_n = len(self.progress.get("labeled", {}))
        remaining = len(self.images) - self.idx
        self.info_label.configure(text=path.name[:55])
        self.progress_label.configure(
            text=f"Imagen {self.idx+1}/{len(self.images)}  |  "
                 f"Revisadas: {labeled_n}  |  Restantes: {remaining}")
        self.action_label.configure(
            text="Enter=confirmar  |  1-0=cambiar", fg="#6a6a8a")

    def _move(self, path, expression):
        """Mueve el archivo al directorio correcto."""
        all_exprs = list(EXPRESSIONS.values()) + ["skip", "delete"]
        for expr in all_exprs:
            old = CLEAN_DIR / expr / path.name
            if old.exists(): old.unlink()
        for d in [SKIP_DIR/path.name, DELETE_DIR/path.name]:
            if d.exists(): d.unlink()
        dest = (DELETE_DIR/path.name if expression == "delete"
                else SKIP_DIR/path.name if expression == "skip"
                else CLEAN_DIR/expression/path.name)
        shutil.copy2(str(path), str(dest))
        return dest

    def _keep(self):
        """Confirma la etiqueta automática."""
        if self.idx >= len(self.images): return
        log_entry = self.auto_log.get(self.images[self.idx].name, {})
        auto_expr = (log_entry.get("expression", "skip")
                     if isinstance(log_entry, dict) else log_entry)
        self._label(auto_expr, confirmed=True)

    def _label(self, expression, confirmed=False):
        if self.idx >= len(self.images): return
        path = self.images[self.idx]
        dest = self._move(path, expression)

        labeled = self.progress.setdefault("labeled", {})
        labeled[path.name] = expression
        self.history.append({
            "filename": path.name, "expression": expression,
            "confirmed": confirmed, "dest": str(dest),
        })
        self.progress["history"] = self.history
        save_progress(self.progress)

        color = EXPR_COLORS.get(expression, "#ffffff")
        emoji = EXPR_EMOJI.get(expression, "")
        tag   = " ✓" if confirmed else ""
        self.action_label.configure(
            text=f"  {emoji}  {expression.upper()}{tag}  ", fg=color)
        self.img_frame.configure(bg=color)
        self.root.after(180, lambda: self.img_frame.configure(bg="#16213e"))
        self.idx += 1
        self.root.after(220, self._load_image)

    def _undo(self):
        if not self.history:
            self.action_label.configure(text="Nada que deshacer", fg="#E74C3C")
            return
        last = self.history.pop()
        self.progress.get("labeled",{}).pop(last["filename"], None)
        self.progress["history"] = self.history
        save_progress(self.progress)
        p = Path(last["dest"])
        if p.exists(): p.unlink()
        self.idx = max(0, self.idx-1)
        self.action_label.configure(
            text=f"↩ {last['expression']}", fg="#2ECC71")
        self._load_image()

    def _finish(self):
        self.action_label.configure(text="✓ ¡Revisión completada!", fg="#2ECC71")
        self.root.after(2000, self._quit)

    def _quit(self):
        save_progress(self.progress)
        print_stats()
        self.root.destroy()


def run_manual_review(input_dir: Path, expression_filter=None):
    auto_log = load_auto_log()
    progress = load_progress()

    if expression_filter:
        src = CLEAN_DIR / expression_filter
        if not src.exists():
            print(f"  ❌ Sin carpeta: {src}"); return
        images = sorted([p for p in src.glob("*")
                         if p.suffix.lower() in VALID_EXT])
    else:
        images = sorted([p for p in input_dir.rglob("*")
                         if p.suffix.lower() in VALID_EXT])

    if not images:
        print("  ❌ Sin imágenes para revisar."); return

    print(f"\n  {'█'*52}")
    print(f"  REVISIÓN MANUAL 🧽  ({len(images)} imágenes)")
    print(f"  {'█'*52}")
    print(f"  Enter → confirmar auto  |  1-0 → cambiar")
    print(f"  s → skip  d → delete  z → undo  q → salir\n")

    root = tk.Tk()
    ManualReviewer(root, images, progress, auto_log)
    root.mainloop()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Etiquetador heurístico + manual de SpongeBob"
    )
    parser.add_argument("--auto",       action="store_true",
                        help="Auto-clasificación con heurística")
    parser.add_argument("--manual",     action="store_true",
                        help="Revisión manual")
    parser.add_argument("--force",      action="store_true",
                        help="Re-clasificar todo")
    parser.add_argument("--stats",      action="store_true",
                        help="Ver estadísticas")
    parser.add_argument("--expression", type=str, default=None,
                        help="Revisar solo una expresión (ej: angry)")
    parser.add_argument("--input",      type=str,
                        default=str(FRAMES_DIR))
    args = parser.parse_args()

    if args.stats:
        print_stats(); return

    if not args.auto and not args.manual:
        args.auto = True; args.manual = True

    input_dir = Path(args.input)

    if args.auto:
        if not input_dir.exists():
            print(f"  ❌ No existe: {input_dir}")
            print(f"     Ejecuta: python 01_video_to_frames.py primero")
            sys.exit(1)
        run_auto_label(input_dir, force=args.force)

    if args.manual:
        run_manual_review(input_dir, args.expression)


if __name__ == "__main__":
    main()
