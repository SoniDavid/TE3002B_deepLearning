# SpongeBob Expression Classifier 🧽
## Reto Ridge Regression (L2) — TE3002B

### Pipeline completo
```
01_video_to_frames.py   ← Descarga episodio YouTube + detecta SpongeBob por color amarillo
02_labeler.py           ← Etiquetado semi-automático con UI visual
03_preprocessing.py     ← 32x32, escala de grises, normalización, dataset.npz
04_ridge_model.py       ← Ridge manual + símplex R⁹ + cross-validation
05_visualizations.py    ← 5 gráficas requeridas por la rúbrica
06_predict.py           ← python 06_predict.py imagen.png → expresión
```

### Instalación
```bash
pip install yt-dlp opencv-python Pillow tqdm numpy matplotlib
sudo apt install ffmpeg   # Linux
brew install ffmpeg        # Mac
```

### Paso a paso

#### 1. Descargar episodio y extraer frames
```bash
# Busca en YouTube: "SpongeBob full episode" o "SpongeBob season 1"
python 01_video_to_frames.py \
  --url "https://youtube.com/watch?v=..." 

# Para múltiples episodios (un URL por línea en urls.txt):
python 01_video_to_frames.py --list urls.txt

# Probar detector en una imagen:
python 01_video_to_frames.py --preview frame.jpg
```
Guarda frames de SpongeBob en `data/frames_raw/`
Yield esperado: **~80% de frames** son SpongeBob (vs 25% con Megumin)

#### 2. Etiquetar expresiones
```bash
python 02_labeler.py
# Presiona 1-0 para la expresión, s=skip, d=delete, z=undo, q=salir

python 02_labeler.py --resume  # continuar sesión anterior
python 02_labeler.py --stats   # ver conteo por expresión
```

#### 3. Preprocesar
```bash
python 03_preprocessing.py
```

#### 4. Entrenar Ridge
```bash
python 04_ridge_model.py
```

#### 5. Visualizaciones
```bash
python 05_visualizations.py
```

#### 6. Predecir
```bash
python 06_predict.py spongebob_test.jpg
```

### 10 Expresiones
| Tecla | Expresión  | Descripción visual                          |
|-------|-----------|---------------------------------------------|
| 1     | happy     | Sonrisa grande, ojos normales               |
| 2     | angry     | Cejas fruncidas, boca torcida               |
| 3     | crying    | Lágrimas masivas, boca hacia abajo          |
| 4     | smug      | Sonrisa de lado, un ojo entrecerrado        |
| 5     | surprised | Ojos enormes, boca abierta en O             |
| 6     | scared    | Pupilas pequeñas, sudor, temblando          |
| 7     | serious   | Boca plana, cejas rectas                    |
| 8     | confused  | Un ojo más grande, cabeza ladeada           |
| 9     | disgusted | Boca fruncida, cejas altas                  |
| 0     | excited   | Ojos brillantes, sonrisa máxima             |
