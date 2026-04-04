"""
Backend FastAPI para el sistema de detección de fracturas pediátricas.
Carga el modelo YOLOv8 una sola vez en el arranque y expone:
  - POST /predict  → CLAHE → YOLOv8 → detecciones JSON + imagen anotada base64
  - POST /xai      → CLAHE → EigenCAM → heatmap base64
  - GET  /health   → comprobación de estado
"""

from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ultralytics import YOLO
import hashlib

# ---------------------------------------------------------------------------
# Ruta del modelo (relativa al directorio de trabajo = raíz del repositorio)
# ---------------------------------------------------------------------------
MODEL_PATH = "E6_test.pt"

_model: YOLO | None = None
_target_layers: list | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar y lo libera al cerrar."""
    global _model, _target_layers
    print(f"[Backend] Cargando modelo desde {MODEL_PATH} …")
    _model = YOLO(str(MODEL_PATH))
    _target_layers = [_model.model.model[-2]]  # capa SPPF, igual que xai.py
    print("[Backend] Modelo cargado correctamente.")
    yield
    print("[Backend] Cerrando backend.")


app = FastAPI(
    title="Viamed IA — API de Detección de Fracturas",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocessing_pipeline(img_bytes: bytes):
    """
    Versión para API: Recibe bytes, procesa en memoria y devuelve un array 
    listo para que YOLO realice la inferencia.
    """
    # 1. Decodificar bytes a imagen (OpenCV lee desde memoria)
    # Usamos IMREAD_UNCHANGED para mantener la profundidad (16-bit si existe)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    
    if img is None:
        raise ValueError("No se pudo decodificar la imagen. Formato no soportado.")

    # 2. Normalización Médica (16-bit -> 8-bit)
    # Crucial para mantener la compatibilidad con el entrenamiento
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    # 3. Denoising (Filtro Bilateral) - Preserva bordes de fractura
    img_denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 4. CLAHE (Realce de contraste adaptativo local)
    # Mejora la visibilidad de líneas de fractura sutiles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_denoised)
    
    # 5. Formato YOLO (3 canales RGB)
    # YOLOv8 espera tensores de 3 canales
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    # Devolvemos el array de la imagen procesada (no guardamos en disco)
    return img_rgb

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...), confidence: float = Form(0.40)):
    img_bytes = await file.read()
    # Calcular el HASH SHA-256 de los bytes recibidos
    file_hash = hashlib.sha256(img_bytes).hexdigest()
    
    print(f"\n[AUDITORÍA] Hash recibido: {file_hash}")
    print(f"[AUDITORÍA] Tamaño: {len(img_bytes)} bytes\n")
    
    # IMPORTANTE: Volver a
    # 1. Imagen para la IA (con CLAHE y 8-bit)
    img_para_ia = preprocessing_pipeline(img_bytes) 
    
    # 2. Inferencia YOLOv8
    results = _model.predict(img_para_ia, conf=confidence)
    boxes = results[0].boxes

    detections = []
    for box in boxes:
        detections.append({
            "confidence": float(box.conf),
            "xyxy": box.xyxy.tolist(), # Coordenadas en píxeles para dibujar
            "class_name": _model.names[int(box.cls)]
        })

    # Devolvemos solo los datos numéricos
    return {"num_detections": len(detections), "detections": detections}
