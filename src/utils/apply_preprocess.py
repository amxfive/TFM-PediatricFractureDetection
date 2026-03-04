import cv2
import os
import glob
from tqdm import tqdm # Para ver la barra de progreso

def pipeline_medico_pro(img_path, ruta_salida):
    """Aplica el preprocesado estándar de radiología y guarda en disco."""
    # 1. Carga en Grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False # Error al cargar
    
    # 2. CLAHE (Realce de bordes óseos)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    # 3. Normalización a [0, 255]
    img_final = cv2.normalize(img_clahe, None, 0, 255, cv2.NORM_MINMAX)
    
    # 4. Convertir a 3 canales (YOLO compatible)
    img_rgb = cv2.merge([img_final, img_final, img_final])
    
    # 5. Guardar (Forzamos .png para evitar pérdida de calidad)
    success = cv2.imwrite(ruta_salida, img_rgb)
    return success

def procesar_dataset(carpeta_origen, carpeta_destino):
    """Recorre todas las imágenes y aplica el pipeline."""
    # Tipos de archivos a buscar
    extensiones = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG')
    rutas_imagenes = []
    for ext in extensiones:
        rutas_imagenes.extend(glob.glob(os.path.join(carpeta_origen, "**", ext), recursive=True))
    
    print(f"🔍 Encontradas {len(rutas_imagenes)} imágenes para procesar.")
    
    # Crear carpeta de destino si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        print(f"📁 Carpeta creada: {carpeta_destino}")

    # Bucle de procesamiento con barra de progreso
    conteo_exito = 0
    for img_path in tqdm(rutas_imagenes, desc="Procesando Radiografías"):
        # Mantener el nombre del archivo pero asegurar que sea .png
        nombre_base = os.path.splitext(os.path.basename(img_path))[0]
        ruta_guardado = os.path.join(carpeta_destino, f"{nombre_base}.png")
        
        if pipeline_medico_pro(img_path, ruta_guardado):
            conteo_exito += 1
            
    print(f"\n✅ Proceso finalizado. {conteo_exito}/{len(rutas_imagenes)} imágenes guardadas en {carpeta_destino}")

# ==========================================
# CONFIGURA TUS RUTAS AQUÍ
# ==========================================
CARPETA_INPUT = "data/raw/PediaSHF-DX/03.validation"
CARPETA_OUTPUT = "data/processed/PediaSHF-DX/03.validation"

procesar_dataset(CARPETA_INPUT, CARPETA_OUTPUT)