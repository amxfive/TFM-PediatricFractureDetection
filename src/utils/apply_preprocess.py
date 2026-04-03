import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def pipeline_medico_pro(img_path, ruta_salida):
    """
    Aplica el preprocesado estándar de radiología y guarda en disco.
    Optimizado para preservar bordes de fractura y mantener compatibilidad con YOLO.
    """
    # 1. Carga Robusta
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f"⚠️ Error al leer: {img_path}")
        return False
    
    # 2. Normalización Médica Segura (De 16-bit a 8-bit)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    # 3. Denoising Inteligente (Filtro Bilateral)
    img_denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 4. CLAHE (Realce de bordes óseos)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_denoised)
    
    # 5. Convertir a 3 canales (YOLO compatible)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    # 6. Guardar (Forzamos .png para evitar artefactos de compresión JPG)
    # Extraemos la ruta sin extensión y forzamos que acabe en .png 
    # por si el original era .jpg
    base_path, _ = os.path.splitext(ruta_salida)
    ruta_salida_png = f"{base_path}.png"
    
    success = cv2.imwrite(ruta_salida_png, img_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    return success

def procesar_dataset_arbol(carpeta_origen, carpeta_destino):
    """
    Recorre el directorio origen, clona la estructura en el destino,
    procesa las imágenes con el pipeline médico y copia los demás archivos (.txt).
    """
    # 1. Recopilar todos los archivos para la barra de progreso
    all_files = []
    for root, dirs, files in os.walk(carpeta_origen):
        for file in files:
            all_files.append(os.path.join(root, file))
            
    print(f"📦 Encontrados {len(all_files)} archivos. Replicando estructura y procesando...")

    conteo_exito = 0
    # 2. Iterar sobre cada archivo
    for src_path in tqdm(all_files, desc="Procesando Dataset"):
        # Calcular la ruta relativa (ej: 'train/images/001.png' o 'train/labels/001.txt')
        rel_path = os.path.relpath(src_path, carpeta_origen)
        # Crear la ruta final absoluta
        dest_path = os.path.join(carpeta_destino, rel_path)
        
        # Crear la carpeta padre en el destino si no existe
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # 3. Decidir qué hacer según el tipo de archivo
        ext = src_path.lower().split('.')[-1]
        
        if ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'dcm']:
            # Es una imagen: aplicar tu pipeline
            if pipeline_medico_pro(src_path, dest_path):
                conteo_exito += 1
        else:
            # Es un .txt (o yaml, etc.): copiar tal cual
            shutil.copy2(src_path, dest_path)
            conteo_exito += 1 

    print(f"\n✅ Proceso finalizado. Archivos gestionados: {conteo_exito}/{len(all_files)}")
    print(f"📁 Dataset final listo en: {carpeta_destino}")

# ==========================================
# CONFIGURA TUS RUTAS AQUÍ
# ==========================================
# OJO: Ahora apuntas a la carpeta RAÍZ del dataset, no solo a "images"
CARPETA_INPUT = "data/raw/PediaSHF-DX-yolo-split"
CARPETA_OUTPUT = "data/processed_2/PediaSHF-DX-yolo-split"

if __name__ == '__main__':
    procesar_dataset_arbol(CARPETA_INPUT, CARPETA_OUTPUT)