import os
import pydicom
import numpy as np
import cv2

# --- CONFIGURACIÓN DE RUTAS ---
# Ponemos la ruta "padre" para que busque tanto en DICOM_B1 como en DICOM_B2
ROOT_PATH = r"/home/allohi2002/Repositories/TFM-PediatricFractureDetection/data/raw/DatosPacienteVIAMED"
OUTPUT_PATH = r"/home/allohi2002/Repositories/TFM-PediatricFractureDetection/data/processed"

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"🕵️  INICIANDO RASTREO EN: {ROOT_PATH}")
print(f"📂  GUARDANDO EN: {OUTPUT_PATH}")
print("-" * 50)

processed_count = 0
error_count = 0

# --- EL BUCLE "CAMINANTE" (WALKER) ---
for root, dirs, files in os.walk(ROOT_PATH):
    for filename in files:
        # Detectar si es un DICOM (ignorando mayúsculas/minúsculas)
        if filename.lower().endswith(".dcm"):
            
            full_path = os.path.join(root, filename)
            
            try:
                # 1. Leer el archivo DICOM
                ds = pydicom.dcmread(full_path, force=True)
                
                # Verificar si tiene imagen dentro
                if not hasattr(ds, 'pixel_array'):
                    print(f"⚠️  Saltado (sin píxeles): {filename}")
                    continue

                # 2. Obtener matriz de píxeles
                img = ds.pixel_array.astype(np.float32)

                # 3. CORRECCIÓN MONOCROMÁTICA (Hueso Blanco)
                # Si el DICOM dice que el 0 es blanco (MONOCHROME1), lo invertimos
                if hasattr(ds, 'PhotometricInterpretation'):
                    if ds.PhotometricInterpretation == 'MONOCHROME1':
                        img = np.amax(img) - img

                # 4. NORMALIZACIÓN A 16 BITS (Mejora contraste brutalmente)
                img_min = img.min()
                img_max = img.max()
                
                if img_max - img_min != 0:
                    img_normalized = ((img - img_min) / (img_max - img_min)) * 65535.0
                else:
                    img_normalized = img

                img_uint16 = img_normalized.astype(np.uint16)

                # 5. GENERAR NOMBRE ÚNICO
                # Truco: Usamos el nombre de las carpetas padre para que no se repitan
                # Ejemplo ruta: ...\DICOM_B2\DIR000\00000000\00000000.DCM
                parts = root.split(os.sep) 
                
                # Cogemos 'DICOM_B2', 'DIR000', '00000000' para el nombre
                # Si la ruta es muy larga, coge las ultimas 3 carpetas
                prefix = "_".join(parts[-3:]) 
                
                # Nombre final: DICOM_B2_DIR000_00000000_00000000.png
                save_name = f"{prefix}_{filename.replace('.DCM', '').replace('.dcm', '')}.png"
                save_full_path = os.path.join(OUTPUT_PATH, save_name)

                # 6. Guardar
                cv2.imwrite(save_full_path, img_uint16)
                
                processed_count += 1
                # Imprimimos cada 10 archivos para ver que está vivo
                if processed_count % 10 == 0:
                    print(f"✅ Procesados: {processed_count} ... Último: {save_name}")

            except Exception as e:
                print(f"❌ Error en {filename}: {e}")
                error_count += 1

print("-" * 50)
print(f"FIN DEL PROCESO")
print(f"Total imágenes convertidas: {processed_count}")
print(f"Errores encontrados: {error_count}")