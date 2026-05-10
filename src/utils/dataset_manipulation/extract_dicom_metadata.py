import os
import pydicom
import pandas as pd
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS (Estilo WSL) ---
ROOT_PATH = "data/raw/DatosPacienteVIAMED"
OUTPUT_FILE = "data/raw/DatosPacienteVIAMED/metadatos_pacientes_viamed.csv"

print(f"🔍 Buscando metadatos en: {ROOT_PATH}")

metadata_list = []

# 1. Escaneamos los archivos primero
archivos_dcm = []
for root, dirs, files in os.walk(ROOT_PATH):
    for file in files:
        if file.lower().endswith(".dcm"):
            archivos_dcm.append(os.path.join(root, file))

print(f"📋 Se han encontrado {len(archivos_dcm)} archivos. Extrayendo información...")

# 2. Procesamos cada archivo
for path in tqdm(archivos_dcm):
    try:
        # stop_before_pixels=True hace que el script sea 100 veces más rápido
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        
        info = {
            "Ruta_Relativa": os.path.relpath(path, ROOT_PATH),
            "Nombre_Archivo": os.path.basename(path),
            
            "Paciente_ID": ds.get("PatientID", "Anónimo"),
            "Paciente_Sexo": ds.get("PatientSex", "N/A"),
            "Paciente_Edad": ds.get("PatientAge", "N/A"),
            "Fecha_Estudio": ds.get("StudyDate", "N/A"),
            
            "Modalidad": ds.get("Modality", "N/A"),
            "Fabricante": ds.get("Manufacturer", "N/A"),
            "Modelo_Equipo": ds.get("ManufacturerModelName", "N/A"),
            "Parte_Cuerpo": ds.get("BodyPartExamined", "N/A"),
            "Posicion_Vista": ds.get("ViewPosition", "N/A"),
            
            # --- DATOS DE RESOLUCIÓN ---
            "Filas": ds.get("Rows", "N/A"),
            "Columnas": ds.get("Columns", "N/A"),
            "Bits_Almacenados": ds.get("BitsStored", "N/A"),
            "Interpretacion_Fotométrica": ds.get("PhotometricInterpretation", "N/A"),
        }
        
        metadata_list.append(info)
        
    except Exception as e:
        continue

# 3. Guardamos todo en un CSV
if metadata_list:
    df = pd.DataFrame(metadata_list)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig') # utf-8-sig para que Excel lea bien las tildes
    print(f"\n✅ Proceso completado.")
    print(f"📊 Se han extraído datos de {len(df)} archivos.")
    print(f"📂 Archivo generado: {os.getcwd()}/{OUTPUT_FILE}")
else:
    print("❌ No se pudo extraer información de ningún archivo.")