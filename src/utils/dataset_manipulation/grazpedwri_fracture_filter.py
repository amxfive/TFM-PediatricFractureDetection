# ==========================================
# BLOQUE 2.5: FILTRADO DE CLASES (SOLO FRACTURAS)
# ==========================================
import os
import glob

def aislar_clase_fractura(dataset_path, clase_original=3, nueva_clase=0):
    print("🧹 Iniciando limpieza de etiquetas... (Solo dejaremos 'fracture')")
    
    carpetas_labels = ['train/labels', 'valid/labels', 'test/labels']
    archivos_modificados = 0
    cajas_fractura_guardadas = 0
    cajas_basura_eliminadas = 0
    
    for carpeta in carpetas_labels:
        ruta_completa = os.path.join(dataset_path, carpeta)
        if not os.path.exists(ruta_completa):
            continue
            
        archivos_txt = glob.glob(os.path.join(ruta_completa, '*.txt'))
        
        for archivo in archivos_txt:
            with open(archivo, 'r') as f:
                lineas = f.readlines()
                
            nuevas_lineas = []
            for linea in lineas:
                partes = linea.strip().split()
                if len(partes) > 0:
                    clase_actual = int(partes[0])
                    if clase_actual == clase_original:
                        # Convertimos el 3 en 0 y mantenemos las coordenadas
                        nueva_linea = f"{nueva_clase} {' '.join(partes[1:])}\n"
                        nuevas_lineas.append(nueva_linea)
                        cajas_fractura_guardadas += 1
                    else:
                        cajas_basura_eliminadas += 1
                        
            # Sobreescribimos el archivo SOLO con las fracturas
            # Si un archivo se queda vacío (no había fracturas en esa imagen), 
            # YOLO lo entenderá como "imagen de fondo" (Background image), lo cual es ideal para reducir Falsos Positivos.
            with open(archivo, 'w') as f:
                f.writelines(nuevas_lineas)
            
            archivos_modificados += 1

    print("-" * 40)
    print(f"✅ Archivos .txt procesados: {archivos_modificados}")
    print(f"🦴 Cajas de fractura mantenidas: {cajas_fractura_guardadas}")
    print(f"🗑️ Cajas de otras clases eliminadas: {cajas_basura_eliminadas}")
    print("-" * 40)

# 1. Ejecutamos la función sobre los datos descomprimidos
LOCAL_DATA_DIR = "data/raw/GRAZPEDWRI-DX" # Ajusta si tu ruta es distinta
#aislar_clase_fractura(LOCAL_DATA_DIR, clase_original=3, nueva_clase=0)

# 2. Reescribimos el data.yaml para que YOLO sepa que ahora solo hay 1 clase
nuevo_yaml = f"""
train: {LOCAL_DATA_DIR}/train/images
val: {LOCAL_DATA_DIR}/valid/images
test: {LOCAL_DATA_DIR}/test/images

nc: 1
names: ['fracture']
"""

yaml_path = f"{LOCAL_DATA_DIR}/data.yaml"
with open(yaml_path, "w") as f:
    f.write(nuevo_yaml)

print(f"📄 Archivo {yaml_path} reescrito correctamente para 1 sola clase.")