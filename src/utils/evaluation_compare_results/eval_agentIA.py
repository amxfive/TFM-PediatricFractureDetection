import os
import json
import uuid
from datetime import datetime
from ultralytics import YOLO

def generar_json_evaluacion(model_path, images_path, output_file):
    # 1. Cargar el modelo
    model = YOLO(model_path)
    
    # 2. Obtener imágenes del test set
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    ls_results = []
    
    print(f"🚀 Procesando {len(image_files)} imágenes para evaluación...")

    for img_name in image_files:
        full_path = os.path.join(images_path, img_name)
        
        # 3. Realizar predicción
        # Usamos un conf bajo para captar más variabilidad en el estudio de distancias
        results = model.predict(source=full_path, conf=0.3, imgsz=1024, verbose=False)
        
        for r in results:
            # Estructura base de Label Studio por cada imagen
            task_entry = {
                "id": r.path.split('/')[-1], # Usamos el nombre del archivo como ID
                "annotations": [{
                    "completed_by": 99, # ID reservado para la IA
                    "result": [],
                    "lead_time": r.speed['inference'] / 1000.0, # Tiempo convertido a segundos
                    "created_at": datetime.utcnow().isoformat() + "Z"
                }],
                "file_upload": img_name,
                "data": {"image": f"/data/upload/ia/{img_name}"}
            }

            # 4. Convertir cajas de YOLO (0-1) a Label Studio (0-100%)
            if len(r.boxes) > 0:
                for box in r.boxes:
                    # YOLO xywh (normalizado 0-1) -> Label Studio (0-100)
                    # Ojo: YOLO xywh es centro, Label Studio es esquina superior izquierda (top-left)
                    x_norm, y_norm, w_norm, h_norm = box.xywhn.tolist()[0]
                    
                    # Convertir a top-left (esquina superior izquierda)
                    ls_x = (x_norm - (w_norm / 2)) * 100
                    ls_y = (y_norm - (h_norm / 2)) * 100
                    ls_w = w_norm * 100
                    ls_h = h_norm * 100

                    # Añadir el resultado al JSON
                    task_entry["annotations"][0]["result"].append({
                        "original_width": r.orig_shape[1],
                        "original_height": r.orig_shape[0],
                    "image_rotation": 0,
                    "value": {
                        "x": ls_x,
                        "y": ls_y,
                        "width": ls_w,
                        "height": ls_h,
                        "rotation": 0,
                        "rectanglelabels": ["0"] # Clase 0: Fractura
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels"
                })
            
            ls_results.append(task_entry)

    # 5. Guardar archivo formateado
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ls_results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Evaluación IA completada. Archivo guardado como: {output_file}")

# --- CONFIGURACIÓN ---
# Ajusta estas rutas a tu entorno WSL
PATH_MODELO = 'models_weights/E7_yoloV11n_optA.pt'
PATH_IMAGENES_TEST = 'EvaluacionAgentes/EvalDatasetProperID'
NOMBRE_SALIDA = 'IA_Evaluation_E7_yoloV11n_optA.json'

generar_json_evaluacion(PATH_MODELO, PATH_IMAGENES_TEST, NOMBRE_SALIDA)