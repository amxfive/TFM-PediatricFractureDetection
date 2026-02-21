from ultralytics import YOLO

def train():
    # 1. Cargar modelo base pre-entrenado
    # Usamos YOLOv8 Nano por ser el más ligero para pruebas rápidas
    model = YOLO("yolov8n.pt") 

    # 2. Lanzar entrenamiento
    # Asegúrate de tener el dataset GRAZ en la ruta especificada en dataset.yaml
    model.train(
        data="/home/allohi2002/Repositories/TFM-PediatricFractureDetection/data/raw/GRAZPEDWRI-DX/data.yaml", 
        epochs=10, 
        imgsz=640, 
        project="reports", 
        name="graz_benchmark_v1"
    )

if __name__ == "__main__":
    train()