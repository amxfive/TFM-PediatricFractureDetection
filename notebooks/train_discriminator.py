import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración
DATA_YAML = "/home/allohi2002/Repositories/TFM-PediatricFractureDetection/data/processed_2/ExpDataset_classification"
MODEL_NAME = "yolov8n-cls.pt"
EPOCHS = 50
IMG_SIZE = 224
BATCH_SIZE = 32
PROJECT = "runs/classification"
EXPERIMENT = "discriminator_yolov8n"

def main():
    # Cargar modelo preentrenado de clasificación
    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # Entrenamiento
    print("\nStarting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=EXPERIMENT,
        exist_ok=True,
        verbose=True
    )

    # Validación en test set
    print("\nValidating on test set...")
    val_results = model.val(data=DATA_YAML, split="test")

    # Mostrar métricas principales
    print(f"\nResults:")
    print(f"Top-1 Accuracy: {val_results.top1:.4f}")
    print(f"Top-5 Accuracy: {val_results.top5:.4f}")

    # Guardar métricas en archivo
    metrics_path = Path(PROJECT) / EXPERIMENT / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Top-1 Accuracy: {val_results.top1:.4f}\n")
        f.write(f"Top-5 Accuracy: {val_results.top5:.4f}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Image Size: {IMG_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Best model saved to: {Path(PROJECT) / EXPERIMENT / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))
    main()
