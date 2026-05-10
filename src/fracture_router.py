from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

CLASSIFIER_PATH = BASE_DIR / "runs" / "classification" / "discriminator_yolov8n" / "weights" / "best.pt"

SPECIALIST_DIRS = {
    "wrist": BASE_DIR / "models_weights" / "specialist_wrist" / "best.pt",
    "ulna_radius": BASE_DIR / "models_weights" / "specialist_ulna_radius" / "best.pt",
    "supracondylar": BASE_DIR / "models_weights" / "specialist_supracondylar" / "best.pt",
}


class FractureRouter:
    def __init__(self, classifier_path=None, specialist_dirs=None, device="cpu"):
        classifier_path = classifier_path or CLASSIFIER_PATH
        specialist_dirs = specialist_dirs or SPECIALIST_DIRS

        print(f"Loading discriminator: {classifier_path}")
        self.discriminator = YOLO(str(classifier_path))

        self.specialists = {}
        for body_part, path in specialist_dirs.items():
            if path.exists():
                print(f"Loading specialist ({body_part}): {path}")
                self.specialists[body_part] = YOLO(str(path))
            else:
                print(f"Warning: {body_part} specialist not found at {path}")

    def predict(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        body_part = None
        confidence_router = 0.0

        cls_result = self.discriminator.predict(image_path, verbose=False)[0]
        if cls_result.probs is not None:
            top1_idx = int(cls_result.probs.top1)
            body_part = cls_result.names[top1_idx]
            confidence_router = float(cls_result.probs.top1conf)

        detections = []
        plot_img = None

        specialist = self.specialists.get(body_part)
        if specialist is not None:
            det_result = specialist.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )[0]
            if det_result.boxes is not None:
                detections = det_result.boxes.data.tolist()
            plot_img = det_result.plot()

        return {
            "body_part": body_part,
            "router_confidence": round(confidence_router, 4),
            "num_detections": len(detections),
            "detections": detections,
            "plot": plot_img,
        }

    def predict_bytes(self, image_bytes, conf_threshold=0.25, iou_threshold=0.45):
        import numpy as np
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        body_part = None
        confidence_router = 0.0

        cls_result = self.discriminator.predict(img, verbose=False)[0]
        if cls_result.probs is not None:
            top1_idx = int(cls_result.probs.top1)
            body_part = cls_result.names[top1_idx]
            confidence_router = float(cls_result.probs.top1conf)

        detections = []
        plot_img = None

        specialist = self.specialists.get(body_part)
        if specialist is not None:
            det_result = specialist.predict(
                img,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )[0]
            if det_result.boxes is not None:
                detections = det_result.boxes.data.tolist()
            plot_img = det_result.plot()

        return {
            "body_part": body_part,
            "router_confidence": round(confidence_router, 4),
            "num_detections": len(detections),
            "detections": detections,
            "plot": plot_img,
        }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python fracture_router.py <image_path> [conf_threshold]")
        sys.exit(1)

    img_path = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    router = FractureRouter()
    result = router.predict(img_path, conf_threshold=conf)

    print(json.dumps({
        "body_part": result["body_part"],
        "router_confidence": result["router_confidence"],
        "num_detections": result["num_detections"],
        "detections": result["detections"],
    }, indent=2))
