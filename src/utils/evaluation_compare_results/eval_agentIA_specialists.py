#!/usr/bin/env python3
"""Generate Label Studio JSON for the specialist-agent architecture.

Pipeline:
    image -> YOLO  /router -> selected YOLO specialist -> Label Studio JSON

The JSON structure intentionally mirrors eval_agentIA.py so the existing IoU and
efficiency scripts can consume the output with minimal or no changes.
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


DEFAULT_SPECIALISTS = {
    "supracondylar": "models_weights/especialist_architectures/esp_pediaSHF.pt",
    "wrist": "models_weights/especialist_architectures/esp_grazpedwri.pt",
    "ulna_radius": "models_weights/especialist_architectures/esp_pediURF.pt",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def parse_specialist_map(raw_items: list[str] | None) -> dict[str, str]:
    """Parse class=path CLI overrides and merge them with defaults."""
    mapping = dict(DEFAULT_SPECIALISTS)
    if not raw_items:
        return mapping

    for item in raw_items:
        if "=" not in item:
            raise ValueError(
                f"Invalid specialist mapping '{item}'. Use class_name=/path/to/model.pt"
            )
        class_name, model_path = item.split("=", 1)
        mapping[normalize_class_name(class_name)] = model_path
    return mapping


def validate_paths(router_path: Path, specialist_map: dict[str, str], images_path: Path) -> None:
    missing = []
    if not router_path.exists():
        missing.append(f"router model: {router_path}")
    if not images_path.exists():
        missing.append(f"images path: {images_path}")

    for class_name, model_path in specialist_map.items():
        path = Path(model_path)
        if not path.exists():
            missing.append(f"specialist for '{class_name}': {path}")

    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing required input(s):\n  - {formatted}")


def list_images(images_path: Path) -> list[Path]:
    return sorted(
        path for path in images_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_specialists(specialist_map: dict[str, str]) -> dict[str, YOLO]:
    specialists = {}
    for class_name, model_path in specialist_map.items():
        print(f"[Load] Specialist '{class_name}': {model_path}")
        specialists[class_name] = YOLO(model_path)
    return specialists


def route_image(router: YOLO, image_path: Path, imgsz: int, device: str | None):
    kwargs = {"source": str(image_path), "imgsz": imgsz, "verbose": False}
    if device:
        kwargs["device"] = device

    result = router.predict(**kwargs)[0]
    if result.probs is None:
        raise ValueError(
            f"Router output for {image_path.name} has no classification probabilities. "
            "Use a YOLO classification model, e.g. a *-cls.pt model."
        )

    class_id = int(result.probs.top1)
    class_conf = float(result.probs.top1conf)
    raw_name = result.names[class_id]
    class_name = normalize_class_name(str(raw_name))
    router_time_s = float(result.speed.get("inference", 0.0)) / 1000.0

    return class_name, class_conf, router_time_s


def detect_with_specialist(
    specialist: YOLO,
    image_path: Path,
    confidence: float,
    imgsz: int,
    device: str | None,
):
    kwargs = {
        "source": str(image_path),
        "conf": confidence,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    return specialist.predict(**kwargs)[0]


def result_to_label_studio(task_entry: dict, result) -> None:
    """Append YOLO boxes to an existing Label Studio task entry."""
    if len(result.boxes) == 0:
        return

    for box in result.boxes:
        x_norm, y_norm, w_norm, h_norm = box.xywhn.tolist()[0]

        task_entry["annotations"][0]["result"].append({
            "original_width": result.orig_shape[1],
            "original_height": result.orig_shape[0],
            "image_rotation": 0,
            "value": {
                "x": (x_norm - (w_norm / 2)) * 100,
                "y": (y_norm - (h_norm / 2)) * 100,
                "width": w_norm * 100,
                "height": h_norm * 100,
                "rotation": 0,
                "rectanglelabels": ["0"],
            },
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
        })


def build_task_entry(img_name: str, lead_time_s: float, completed_by: int) -> dict:
    return {
        "id": img_name,
        "annotations": [{
            "completed_by": completed_by,
            "result": [],
            "lead_time": lead_time_s,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }],
        "file_upload": img_name,
        "data": {"image": f"/data/upload/ia/{img_name}"},
    }


def generate_specialist_evaluation_json(
    router_path: Path,
    images_path: Path,
    output_file: Path,
    specialist_map: dict[str, str],
    confidence: float,
    classifier_imgsz: int,
    detector_imgsz: int,
    completed_by: int,
    device: str | None,
) -> None:
    validate_paths(router_path, specialist_map, images_path)

    print(f"[Load] Router: {router_path}")
    router = YOLO(str(router_path))
    specialists = load_specialists(specialist_map)

    image_files = list_images(images_path)
    print(f"\n[Eval] Processing {len(image_files)} images from {images_path}")

    ls_results = []
    route_counts = Counter()
    unknown_routes = Counter()

    for idx, image_path in enumerate(image_files, start=1):
        routed_class, routed_conf, router_time_s = route_image(
            router=router,
            image_path=image_path,
            imgsz=classifier_imgsz,
            device=device,
        )

        specialist = specialists.get(routed_class)
        if specialist is None:
            unknown_routes[routed_class] += 1
            raise KeyError(
                f"Router predicted class '{routed_class}' for {image_path.name}, "
                f"but no specialist model is configured for that class. "
                f"Available classes: {sorted(specialists)}"
            )

        route_counts[routed_class] += 1
        result = detect_with_specialist(
            specialist=specialist,
            image_path=image_path,
            confidence=confidence,
            imgsz=detector_imgsz,
            device=device,
        )

        detector_time_s = float(result.speed.get("inference", 0.0)) / 1000.0
        total_time_s = router_time_s + detector_time_s

        task_entry = build_task_entry(
            img_name=image_path.name,
            lead_time_s=total_time_s,
            completed_by=completed_by,
        )
        result_to_label_studio(task_entry, result)
        ls_results.append(task_entry)

        if idx % 25 == 0 or idx == len(image_files):
            print(
                f"  {idx}/{len(image_files)} processed "
                f"(last route: {routed_class}, router_conf={routed_conf:.3f})"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(ls_results, f, indent=4, ensure_ascii=False)

    print(f"\n[Done] Specialist-agent evaluation saved to: {output_file}")
    print("[Routes] Images per specialist:")
    for class_name, count in sorted(route_counts.items()):
        print(f"  - {class_name}: {count}")

    if unknown_routes:
        print("[Warning] Unknown routes:")
        for class_name, count in sorted(unknown_routes.items()):
            print(f"  - {class_name}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate router + specialist YOLO detectors and export Label Studio JSON."
    )
    parser.add_argument(
        "--router-model",
        default="models_weights/classifier_models/router.pt",
        help="Path to the YOLO classification router model.",
    )
    parser.add_argument(
        "--images",
        default="data/processed_2/EvalDatasetProperID",
        help="Directory with evaluation images.",
    )
    parser.add_argument(
        "--output",
        default="src/evaluation/annotation_json/IA_Evaluation_specialist_agents.json",
        help="Output Label Studio JSON file.",
    )
    parser.add_argument(
        "--specialist",
        action="append",
        help=(
            "Override or add a specialist mapping as class_name=model.pt. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Detector confidence threshold.",
    )
    parser.add_argument(
        "--classifier-imgsz",
        type=int,
        default=224,
        help="Image size for the classifier/router.",
    )
    parser.add_argument(
        "--detector-imgsz",
        type=int,
        default=1024,
        help="Image size for specialist detectors.",
    )
    parser.add_argument(
        "--completed-by",
        type=int,
        default=98,
        help="Label Studio annotator ID reserved for this IA architecture.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Ultralytics device argument, e.g. 0, cpu, cuda:0.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    specialist_map = parse_specialist_map(args.specialist)
    generate_specialist_evaluation_json(
        router_path=Path(args.router_model),
        images_path=Path(args.images),
        output_file=Path(args.output),
        specialist_map=specialist_map,
        confidence=args.conf,
        classifier_imgsz=args.classifier_imgsz,
        detector_imgsz=args.detector_imgsz,
        completed_by=args.completed_by,
        device=args.device,
    )


if __name__ == "__main__":
    main()
