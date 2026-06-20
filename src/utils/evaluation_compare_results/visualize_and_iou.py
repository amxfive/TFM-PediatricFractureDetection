#!/usr/bin/env python3
"""Visualiza anotaciones y calcula IoU humano-humano y humano-IA."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evaluation_common import (
    ANNOTATION_DIR,
    EVALUATORS,
    PROJECT_ROOT,
    optimal_iou_match,
    pair_order,
)


IMG_DIR = PROJECT_ROOT / "data" / "EvalDatasetProperID"
OUT_DIR = PROJECT_ROOT / "src" / "evaluation" / "results" / "images" / "iou_human_pairs"
CSV_OUT = PROJECT_ROOT / "src" / "evaluation" / "results" / "tables" / "iou_per_image_human_pairs.csv"

JSONS = {
    evaluator.key: ANNOTATION_DIR / evaluator.filename
    for evaluator in EVALUATORS
}

EVAL_ORDER = [evaluator.key for evaluator in EVALUATORS]
HUMAN_EVALUATORS = [evaluator.key for evaluator in EVALUATORS if evaluator.kind == "human"]
PAIR_ORDER = pair_order()

COLORS = {
    "Usuario_Control": (0, 180, 0),
    "R1_Radiologia": (0, 0, 255),
    "Experto_Radiologo": (255, 120, 0),
    "E5": (200, 0, 200),
    "E6": (0, 140, 200),
    "E7": (180, 0, 100),
    "ModeloEspecialista": (0, 180, 180),
}

LABELS = {
    evaluator.key: evaluator.display_name
    for evaluator in EVALUATORS
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_filename(item):
    """Obtiene un nombre comparable desde exportaciones humanas o de IA."""
    raw_name = (
        item.get("file_upload")
        or item.get("data", {}).get("image")
        or str(item.get("id", ""))
    )
    return Path(str(raw_name).replace("\\", "/")).name


def extract_boxes_all(data):
    """Extrae bounding boxes agrupadas por nombre de imagen."""
    result = {}
    for item in data:
        filename = extract_filename(item)
        boxes = []

        for annotation in item.get("annotations", []):
            for annotation_result in annotation.get("result", []):
                value = annotation_result.get("value", {})
                original_width = annotation_result.get("original_width", 1)
                original_height = annotation_result.get("original_height", 1)

                boxes.append({
                    "x": value["x"] / 100 * original_width,
                    "y": value["y"] / 100 * original_height,
                    "w": value["width"] / 100 * original_width,
                    "h": value["height"] / 100 * original_height,
                    "orig_w": original_width,
                    "orig_h": original_height,
                })

        result[filename] = boxes

    return result


def scale_boxes(boxes, image_width, image_height):
    scaled = []
    for box in boxes:
        scaled.append({
            "x": box["x"] * image_width / box["orig_w"],
            "y": box["y"] * image_height / box["orig_h"],
            "w": box["w"] * image_width / box["orig_w"],
            "h": box["h"] * image_height / box["orig_h"],
        })
    return scaled


def draw_boxes(image, boxes, color, thickness=2):
    for box in boxes:
        x1 = int(box["x"])
        y1 = int(box["y"])
        x2 = int(box["x"] + box["w"])
        y2 = int(box["y"] + box["h"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def calculate_pairwise_iou(boxes_by_evaluator):
    values = {}
    for evaluator1, evaluator2 in PAIR_ORDER:
        key = f"IoU_{evaluator1}_{evaluator2}"
        values[key] = optimal_iou_match(
            boxes_by_evaluator[evaluator1],
            boxes_by_evaluator[evaluator2],
        )
    return values


def create_result_canvas(image, filename, iou_values):
    """Anade un panel lateral para no tapar cajas sobre la radiografia."""
    panel_width = 440
    line_height = 21
    required_height = 75 + len(EVAL_ORDER) * line_height
    required_height += 45 + len(PAIR_ORDER) * line_height
    canvas_height = max(image.shape[0], required_height)

    canvas = np.full(
        (canvas_height, image.shape[1] + panel_width, 3),
        (24, 24, 24),
        dtype=np.uint8,
    )
    canvas[:image.shape[0], :image.shape[1]] = image

    panel_x = image.shape[1] + 16
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        canvas,
        filename,
        (panel_x, 28),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Anotaciones",
        (panel_x, 58),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y = 82
    for evaluator in EVAL_ORDER:
        color = COLORS[evaluator]
        cv2.rectangle(canvas, (panel_x, y - 10), (panel_x + 14, y + 3), color, -1)
        cv2.putText(
            canvas,
            LABELS[evaluator],
            (panel_x + 24, y + 2),
            font,
            0.43,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += line_height

    y += 20
    cv2.putText(
        canvas,
        "IoU por parejas",
        (panel_x, y),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y += 26

    for evaluator1, evaluator2 in PAIR_ORDER:
        key = f"IoU_{evaluator1}_{evaluator2}"
        value = iou_values[key]
        value_text = "N/A" if value is None else f"{value:.3f}"
        pair_label = f"{LABELS[evaluator1]} vs {LABELS[evaluator2]}"

        cv2.putText(
            canvas,
            f"{pair_label}: {value_text}",
            (panel_x, y),
            font,
            0.39,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        y += line_height

    return canvas


def validate_evaluator_images(all_boxes):
    """Avisa si algun evaluador no contiene exactamente las mismas imagenes."""
    expected = set(all_boxes[HUMAN_EVALUATORS[0]])
    for evaluator in EVAL_ORDER[1:]:
        current = set(all_boxes[evaluator])
        missing = expected - current
        extra = current - expected
        if missing or extra:
            print(
                f"[WARNING] {evaluator}: "
                f"{len(missing)} imagenes ausentes, {len(extra)} adicionales"
            )
    return sorted(expected)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

    all_boxes = {}
    for evaluator, path in JSONS.items():
        data = load_json(path)
        all_boxes[evaluator] = extract_boxes_all(data)
        box_count = sum(len(boxes) for boxes in all_boxes[evaluator].values())
        print(
            f"  {LABELS[evaluator]}: "
            f"{len(all_boxes[evaluator])} imagenes, {box_count} cajas"
        )

    filenames = validate_evaluator_images(all_boxes)
    print(f"\nTotal imagenes: {len(filenames)}")
    print(f"Parejas IoU por imagen: {len(PAIR_ORDER)}")

    csv_rows = []

    for index, filename in enumerate(filenames, start=1):
        image_path = IMG_DIR / filename
        if not image_path.exists():
            print(f"  [SKIP] {filename} no encontrada")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [ERROR] No se pudo leer {filename}")
            continue

        image_height, image_width = image.shape[:2]
        boxes_by_evaluator = {}

        for evaluator in EVAL_ORDER:
            raw_boxes = all_boxes[evaluator].get(filename, [])
            boxes_by_evaluator[evaluator] = scale_boxes(
                raw_boxes,
                image_width,
                image_height,
            )
            draw_boxes(
                image,
                boxes_by_evaluator[evaluator],
                COLORS[evaluator],
                thickness=2,
            )

        iou_values = calculate_pairwise_iou(boxes_by_evaluator)
        result_image = create_result_canvas(image, filename, iou_values)

        output_path = OUT_DIR / filename
        if not cv2.imwrite(str(output_path), result_image):
            print(f"  [ERROR] No se pudo guardar {output_path}")
            continue

        row = {"image": filename}
        for key, value in iou_values.items():
            row[key] = "" if value is None else round(value, 4)
        csv_rows.append(row)

        if index % 25 == 0 or index == len(filenames):
            print(f"  Procesadas {index}/{len(filenames)}")

    if not csv_rows:
        print("\nNo se genero ningun resultado.")
        return

    csv_keys = ["image"] + [
        f"IoU_{evaluator1}_{evaluator2}"
        for evaluator1, evaluator2 in PAIR_ORDER
    ]
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV guardado: {CSV_OUT} ({len(csv_rows)} filas)")
    print(f"Imagenes guardadas en: {OUT_DIR}/ ({len(csv_rows)} archivos)")
    print("\n--- Resumen IoU promedio ---")

    for evaluator1, evaluator2 in PAIR_ORDER:
        key = f"IoU_{evaluator1}_{evaluator2}"
        values = [
            row[key]
            for row in csv_rows
            if row[key] != ""
        ]
        average = float(np.mean(values)) if values else float("nan")
        label = f"{LABELS[evaluator1]} vs {LABELS[evaluator2]}"
        print(f"  {label}: {average:.4f}")


if __name__ == "__main__":
    main()
