#!/usr/bin/env python3
"""Dibuja bounding boxes de cada evaluador y calcula IoU por imagen."""

import json
import csv
import numpy as np
import cv2
from pathlib import Path

IMG_DIR = Path('data/EvalDatasetProperID')
OUT_DIR = Path('data/EvalEvaluatedImages')
CSV_OUT = Path('src/evaluation/results/iou_per_image.csv')

COLORS = {
    'Humano': (0, 180, 0),
    'IA_E3': (200, 0, 0),
    'IA_E6': (0, 140, 200),
    'IA_E7': (180, 0, 180),
}

EVAL_ORDER = ['Humano', 'IA_E3', 'IA_E6', 'IA_E7']

LABELS = {
    'Humano': 'Humano',
    'IA_E3': 'IA E3 YOLOv8n',
    'IA_E6': 'IA E6 YOLOv8m',
    'IA_E7': 'IA E7 YOLOv11n',
}

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_boxes_all(data):
    """Extrae bounding boxes keyed by filename (sin hash uuid)."""
    result = {}
    for item in data:
        fn = (item.get('file_upload')
              or item.get('data', {}).get('image', '')
              or str(item.get('id', '')))
        annotations = item.get('annotations', [])
        boxes = []
        for ann in annotations:
            for r in ann.get('result', []):
                value = r.get('value', {})
                orig_w = r.get('original_width', 1)
                orig_h = r.get('original_height', 1)
                x = value['x'] / 100 * orig_w
                y = value['y'] / 100 * orig_h
                w = value['width'] / 100 * orig_w
                h = value['height'] / 100 * orig_h
                boxes.append({'x': x, 'y': y, 'w': w, 'h': h,
                              'orig_w': orig_w, 'orig_h': orig_h})
        result[fn] = boxes
    return result

def iou(b1, b2):
    x1 = max(b1['x'], b2['x'])
    y1 = max(b1['y'], b2['y'])
    x2 = min(b1['x'] + b1['w'], b2['x'] + b2['w'])
    y2 = min(b1['y'] + b1['h'], b2['y'] + b2['h'])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = b1['w'] * b1['h']
    area2 = b2['w'] * b2['h']
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0.0

def best_iou_match(boxes1, boxes2):
    if not boxes1 and not boxes2:
        return None
    if not boxes1 or not boxes2:
        return 0.0
    total = 0.0
    used = set()
    for b1 in boxes1:
        best_val = 0.0
        best_j = -1
        for j, b2 in enumerate(boxes2):
            if j not in used:
                val = iou(b1, b2)
                if val > best_val:
                    best_val = val
                    best_j = j
        if best_j >= 0:
            total += best_val
            used.add(best_j)
    n = max(len(boxes1), len(boxes2))
    return total / n if n > 0 else 0.0

def draw_boxes(img, boxes, color, thickness=2):
    for b in boxes:
        x1 = int(b['x'])
        y1 = int(b['y'])
        x2 = int(b['x'] + b['w'])
        y2 = int(b['y'] + b['h'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_legend(img, evaluator_names):
    h, w = img.shape[:2]
    box_w = 220
    box_h = 20 + len(evaluator_names) * 22
    x0 = 10
    y0 = 10

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Evaluadores:', (x0 + 5, y0 + 16), font, 0.45, (255, 255, 255), 1)

    for i, name in enumerate(evaluator_names):
        cy = y0 + 18 + i * 22 + 10
        cv2.rectangle(img, (x0 + 5, cy - 4), (x0 + 17, cy + 6), COLORS[name], -1)
        cv2.putText(img, LABELS.get(name, name), (x0 + 22, cy + 6), font, 0.4, (255, 255, 255), 1)

def draw_filename(img, filename):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, filename, (10, h - 10), font, 0.5, (255, 255, 255), 1)

def calc_pairwise_iou(boxes_dict):
    names = list(boxes_dict.keys())
    ious = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f'IoU_{names[i]}_{names[j]}'
            val = best_iou_match(boxes_dict[names[i]], boxes_dict[names[j]])
            ious[key] = val if val is not None else 1.0
    return ious

def main():
    JSONS = {
        'Humano': 'src/evaluation/annotation_json/Control_User_Evaluation_Yasmina_Moreira.json',
        'IA_E3': 'src/evaluation/annotation_json/IA_Evaluation_E3_yoloV8n_optA.json',
        'IA_E6': 'src/evaluation/annotation_json/IA_Evaluation_E6_yoloV8m_optA.json',
        'IA_E7': 'src/evaluation/annotation_json/IA_Evaluation_E7_yoloV11n_optA.json',
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    all_boxes = {}
    for name, path in JSONS.items():
        data = load_json(path)
        all_boxes[name] = extract_boxes_all(data)
        print(f'  {name}: {len(all_boxes[name])} imágenes, {sum(len(v) for v in all_boxes[name].values())} cajas')

    # Get all filenames in evaluation order
    filenames = list(all_boxes['Humano'].keys())
    filenames.sort()
    print(f'\nTotal imágenes únicas: {len(filenames)}')

    # CSV rows
    csv_rows = []
    csv_keys = []

    for i, fn in enumerate(filenames):
        img_path = IMG_DIR / fn
        if not img_path.exists():
            print(f'  [SKIP] {fn} no encontrada')
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f'  [ERROR] No se pudo leer {fn}')
            continue

        img_h, img_w = img.shape[:2]

        boxes_by_eval = {}
        for name in EVAL_ORDER:
            raw_boxes = all_boxes[name].get(fn, [])
            scaled = []
            for b in raw_boxes:
                sx = b['x'] * img_w / b['orig_w']
                sy = b['y'] * img_h / b['orig_h']
                sw = b['w'] * img_w / b['orig_w']
                sh = b['h'] * img_h / b['orig_h']
                scaled.append({'x': sx, 'y': sy, 'w': sw, 'h': sh})
            boxes_by_eval[name] = scaled

        # Draw bounding boxes for each evaluator
        for name in EVAL_ORDER:
            draw_boxes(img, boxes_by_eval[name], COLORS[name], thickness=2)

        draw_legend(img, EVAL_ORDER)
        draw_filename(img, fn)

        # Calculate pairwise IoU
        iou_vals = calc_pairwise_iou(boxes_by_eval)

        # Draw IoU at bottom right
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        iou_w, iou_h = 280, 20 + len(iou_vals) * 20
        ox = img_w - iou_w - 10
        oy = 10
        overlay = img.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + iou_w, oy + iou_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.putText(img, 'IoU Pairwise:', (ox + 5, oy + 16), font, 0.45, (255, 255, 255), 1)

        for k, (key, val) in enumerate(sorted(iou_vals.items())):
            label = key.replace('IoU_', '').replace('_', ' vs ')
            cv2.putText(img, f'{label}: {val:.3f}', (ox + 5, oy + 18 + k * 20 + 16),
                        font, 0.4, (255, 255, 255), 1)

        # Save image
        out_path = OUT_DIR / fn
        cv2.imwrite(str(out_path), img)

        # Build CSV row
        row = {'image': fn}
        for key, val in iou_vals.items():
            row[key] = round(val, 4)
        csv_rows.append(row)

        if (i + 1) % 25 == 0:
            print(f'  Procesadas {i + 1}/{len(filenames)}')

    # Write CSV
    if csv_rows:
        csv_keys = list(csv_rows[0].keys())
        with open(CSV_OUT, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_keys)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f'\nCSV guardado: {CSV_OUT} ({len(csv_rows)} filas)')
        print(f'Imágenes guardadas en: {OUT_DIR}/ ({len(csv_rows)} archivos)')

    # Summary
    print(f'\n--- Resumen IoU Promedio ---')
    averages = {}
    for key in csv_keys[1:]:
        vals = [r[key] for r in csv_rows if r[key] is not None]
        avg = np.mean(vals) if vals else 0
        averages[key] = avg
        label = key.replace('IoU_', '').replace('_', ' vs ')
        print(f'  {label}: {avg:.4f}')


if __name__ == '__main__':
    main()