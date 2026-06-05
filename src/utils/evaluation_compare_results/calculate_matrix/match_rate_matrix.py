#!/usr/bin/env python3
"""Matriz de tasa de matching simétrica (IoU > threshold) entre evaluadores."""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────
IOU_THRESHOLD = 0.2
# ────────────────────────────────────────────────────

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_boxes(data):
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
                boxes.append({'x': x, 'y': y, 'w': w, 'h': h})
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
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def best_iou_val(box, other_boxes):
    best = 0.0
    for ob in other_boxes:
        val = iou(box, ob)
        if val > best:
            best = val
    return best

def calculate_match_rate_matrix(boxes_data, evaluadores, threshold):
    nombres = list(evaluadores.keys())
    n = len(nombres)
    matriz = np.zeros((n, n))

    for i, nom1 in enumerate(nombres):
        for j, nom2 in enumerate(nombres):
            if i == j:
                matriz[i, j] = 1.0
                continue

            total_matches_1in2 = 0
            total_boxes_1 = 0
            total_matches_2in1 = 0
            total_boxes_2 = 0

            for img in boxes_data[nom1]:
                boxes1 = boxes_data[nom1].get(img, [])
                boxes2 = boxes_data[nom2].get(img, [])

                if not boxes1 and not boxes2:
                    continue

                # A→B: cajas de 1 que tienen match > threshold en 2
                if boxes1:
                    for b1 in boxes1:
                        if best_iou_val(b1, boxes2) > threshold:
                            total_matches_1in2 += 1
                    total_boxes_1 += len(boxes1)

                # B→A: cajas de 2 que tienen match > threshold en 1
                if boxes2:
                    for b2 in boxes2:
                        if best_iou_val(b2, boxes1) > threshold:
                            total_matches_2in1 += 1
                    total_boxes_2 += len(boxes2)

            rate_1in2 = total_matches_1in2 / total_boxes_1 if total_boxes_1 > 0 else 0
            rate_2in1 = total_matches_2in1 / total_boxes_2 if total_boxes_2 > 0 else 0
            matriz[i, j] = (rate_1in2 + rate_2in1) / 2

    return matriz, nombres

def main():
    JSONS = {
        'yoloV8n_optA': 'IA_Evaluation_E3_yoloV8n_optA.json',
        'yoloV8m_optA': 'IA_Evaluation_E6_yoloV8m_optA.json',
        'yoloV11n_optA': 'IA_Evaluation_E7_yoloV11n_optA.json',
        'Usuario_Control': 'Control_User_Evaluation_Yasmina_Moreira.json',
        'Usuario_R1': 'R1_User_Evaluation_Marina.json',
        'Usuario_Catedratico': 'Catedratico_User_Evaluation_Jose_Carlos.json'
    }
    base = Path('src/evaluation/annotation_json')

    boxes_data = {}
    for nombre, path in JSONS.items():
        data = load_json(base / path)
        boxes_data[nombre] = extract_boxes(data)
        n_boxes = sum(len(v) for v in boxes_data[nombre].values())
        print(f'{nombre}: {len(boxes_data[nombre])} imágenes, {n_boxes} cajas')

    threshold = IOU_THRESHOLD
    print(f'\nThreshold IoU = {threshold}')

    matriz, nombres = calculate_match_rate_matrix(boxes_data, JSONS, threshold)

    print('\n' + " " * 20 + "".join(f"{n:>18}" for n in nombres))
    for i, nom in enumerate(nombres):
        row = "".join(f"{matriz[i, j]:>18.4f}" for j in range(len(nombres)))
        print(f"{nom:>20}{row}")

    # Save matrix
    out_dir = Path('src/evaluation/matrix')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'match_rate_matrix_t{int(threshold*100)}.npy', matriz)

    # CSV
    csv_path = Path('src/evaluation/results') / f'match_rate_t{int(threshold*100)}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + nombres)
        for i, nom in enumerate(nombres):
            writer.writerow([nom] + [f"{matriz[i, j]:.4f}" for j in range(len(nombres))])
    print(f'\nCSV guardado: {csv_path}')

    # Half-matrix heatmap
    n = len(nombres)
    half = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            half[i, j] = matriz[i, j]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        half,
        annot=True,
        fmt='.3f',
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        xticklabels=nombres,
        yticklabels=nombres,
        vmin=0,
        vmax=1,
        center=0.5,
        square=True,
        cbar_kws={'label': f'Match Rate (IoU > {threshold})'},
        ax=ax,
        annot_kws={'size': 10},
        linewidths=0.5,
        linecolor='white'
    )
    ax.set_title(f'Tasa de Matching (IoU > {threshold})', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    png_path = Path('src/evaluation/results') / f'match_rate_heatmap_t{int(threshold*100)}.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'Heatmap guardado: {png_path}')

if __name__ == '__main__':
    main()