#!/usr/bin/env python3
"""Matriz IoU por zona corporal. Calcula directo desde los JSONs."""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

JSONS = {
    'yoloV8n_optA': 'IA_Evaluation_E3_yoloV8n_optA.json',
    'yoloV8m_optA': 'IA_Evaluation_E6_yoloV8m_optA.json',
    'yoloV11n_optA': 'IA_Evaluation_E7_yoloV11n_optA.json',
    'Usuario_Control': 'Control_User_Evaluation_Yasmina_Moreira.json',
    'Usuario_R1': 'R1_User_Evaluation_Marina.json',
    'Usuario_Catedratico': 'Catedratico_User_Evaluation_Jose_Carlos.json',
}

ZONE_PREFIXES = {'UR': 'UR', 'NoF_UR': 'UR', 'WRI': 'WRI', 'SHF': 'SHF'}
ZONE_LABELS = {'UR': 'Ulna/Radius', 'WRI': 'Muñeca', 'SHF': 'Húmero'}
ZONE_ORDER = ['UR', 'WRI', 'SHF']

SHORT = {
    'yoloV8n_optA': 'v8n',
    'yoloV8m_optA': 'v8m',
    'yoloV11n_optA': 'v11n',
    'Usuario_Control': 'Control',
    'Usuario_R1': 'R1',
    'Usuario_Catedratico': 'Catedrático',
}

PAIR_ORDER = [
    ('Usuario_Catedratico', 'Usuario_Control'),
    ('Usuario_Catedratico', 'Usuario_R1'),
    ('Usuario_Catedratico', 'yoloV8n_optA'),
    ('Usuario_Catedratico', 'yoloV8m_optA'),
    ('Usuario_Catedratico', 'yoloV11n_optA'),
    ('Usuario_R1', 'Usuario_Control'),
    ('Usuario_R1', 'yoloV8n_optA'),
    ('Usuario_R1', 'yoloV8m_optA'),
    ('Usuario_R1', 'yoloV11n_optA'),
    ('Usuario_Control', 'yoloV8n_optA'),
    ('Usuario_Control', 'yoloV8m_optA'),
    ('Usuario_Control', 'yoloV11n_optA'),
    ('yoloV8n_optA', 'yoloV8m_optA'),
    ('yoloV8n_optA', 'yoloV11n_optA'),
    ('yoloV8m_optA', 'yoloV11n_optA'),
]

BASE = Path('src/evaluation/annotation_json')

def load_json(path):
    with open(path) as f:
        return json.load(f)

def extract_boxes(data):
    result = {}
    for item in data:
        fn = item.get('file_upload') or item.get('data', {}).get('image', '') or str(item.get('id', ''))
        boxes = []
        for ann in item.get('annotations', []):
            for r in ann.get('result', []):
                v = r.get('value', {})
                ow = r.get('original_width', 1)
                oh = r.get('original_height', 1)
                boxes.append({
                    'x': v['x'] / 100 * ow, 'y': v['y'] / 100 * oh,
                    'w': v['width'] / 100 * ow, 'h': v['height'] / 100 * oh,
                })
        result[fn] = boxes
    return result

def iou(b1, b2):
    x1, y1 = max(b1['x'], b2['x']), max(b1['y'], b2['y'])
    x2, y2 = min(b1['x'] + b1['w'], b2['x'] + b2['w']), min(b1['y'] + b1['h'], b2['y'] + b2['h'])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = b1['w'] * b1['h'] + b2['w'] * b2['h'] - inter
    return inter / union if union > 0 else 0.0

def best_iou_match(boxes1, boxes2):
    if not boxes1 and not boxes2:
        return None
    if not boxes1 or not boxes2:
        return 0.0
    total, used = 0.0, set()
    for b1 in boxes1:
        best_val, best_j = 0.0, -1
        for j, b2 in enumerate(boxes2):
            if j not in used:
                val = iou(b1, b2)
                if val > best_val:
                    best_val, best_j = val, j
        if best_j >= 0:
            total += best_val
            used.add(best_j)
    return total / max(len(boxes1), len(boxes2))

def extract_zone(filename):
    for prefix, zone in ZONE_PREFIXES.items():
        if filename.startswith(prefix):
            return zone
    return 'OTHER'

def main():
    # Load all data
    boxes_data = {}
    for nombre, path in JSONS.items():
        data = load_json(BASE / path)
        boxes_data[nombre] = extract_boxes(data)
        n_boxes = sum(len(v) for v in boxes_data[nombre].values())
        print(f'{nombre}: {len(boxes_data[nombre])} imágenes, {n_boxes} cajas')

    nombres = list(JSONS.keys())
    PAIR_ORDER = [
        ('Usuario_Catedratico', 'Usuario_Control'),
        ('Usuario_Catedratico', 'Usuario_R1'),
        ('Usuario_Catedratico', 'yoloV8n_optA'),
        ('Usuario_Catedratico', 'yoloV8m_optA'),
        ('Usuario_Catedratico', 'yoloV11n_optA'),
        ('Usuario_R1', 'Usuario_Control'),
        ('Usuario_R1', 'yoloV8n_optA'),
        ('Usuario_R1', 'yoloV8m_optA'),
        ('Usuario_R1', 'yoloV11n_optA'),
        ('Usuario_Control', 'yoloV8n_optA'),
        ('Usuario_Control', 'yoloV8m_optA'),
        ('Usuario_Control', 'yoloV11n_optA'),
        ('yoloV8n_optA', 'yoloV8m_optA'),
        ('yoloV8n_optA', 'yoloV11n_optA'),
        ('yoloV8m_optA', 'yoloV11n_optA'),
    ]

    iou_keys = [f'IoU_{n1}_{n2}' for n1, n2 in PAIR_ORDER]
    pair_labels = [f'{SHORT[n1]} vs {SHORT[n2]}' for n1, n2 in PAIR_ORDER]

    # Group by zone and calculate IoU per image
    zone_img_data = {z: {k: [] for k in iou_keys} for z in ZONE_ORDER}
    common = set(boxes_data[nombres[0]].keys())

    for fn in common:
        zone = extract_zone(fn)
        if zone not in zone_img_data:
            continue
        for n1, n2 in PAIR_ORDER:
            b1 = boxes_data[n1].get(fn, [])
            b2 = boxes_data[n2].get(fn, [])
            key = f'IoU_{n1}_{n2}'
            val = best_iou_match(b1, b2)
            zone_img_data[zone][key].append(val if val is not None else 1.0)

    # Print matrix
    print('\n--- IoU Promedio por Zona Corporal ---\n')
    header = f"{'Zona':<22}" + "".join(f"{p:>22}" for p in pair_labels)
    print(header)
    print('-' * len(header))

    matrix = np.zeros((len(ZONE_ORDER), len(iou_keys)))
    csv_rows = []

    for i, zone in enumerate(ZONE_ORDER):
        vals = []
        row = {'zona': ZONE_LABELS[zone], 'codigo': zone, 'n_imagenes': len(zone_img_data[zone][iou_keys[0]])}
        for j, k in enumerate(iou_keys):
            avg = np.mean([v for v in zone_img_data[zone][k] if v is not None])
            row[k] = round(avg, 4)
            matrix[i, j] = avg
            vals.append(avg)
        csv_rows.append(row)
        line = f"{ZONE_LABELS[zone]:<22}" + "".join(f"{v:>22.4f}" for v in vals)
        print(line)

    print(f"\n{'MEDIA GLOBAL':<22}" + "".join(f"{np.mean(matrix[:, j]):>22.4f}" for j in range(len(iou_keys))))

    # Save CSV
    csv_out = Path('src/evaluation/results/iou_by_zone.csv')
    with open(csv_out, 'w', newline='') as f:
        fieldnames = ['zona', 'codigo', 'n_imagenes'] + iou_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'\nCSV guardado: {csv_out}')

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        matrix, annot=True, fmt='.3f',
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        xticklabels=pair_labels,
        yticklabels=[ZONE_LABELS[z] for z in ZONE_ORDER],
        vmin=0, vmax=1, center=0.5,
        cbar_kws={'label': 'IoU promedio'},
        ax=ax, annot_kws={'size': 10},
        linewidths=0.5, linecolor='white'
    )
    ax.set_title('IoU Promedio por Zona Corporal', fontsize=14, fontweight='bold')
    ax.set_xlabel('Par de evaluadores', fontsize=12)
    ax.set_ylabel('Zona corporal', fontsize=12)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig('src/evaluation/results/iou_by_zone_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig('src/evaluation/results/iou_by_zone_heatmap.pdf', bbox_inches='tight')
    print(f'Heatmap guardado: src/evaluation/results/iou_by_zone_heatmap.png')

if __name__ == '__main__':
    main()