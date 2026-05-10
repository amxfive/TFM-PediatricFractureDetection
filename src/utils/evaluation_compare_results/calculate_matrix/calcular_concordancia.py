#!/usr/bin/env python3
"""Calcula matriz de concordancia basada en IoU entre evaluadores."""

import json
import numpy as np
from pathlib import Path

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_boxes(data):
    """Extrae bounding boxes de un JSON de evaluación."""
    result = {}
    for item in data:
        file_upload = item.get('data').get("image") or item.get('file_upload', '')
        annotations = item.get('annotations', [])
        
        boxes = []
        for ann in annotations:
            results = ann.get('result', [])
            for r in results:
                value = r.get('value', {})
                orig_w = r.get('original_width', 1)
                orig_h = r.get('original_height', 1)
                
                x = value['x'] / 100 * orig_w
                y = value['y'] / 100 * orig_h
                w = value['width'] / 100 * orig_w
                h = value['height'] / 100 * orig_h
                
                label = value.get('rectanglelabels', ['0'])[0]
                boxes.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'label': label,
                    'orig_w': orig_w, 'orig_h': orig_h
                })
        
        result[file_upload] = boxes
    return result

def iou(box1, box2):
    """Calcula IoU entre dos bounding boxes."""
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
    y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0

def best_iou_match(boxes1, boxes2):
    """Encuentra el mejor matching IoU entre dos listas de boxes."""
    if not boxes1 and not boxes2:
        return None  # Acuerdo (ambos vacíos)
    
    if not boxes1 or not boxes2:
        return 0.0  # Discrepancia (uno vacío)
    
    max_iou = 0.0
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
            max_iou += best_val
            used.add(best_j)
    
    n_matches = max(len(boxes1), len(boxes2))
    return max_iou / n_matches if n_matches > 0 else 0.0

def calculate_concordance(eval1_boxes, eval2_boxes):
    """Calcula concordancia promedio IoU entre dos evaluadores."""
    ious = []
    
    common_images = set(eval1_boxes.keys()) & set(eval2_boxes.keys())
    
    for img in common_images:
        boxes1 = eval1_boxes[img]
        boxes2 = eval2_boxes[img]
        
        iou_val = best_iou_match(boxes1, boxes2)
        if iou_val is not None:
            ious.append(iou_val)
    
    return np.mean(ious) if ious else 0.0

def main():
    base = Path('src/evaluation/annotation_json')
    
    evaluadores = {
        'yoloV8n_optA': 'IA_Evaluation_E3_yoloV8n_optA.json',
        'yoloV8m_optA': 'IA_Evaluation_E6_yoloV8m_optA.json',
        'yoloV11n_optA': 'IA_Evaluation_E7_yoloV11n_optA.json',
        "Usuario_Control": "Control_User_Evaluation_Yasmina_Moreira.json"
    }
    
    boxes_data = {}
    for nombre, path in evaluadores.items():
        data = load_json(base / path)
        boxes_data[nombre] = extract_boxes(data)
        print(f"{nombre}: {len(boxes_data[nombre])} imágenes")
    
    nombres = list(evaluadores.keys())
    n = len(nombres)
    matriz = np.zeros((n, n))
    
    print("\n--- Matriz de Concordancia IoU ---")
    
    for i, nom1 in enumerate(nombres):
        for j, nom2 in enumerate(nombres):
            if i == j:
                matriz[i, j] = 1.0
            else:
                matriz[i, j] = calculate_concordance(boxes_data[nom1], boxes_data[nom2])
    
    print("\n" + " " * 15 + "".join(f"{n:>12}" for n in nombres))
    for i, nom in enumerate(nombres):
        row = "".join(f"{matriz[i, j]:>12.4f}" for j in range(n))
        print(f"{nom:>15}{row}")
    
    np.save('src/evaluation/matrix/concordance_matrix.npy', matriz)
    print("\nMatriz guardada en src/evaluation/matrix/concordance_matrix.npy")
    
    return matriz, nombres

if __name__ == '__main__':
    main()