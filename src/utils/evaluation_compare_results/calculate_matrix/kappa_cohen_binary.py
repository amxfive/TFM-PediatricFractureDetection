#!/usr/bin/env python3
"""Calcula matriz de Kappa de Cohen binario entre evaluadores."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def cohen_kappa_binary(boxes1, boxes2):
    """
    Calcula Kappa de Cohen binario para un par de evaluadores.
    - Acuerdo positivo: ambos detectan >= 1 caja
    - Acuerdo negativo: ambos NO detectan ninguna caja
    - Desacuerdo: uno detecta y el otro no
    """
    n = len(boxes1)
    if n != len(boxes2):
        raise ValueError("Los diccionarios deben tener las mismas imágenes")
    
    # Contadores
    agreement_pos = 0  # Ambos detectan
    agreement_neg = 0  # Ambos no detectan
    disagreement_1o2 = 0  # evaluador1 detecta, evaluador2 no
    disagreement_2o1 = 0  # evaluador2 detecta, evaluador1 no
    
    for img in boxes1:
        has_1 = len(boxes1[img]) > 0
        has_2 = len(boxes2[img]) > 0
        
        if has_1 and has_2:
            agreement_pos += 1
        elif not has_1 and not has_2:
            agreement_neg += 1
        elif has_1 and not has_2:
            disagreement_1o2 += 1
        else:  # not has_1 and has_2
            disagreement_2o1 += 1
    
    n_total = agreement_pos + agreement_neg + disagreement_1o2 + disagreement_2o1
    
    # Probabilidad observada de acuerdo
    po = (agreement_pos + agreement_neg) / n_total
    
    # Probabilidad esperada de acuerdo
    p1_pos = (agreement_pos + disagreement_1o2) / n_total
    p2_pos = (agreement_pos + disagreement_2o1) / n_total
    p1_neg = (agreement_neg + disagreement_2o1) / n_total
    p2_neg = (agreement_neg + disagreement_1o2) / n_total
    
    pe = (p1_pos * p2_pos) + (p1_neg * p2_neg)
    
    # Kappa
    if pe == 1:
        return 1.0
    
    kappa = (po - pe) / (1 - pe)
    
    return kappa

def calculate_kappa_matrix(boxes_data, nombres):
    """Calcula matriz de Kappa entre todos los pares de evaluadores."""
    n = len(nombres)
    matriz = np.zeros((n, n))
    
    for i, nom1 in enumerate(nombres):
        for j, nom2 in enumerate(nombres):
            if i == j:
                matriz[i, j] = 1.0
            else:
                matriz[i, j] = cohen_kappa_binary(boxes_data[nom1], boxes_data[nom2])
    
    return matriz

def main():
    base = Path('src/evaluation/annotation_json')
    
    evaluadores = {
        'yoloV8n_optA': 'IA_Evaluation_E3_yoloV8n_optA.json',
        'yoloV8m_optA': 'IA_Evaluation_E6_yoloV8m_optA.json',
        'yoloV11n_optA': 'IA_Evaluation_E7_yoloV11n_optA.json',
        "Usuario_Control": "Control_User_Evaluation_Yasmina_Moreira.json",
        "Usuario_R1": "R1_User_Evaluation_Marina.json",
        "Usuario_Catedratico": "Catedratico_User_Evaluation_Jose_Carlos.json"
    }
    
    boxes_data = {}
    for nombre, path in evaluadores.items():
        data = load_json(base / path)
        boxes_data[nombre] = extract_boxes(data)
        print(f"{nombre}: {len(boxes_data[nombre])} imágenes")
    
    nombres = list(evaluadores.keys())
    
    matriz = calculate_kappa_matrix(boxes_data, nombres)
    
    print("\n--- Matriz de Kappa de Cohen (Binario) ---")
    print("\n" + " " * 15 + "".join(f"{n:>12}" for n in nombres))
    for i, nom in enumerate(nombres):
        row = "".join(f"{matriz[i, j]:>12.4f}" for j in range(len(nombres)))
        print(f"{nom:>15}{row}")
    
    np.save('src/evaluation/matrix/kappa_matrix_v2.npy', matriz)
    print("\nMatriz guardada en src/evaluation/matrix/kappa_matrix_v2.npy")

    # Heatmap
    n = len(nombres)
    half = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            half[i, j] = matriz[i, j]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        half, annot=True, fmt='.3f', cmap=sns.diverging_palette(220, 20, as_cmap=True),
        xticklabels=nombres, yticklabels=nombres,
        vmin=-0.3, vmax=1, center=0.35, square=True,
        cbar_kws={'label': 'Kappa de Cohen'}, ax=ax,
        annot_kws={'size': 9}, linewidths=0.5, linecolor='white'
    )
    ax.set_title('Matriz de Kappa de Cohen (Binario)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('src/evaluation/results/kappa_halfmatrix.png', dpi=500, bbox_inches='tight')
    print("Heatmap guardado en src/evaluation/results/kappa_halfmatrix.png")
    
    return matriz, nombres

if __name__ == '__main__':
    main()