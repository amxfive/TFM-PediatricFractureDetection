#!/usr/bin/env python3
"""Calcula eficiencia operativa y guarda en CSV y tabla."""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_lead_times(data):
    lead_times = []
    for item in data:
        for ann in item.get('annotations', []):
            lt = ann.get('lead_time')
            if lt is not None:
                lead_times.append(lt)
    return lead_times

def create_table_plot(resultados):
    """Crea visualización en forma de tabla."""
    nombres = [r['evaluador'] for r in resultados]
    media = [r['media_seg'] for r in resultados]
    std = [r['std_seg'] for r in resultados]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    table_data = [[n, f"{m:.3f}", f"{s:.3f}"] for n, m, s in zip(nombres, media, std)]
    columns = ['Evaluador', 'Media (s)', 'Std (s)']
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#4472C4'] * 3,
        cellColours=[['#E6E6E6'] * 3 for _ in nombres]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_edgecolor('white')
        else:
            cell.set_edgecolor('#CCCCCC')
    
    ax.set_title('Eficiencia Operativa: Tiempo por Imagen', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('src/evaluation/results/efficiency_table.png', dpi=500, bbox_inches='tight', facecolor='white')
    print("Tabla guardada en efficiency_table.png")

def main():
    base = Path('src/evaluation/annotation_json')
    
    evaluadores = {
        'yoloV8n_optA': 'IA_Evaluation_E3_yoloV8n_optA.json',
        'yoloV8m_optA': 'IA_Evaluation_E6_yoloV8m_optA.json',
        'yoloV11n_optA': 'IA_Evaluation_E7_yoloV11n_optA.json',
        "Usuario_Control": "Control_User_Evaluation_Yasmina_Moreira.json"
    }
    
    resultados = []
    
    for nombre, path in evaluadores.items():
        data = load_json(base / path)
        lead_times = extract_lead_times(data)
        
        resultados.append({
            'evaluador': nombre,
            'n': len(lead_times),
            'media_seg': np.mean(lead_times),
            'std_seg': np.std(lead_times),
            'min_seg': np.min(lead_times),
            'max_seg': np.max(lead_times),
            'total_seg': np.sum(lead_times)
        })
    
    csv_path = 'src/evaluation/results/efficiency_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['evaluador', 'n', 'media_seg', 'std_seg', 'min_seg', 'max_seg', 'total_seg'])
        writer.writeheader()
        writer.writerows(resultados)
    
    print(f"CSV guardado en {csv_path}")
    
    print("\n--- Eficiencia Operativa ---")
    print(f"{'Evaluador':<15} {'N':>6} {'Media (s)':>12} {'Std (s)':>12}")
    print("-" * 50)
    for r in resultados:
        print(f"{r['evaluador']:<15} {r['n']:>6} {r['media_seg']:>12.4f} {r['std_seg']:>12.4f}")
    
    create_table_plot(resultados)
    
    return resultados

if __name__ == '__main__':
    main()