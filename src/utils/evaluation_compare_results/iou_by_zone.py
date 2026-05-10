#!/usr/bin/env python3
"""Matriz IoU por zona corporal."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CSV_IN = Path('src/evaluation/results/iou_per_image.csv')
CSV_OUT = Path('src/evaluation/results/iou_by_zone.csv')
PNG_OUT = Path('src/evaluation/results/iou_by_zone_heatmap.png')
PDF_OUT = Path('src/evaluation/results/iou_by_zone_heatmap.pdf')

ZONE_PREFIXES = {
    'UR': 'UR',
    'NoF_UR': 'UR',
    'WRI': 'WRI',
    'SHF': 'SHF',
}

ZONE_LABELS = {
    'UR': 'Ulna/Radius + NoF',
    'WRI': 'Muñeca',
    'SHF': 'Húmero',
}

def extract_zone(filename):
    for prefix, zone in ZONE_PREFIXES.items():
        if filename.startswith(prefix):
            return zone
    return 'OTHER'

def main():
    with open(CSV_IN) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    iou_keys = [k for k in rows[0].keys() if k.startswith('IoU_')]
    PAIR_LABEL_MAP = {
        'IoU_Humano_IA_E3': 'Humano vs IA_E3',
        'IoU_Humano_IA_E6': 'Humano vs IA_E6',
        'IoU_Humano_IA_E7': 'Humano vs IA_E7',
        'IoU_IA_E3_IA_E6': 'IA_E3 vs IA_E6',
        'IoU_IA_E3_IA_E7': 'IA_E3 vs IA_E7',
        'IoU_IA_E6_IA_E7': 'IA_E6 vs IA_E7',
    }
    pair_labels = [PAIR_LABEL_MAP[k] for k in iou_keys]

    zones_data = {}
    for row in rows:
        fn = row['image']
        zone = extract_zone(fn)
        if zone not in zones_data:
            zones_data[zone] = {k: [] for k in iou_keys}
        for k in iou_keys:
            v = float(row[k])
            zones_data[zone][k].append(v)

    zone_order = ['UR', 'WRI', 'SHF']
    print('--- IoU Promedio por Zona Corporal ---\n')

    # Print table
    header = f"{'Zona':<20}" + "".join(f"{pair:>20}" for pair in pair_labels)
    print(header)
    print('-' * len(header))

    matrix = np.zeros((len(zone_order), len(iou_keys)))
    csv_rows = []

    for i, zone in enumerate(zone_order):
        row_data = {'zona': ZONE_LABELS.get(zone, zone), 'codigo': zone, 'n_imagenes': len(zones_data[zone][iou_keys[0]])}
        vals = []
        for j, k in enumerate(iou_keys):
            avg = np.mean(zones_data[zone][k])
            row_data[k] = round(avg, 4)
            matrix[i, j] = avg
            vals.append(avg)

        line = f"{ZONE_LABELS.get(zone, zone):<20}" + "".join(f"{v:>20.4f}" for v in vals)
        print(line)
        csv_rows.append(row_data)

    print(f"\n{'MEDIA GLOBAL':<20}" + "".join(f"{np.mean(matrix[:, j]):>20.4f}" for j in range(len(iou_keys))))

    with open(CSV_OUT, 'w', newline='') as f:
        fieldnames = ['zona', 'codigo', 'n_imagenes'] + iou_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f'\nCSV guardado: {CSV_OUT}')

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        xticklabels=pair_labels,
        yticklabels=[ZONE_LABELS.get(z, z) for z in zone_order],
        vmin=0,
        vmax=1,
        center=0.5,
        cbar_kws={'label': 'IoU promedio'},
        ax=ax,
        annot_kws={'size': 11},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_title('IoU Promedio por Zona Corporal', fontsize=14, fontweight='bold')
    ax.set_xlabel('Par de evaluadores', fontsize=12)
    ax.set_ylabel('Zona corporal', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    plt.savefig(PNG_OUT, dpi=150, bbox_inches='tight')
    plt.savefig(PDF_OUT, bbox_inches='tight')
    print(f'Heatmap guardado: {PNG_OUT}')

if __name__ == '__main__':
    main()