#!/usr/bin/env python3
"""Calcula IoU promedio por zona anatomica entre agentes."""

from __future__ import annotations

import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation_common import (
    EVALUATORS,
    IMAGES_DIR,
    TABLES_DIR,
    ZONE_LABELS,
    ZONE_ORDER,
    ensure_output_dirs,
    extract_boxes,
    extract_zone,
    load_evaluator_json,
    optimal_iou_match,
    pair_label,
    pair_order,
)


def main():
    ensure_output_dirs()

    boxes_data = {}
    for evaluator in EVALUATORS:
        data = load_evaluator_json(evaluator)
        boxes_data[evaluator.key] = extract_boxes(data)
        n_boxes = sum(len(boxes) for boxes in boxes_data[evaluator.key].values())
        print(f"{evaluator.display_name}: {len(boxes_data[evaluator.key])} imágenes, {n_boxes} cajas")

    pairs = pair_order()
    pair_labels = [pair_label(first, second) for first, second in pairs]
    matrix = np.full((len(ZONE_ORDER), len(pairs)), np.nan)
    count_matrix = np.zeros((len(ZONE_ORDER), len(pairs)), dtype=int)

    for col, (first, second) in enumerate(pairs):
        common_images = set(boxes_data[first]) & set(boxes_data[second])
        zone_values = {zone: [] for zone in ZONE_ORDER}

        for image_name in common_images:
            zone = extract_zone(image_name)
            if zone not in zone_values:
                continue

            value = optimal_iou_match(
                boxes_data[first].get(image_name, []),
                boxes_data[second].get(image_name, []),
            )
            if value is not None:
                zone_values[zone].append(value)

        for row, zone in enumerate(ZONE_ORDER):
            values = zone_values[zone]
            count_matrix[row, col] = len(values)
            if values:
                matrix[row, col] = float(np.mean(values))

    csv_path = TABLES_DIR / "iou_average_by_zone.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["zona", "codigo", "n_imagenes"] + pair_labels
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, zone in enumerate(ZONE_ORDER):
            row_data = {
                "zona": ZONE_LABELS[zone],
                "codigo": zone,
                "n_imagenes": int(np.nanmax(count_matrix[row])) if count_matrix.shape[1] else 0,
            }
            for col, label in enumerate(pair_labels):
                row_data[label] = "" if np.isnan(matrix[row, col]) else f"{matrix[row, col]:.4f}"
            writer.writerow(row_data)
    print(f"CSV guardado: {csv_path}")

    fig_width = max(14, len(pair_labels) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#F2F2F2")
    image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("IoU promedio")

    ax.set_xticks(np.arange(len(pair_labels)))
    ax.set_yticks(np.arange(len(ZONE_ORDER)))
    ax.set_xticklabels(pair_labels)
    ax.set_yticklabels([ZONE_LABELS[zone] for zone in ZONE_ORDER])
    ax.set_xticks(np.arange(-0.5, len(pair_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ZONE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if not np.isnan(matrix[row, col]):
                ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title("Matriz de IoU promedio por zona anatomica", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Par de agentes")
    ax.set_ylabel("Zona anatomica")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    png_path = IMAGES_DIR / "iou_average_by_zone.png"
    pdf_path = IMAGES_DIR / "iou_average_by_zone.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Heatmap guardado: {png_path}")
    print(f"Heatmap guardado: {pdf_path}")

    return matrix, pairs


if __name__ == "__main__":
    main()
