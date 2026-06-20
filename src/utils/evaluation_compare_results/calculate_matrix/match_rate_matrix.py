#!/usr/bin/env python3
"""Calcula matrices de tasa de matching simétrica para distintos umbrales IoU."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation_common import (  # noqa: E402
    EVALUATORS,
    IMAGES_DIR,
    MATRIX_DIR,
    TABLES_DIR,
    display_name,
    ensure_output_dirs,
    extract_boxes,
    iou,
    load_evaluator_json,
)


THRESHOLDS = [
    (0.2, "t20", "Matriz IoU > 0.2"),
    (1e-6, "epsilon", "Matriz IoU > ε"),
]


def best_iou_val(box, other_boxes) -> float:
    return max((iou(box, other_box) for other_box in other_boxes), default=0.0)


def calculate_match_rate(eval1_boxes, eval2_boxes, threshold: float) -> float:
    total_matches_1in2 = 0
    total_boxes_1 = 0
    total_matches_2in1 = 0
    total_boxes_2 = 0

    common_images = set(eval1_boxes) & set(eval2_boxes)
    for image_name in common_images:
        boxes1 = eval1_boxes.get(image_name, [])
        boxes2 = eval2_boxes.get(image_name, [])

        for box1 in boxes1:
            if best_iou_val(box1, boxes2) > threshold:
                total_matches_1in2 += 1
        total_boxes_1 += len(boxes1)

        for box2 in boxes2:
            if best_iou_val(box2, boxes1) > threshold:
                total_matches_2in1 += 1
        total_boxes_2 += len(boxes2)

    rate_1in2 = total_matches_1in2 / total_boxes_1 if total_boxes_1 else np.nan
    rate_2in1 = total_matches_2in1 / total_boxes_2 if total_boxes_2 else np.nan

    if np.isnan(rate_1in2) and np.isnan(rate_2in1):
        return np.nan
    return float(np.nanmean([rate_1in2, rate_2in1]))


def calculate_matrix(boxes_data, keys: list[str], threshold: float) -> np.ndarray:
    matrix = np.full((len(keys), len(keys)), np.nan)
    for i, key1 in enumerate(keys):
        matrix[i, i] = 1.0
        for j in range(i + 1, len(keys)):
            key2 = keys[j]
            value = calculate_match_rate(boxes_data[key1], boxes_data[key2], threshold)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def write_csv(matrix: np.ndarray, keys: list[str], suffix: str) -> None:
    csv_path = TABLES_DIR / f"match_rate_{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([""] + [display_name(key) for key in keys])
        for index, key in enumerate(keys):
            writer.writerow(
                [display_name(key)]
                + [
                    "" if np.isnan(matrix[index, col]) else f"{matrix[index, col]:.4f}"
                    for col in range(len(keys))
                ]
            )
    print(f"CSV guardado: {csv_path}")


def plot_heatmap(matrix: np.ndarray, keys: list[str], suffix: str, title: str) -> None:
    labels = [display_name(key) for key in keys]
    plot_matrix = matrix.copy()
    triangle_mask = np.tril(np.ones_like(plot_matrix, dtype=bool), k=-1)
    plot_matrix[triangle_mask] = np.nan
    masked = np.ma.masked_invalid(plot_matrix)

    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#F2F2F2")

    fig, ax = plt.subplots(figsize=(10.5, 8))
    image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Tasa de coincidencia")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if not np.isnan(plot_matrix[row, col]):
                ax.text(col, row, f"{plot_matrix[row, col]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Agente")
    ax.set_ylabel("Agente")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    png_path = IMAGES_DIR / f"match_rate_matrix_{suffix}.png"
    pdf_path = IMAGES_DIR / f"match_rate_matrix_{suffix}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Heatmap guardado: {png_path}")
    print(f"Heatmap guardado: {pdf_path}")


def main():
    ensure_output_dirs()

    boxes_data = {}
    for evaluator in EVALUATORS:
        data = load_evaluator_json(evaluator)
        boxes_data[evaluator.key] = extract_boxes(data)
        n_boxes = sum(len(boxes) for boxes in boxes_data[evaluator.key].values())
        print(f"{evaluator.display_name}: {len(boxes_data[evaluator.key])} imágenes, {n_boxes} cajas")

    keys = [evaluator.key for evaluator in EVALUATORS]

    for threshold, suffix, title in THRESHOLDS:
        print(f"\n--- {title} ---")
        matrix = calculate_matrix(boxes_data, keys, threshold)
        matrix_path = MATRIX_DIR / f"match_rate_matrix_{suffix}.npy"
        np.save(matrix_path, matrix)
        print(f"Matriz guardada: {matrix_path}")
        write_csv(matrix, keys, suffix)
        plot_heatmap(matrix, keys, suffix, title)


if __name__ == "__main__":
    main()
