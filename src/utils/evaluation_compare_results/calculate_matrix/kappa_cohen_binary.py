#!/usr/bin/env python3
"""Calcula la matriz de Kappa de Cohen binario entre evaluadores."""

from __future__ import annotations

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
    ensure_output_dirs,
    extract_boxes,
    load_evaluator_json,
)


def cohen_kappa_binary(boxes1, boxes2) -> float:
    """Calcula Kappa segun presencia o ausencia de al menos una caja."""
    images = sorted(set(boxes1) | set(boxes2))
    if not images:
        return np.nan

    agreement_positive = 0
    agreement_negative = 0
    disagreement_1_to_2 = 0
    disagreement_2_to_1 = 0

    for image_name in images:
        has_1 = bool(boxes1.get(image_name, []))
        has_2 = bool(boxes2.get(image_name, []))

        if has_1 and has_2:
            agreement_positive += 1
        elif not has_1 and not has_2:
            agreement_negative += 1
        elif has_1:
            disagreement_1_to_2 += 1
        else:
            disagreement_2_to_1 += 1

    total = len(images)
    observed = (agreement_positive + agreement_negative) / total

    evaluator1_positive = (agreement_positive + disagreement_1_to_2) / total
    evaluator2_positive = (agreement_positive + disagreement_2_to_1) / total
    evaluator1_negative = (agreement_negative + disagreement_2_to_1) / total
    evaluator2_negative = (agreement_negative + disagreement_1_to_2) / total
    expected = (
        evaluator1_positive * evaluator2_positive
        + evaluator1_negative * evaluator2_negative
    )

    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def calculate_kappa_matrix(boxes_data, names) -> np.ndarray:
    matrix = np.zeros((len(names), len(names)))
    for row, name1 in enumerate(names):
        for column, name2 in enumerate(names):
            if row == column:
                matrix[row, column] = 1.0
            else:
                matrix[row, column] = cohen_kappa_binary(
                    boxes_data[name1],
                    boxes_data[name2],
                )
    return matrix


def save_heatmap(matrix: np.ndarray, names: list[str]) -> Path:
    size = len(names)
    upper_half = np.full((size, size), np.nan)
    for row in range(size):
        for column in range(row, size):
            upper_half[row, column] = matrix[row, column]

    fig, ax = plt.subplots(figsize=(11, 8))
    masked = np.ma.masked_invalid(upper_half)
    image = ax.imshow(masked, cmap="RdYlBu_r", vmin=-0.3, vmax=1.0)
    image.cmap.set_bad("#f2f2f2")

    ax.set_xticks(range(size), labels=names, rotation=35, ha="right")
    ax.set_yticks(range(size), labels=names)
    ax.set_title(
        "Matriz de Kappa de Cohen (Binario)",
        fontsize=14,
        fontweight="bold",
    )

    for row in range(size):
        for column in range(size):
            if not np.isnan(upper_half[row, column]):
                ax.text(
                    column,
                    row,
                    f"{upper_half[row, column]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Kappa de Cohen")
    fig.tight_layout()

    output_path = IMAGES_DIR / "kappa_halfmatrix.png"
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    ensure_output_dirs()
    boxes_data = {}

    for evaluator in EVALUATORS:
        data = load_evaluator_json(evaluator)
        boxes_data[evaluator.display_name] = extract_boxes(data)
        print(
            f"{evaluator.display_name}: "
            f"{len(boxes_data[evaluator.display_name])} imagenes"
        )

    names = [evaluator.display_name for evaluator in EVALUATORS]
    matrix = calculate_kappa_matrix(boxes_data, names)

    print("\n--- Matriz de Kappa de Cohen (Binario) ---")
    print(matrix)

    matrix_path = MATRIX_DIR / "kappa_matrix_v2.npy"
    np.save(matrix_path, matrix)
    print(f"\nMatriz guardada en {matrix_path}")

    heatmap_path = save_heatmap(matrix, names)
    print(f"Heatmap guardado en {heatmap_path}")
    return matrix, names


if __name__ == "__main__":
    main()
