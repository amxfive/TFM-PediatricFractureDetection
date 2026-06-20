#!/usr/bin/env python3
"""Calcula eficiencia operativa y genera CSV, tabla y boxplot."""

from __future__ import annotations

import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from evaluation_common import EVALUATORS, IMAGES_DIR, TABLES_DIR, ensure_output_dirs, load_evaluator_json


def extract_lead_times(data) -> list[float]:
    lead_times = []
    for item in data:
        for annotation in item.get("annotations", []):
            lead_time = annotation.get("lead_time")
            if lead_time is not None:
                lead_times.append(float(lead_time))
    return lead_times


def summarize_lead_times(evaluator, lead_times: list[float]) -> dict:
    values = np.asarray(lead_times, dtype=float)
    if values.size == 0:
        raise ValueError(f"No lead_time values found for evaluator: {evaluator.display_name}")

    return {
        "evaluador": evaluator.key,
        "nombre": evaluator.display_name,
        "tipo": evaluator.kind,
        "n": int(values.size),
        "media_seg": float(np.mean(values)),
        "std_seg": float(np.std(values)),
        "mediana_seg": float(np.median(values)),
        "q1_seg": float(np.percentile(values, 25)),
        "q3_seg": float(np.percentile(values, 75)),
        "min_seg": float(np.min(values)),
        "max_seg": float(np.max(values)),
        "total_seg": float(np.sum(values)),
        "lead_times": lead_times,
    }


def create_table_plot(results: list[dict]) -> None:
    table_data = [
        [
            row["nombre"],
            f"{row['media_seg']:.3f}",
            f"{row['mediana_seg']:.3f}",
            f"{row['std_seg']:.3f}",
        ]
        for row in results
    ]
    columns = ["Agente", "Media (s)", "Mediana (s)", "Std (s)"]

    fig, ax = plt.subplots(figsize=(11.8, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colColours=["#315C8A"] * len(columns),
        cellColours=[["#F4F6F8"] * len(columns) for _ in table_data],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.15, 1.8)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("white")
        else:
            cell.set_edgecolor("#D5DADF")

    ax.set_title("Grafica de eficiencia", fontsize=14, fontweight="bold", pad=18)

    out_path = IMAGES_DIR / "efficiency_table.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Tabla guardada: {out_path}")


def create_boxplot(results: list[dict]) -> None:
    labels = [row["nombre"] for row in results]
    data = [np.asarray(row["lead_times"], dtype=float) for row in results]
    is_ai = [row["tipo"] == "ai" for row in results]

    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    box = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.62,
        medianprops={"color": "#1F1F1F", "linewidth": 1.5},
        meanprops={"color": "#B3261E", "linewidth": 1.5, "linestyle": "--"},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 2.8,
            "markerfacecolor": "#767676",
            "markeredgecolor": "#555555",
            "alpha": 0.32,
        },
    )

    ai_color = "#2F80ED"
    human_color = "#F2994A"
    edge_color = "#2F2F2F"
    for patch, ai_agent in zip(box["boxes"], is_ai):
        patch.set_facecolor(ai_color if ai_agent else human_color)
        patch.set_alpha(0.62)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(1.1)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(data, start=1):
        positive_values = values[values > 0]
        jitter = rng.normal(0, 0.045, size=len(positive_values))
        ax.scatter(
            positive_values,
            np.full(len(positive_values), idx) + jitter,
            s=9,
            color="#202020",
            alpha=0.22,
            linewidths=0,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Tiempo por imagen (s, escala logarítmica)")
    ax.set_title("Grafica de eficiencia", fontsize=15, fontweight="bold", pad=12)
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for reference in [0.1, 1, 10, 60, 300]:
        ax.axvline(reference, color="#777777", linewidth=0.7, alpha=0.18)

    ax.legend(
        handles=[
            Patch(facecolor=ai_color, edgecolor=edge_color, alpha=0.62, label="Modelos IA"),
            Patch(facecolor=human_color, edgecolor=edge_color, alpha=0.62, label="Agentes humanos"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=10,
    )

    png_path = IMAGES_DIR / "efficiency_boxplot.png"
    pdf_path = IMAGES_DIR / "efficiency_boxplot.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Boxplot guardado: {png_path}")
    print(f"Boxplot guardado: {pdf_path}")


def write_summary_csv(results: list[dict]) -> None:
    csv_path = TABLES_DIR / "efficiency_metrics.csv"
    fieldnames = [
        "evaluador",
        "nombre",
        "tipo",
        "n",
        "media_seg",
        "std_seg",
        "mediana_seg",
        "q1_seg",
        "q3_seg",
        "min_seg",
        "max_seg",
        "total_seg",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"CSV guardado: {csv_path}")


def write_detail_csv(results: list[dict]) -> None:
    csv_path = TABLES_DIR / "efficiency_lead_times.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["evaluador", "nombre", "tipo", "image_index", "lead_time_seg"],
        )
        writer.writeheader()
        for row in results:
            for idx, lead_time in enumerate(row["lead_times"], start=1):
                writer.writerow(
                    {
                        "evaluador": row["evaluador"],
                        "nombre": row["nombre"],
                        "tipo": row["tipo"],
                        "image_index": idx,
                        "lead_time_seg": lead_time,
                    }
                )
    print(f"CSV detallado guardado: {csv_path}")


def main():
    ensure_output_dirs()
    results = []

    for evaluator in EVALUATORS:
        data = load_evaluator_json(evaluator)
        lead_times = extract_lead_times(data)
        results.append(summarize_lead_times(evaluator, lead_times))

    write_summary_csv(results)
    write_detail_csv(results)

    print("\n--- Eficiencia Operativa ---")
    print(f"{'Agente':<24} {'N':>6} {'Media (s)':>12} {'Mediana (s)':>12} {'Std (s)':>12}")
    print("-" * 72)
    for row in results:
        print(
            f"{row['nombre']:<24} "
            f"{row['n']:>6} "
            f"{row['media_seg']:>12.4f} "
            f"{row['mediana_seg']:>12.4f} "
            f"{row['std_seg']:>12.4f}"
        )

    create_table_plot(results)
    create_boxplot(results)
    return results


if __name__ == "__main__":
    main()
