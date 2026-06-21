#!/usr/bin/env python3
"""Train and evaluate the anatomical YOLO classifier/router."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/processed_2/ExpDataset_classification",
        help="Directory containing the YOLO classification dataset.",
    )
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--project", default="runs/classification")
    parser.add_argument("--name", default="discriminator_yolov8n")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    validation = model.val(data=args.data, split="test")
    print(f"Top-1 Accuracy: {validation.top1:.4f}")
    print(f"Top-5 Accuracy: {validation.top5:.4f}")

    experiment_dir = Path(args.project) / args.name
    metrics_path = experiment_dir / "metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"Top-1 Accuracy: {validation.top1:.4f}",
                f"Top-5 Accuracy: {validation.top5:.4f}",
                f"Model: {args.model}",
                f"Epochs: {args.epochs}",
                f"Image Size: {args.imgsz}",
                f"Batch Size: {args.batch}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Metrics saved to: {metrics_path}")
    print(f"Best model saved to: {experiment_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
