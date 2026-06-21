#!/usr/bin/env python3
"""Upload a local dataset directory as a Weights & Biases artifact."""

import argparse
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--project", default="TFM_Fracturas")
    parser.add_argument("--artifact", default="ExpDatasetClassification")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        parser.error(f"Dataset directory not found: {args.dataset}")

    run = wandb.init(project=args.project, job_type="dataset-upload")
    artifact = wandb.Artifact(args.artifact, type="dataset")
    artifact.add_dir(str(args.dataset))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
