#!/usr/bin/env python3
"""Convert a directory tree of DICOM studies to normalized 16-bit PNG files."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pydicom


def convert_dicom(source: Path, destination: Path, root: Path) -> bool:
    dataset = pydicom.dcmread(source, force=True)
    if not hasattr(dataset, "pixel_array"):
        return False

    image = dataset.pixel_array.astype(np.float32)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = np.amax(image) - image

    minimum = float(image.min())
    maximum = float(image.max())
    if maximum > minimum:
        image = (image - minimum) / (maximum - minimum) * 65535.0

    relative_parent = source.parent.relative_to(root)
    prefix = "_".join(relative_parent.parts[-3:])
    output_name = "_".join(filter(None, (prefix, source.stem))) + ".png"
    return bool(cv2.imwrite(str(destination / output_name), image.astype(np.uint16)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Root directory containing DICOM files.")
    parser.add_argument("output", type=Path, help="Destination directory for PNG files.")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Input directory not found: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errors = 0

    for source in sorted(args.input.rglob("*")):
        if not source.is_file() or source.suffix.lower() != ".dcm":
            continue
        try:
            if convert_dicom(source, args.output, args.input):
                processed += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"[ERROR] {source}: {exc}")
            errors += 1

    print(f"Converted: {processed}")
    print(f"Skipped without pixels: {skipped}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
