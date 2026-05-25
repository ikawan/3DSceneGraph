"""Generate YOLO segmentation labels from existing YOLO box labels.

The BIMACS dataset currently has object detection labels:

    class x_center y_center width height

YOLO segmentation training needs polygon labels:

    class x1 y1 x2 y2 ... xn yn

This script uses Ultralytics' SAM-backed converter to generate polygon
pseudo-labels from the existing bounding boxes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("darknet/bimacs_object_detection_data"),
        help="Dataset root containing sibling images/ and labels/ directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save generated YOLO segment labels. Defaults to SOURCE/labels-segment.",
    )
    parser.add_argument(
        "--sam-model",
        default="sam2_b.pt",
        help="SAM checkpoint to use, e.g. sam2_b.pt, sam_b.pt, or mobile_sam.pt.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device for SAM conversion. Use cpu only if necessary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before generating labels.",
    )
    return parser.parse_args()


def assert_box_labels(labels_dir: Path) -> None:
    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No .txt label files found in {labels_dir}")

    checked_rows = 0
    for label_file in label_files:
        for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            checked_rows += 1
            if len(parts) != 5:
                raise ValueError(
                    f"{label_file}:{line_number} does not look like a YOLO box label "
                    f"(expected 5 values, got {len(parts)})."
                )

    if checked_rows == 0:
        raise ValueError(f"Found labels in {labels_dir}, but all label files are empty.")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    images_dir = source / "images"
    labels_dir = source / "labels"
    output_dir = (args.output or source / "labels-segment").resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Detection label directory not found: {labels_dir}")

    assert_box_labels(labels_dir)

    if output_dir.exists():
        existing_labels = list(output_dir.glob("*.txt"))
        if existing_labels and not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already contains labels. Re-run with --overwrite to regenerate them."
            )
        if args.overwrite:
            shutil.rmtree(output_dir)

    from ultralytics.data.converter import yolo_bbox2segment

    yolo_bbox2segment(
        im_dir=images_dir,
        save_dir=output_dir,
        sam_model=args.sam_model,
        device=args.device,
    )

    print(f"Generated segmentation labels in {output_dir}")


if __name__ == "__main__":
    main()
