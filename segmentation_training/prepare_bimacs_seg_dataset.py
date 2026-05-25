"""Package BIMACS images and YOLO segment labels for Ultralytics training."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = [
    "unused1",
    "bowl",
    "knife",
    "screwdriver",
    "cuttingboard",
    "whisk",
    "hammer",
    "bottle",
    "unused2",
    "cup",
    "banana",
    "cereals",
    "sponge",
    "wood",
    "saw",
    "harddrive",
]

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("darknet/bimacs_object_detection_data"),
        help="Original BIMACS dataset root.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="YOLO segment labels directory. Defaults to SOURCE/labels-segment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/bimacs-seg"),
        help="Output Ultralytics segmentation dataset root.",
    )
    parser.add_argument("--train", type=float, default=0.80, help="Training split ratio.")
    parser.add_argument("--val", type=float, default=0.10, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.10, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group split.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output dataset before recreating it.",
    )
    return parser.parse_args()


def image_group(image_path: Path) -> str:
    """Group video frames by recording to reduce train/val/test leakage."""
    stem = image_path.stem
    if "_frame_" in stem:
        return stem.split("_frame_", maxsplit=1)[0]
    return stem


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def validate_segment_label(label_path: Path, class_count: int) -> tuple[int, set[int]]:
    """Return object count and class ids after validating YOLO segment format."""
    object_count = 0
    class_ids: set[int] = set()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            raise ValueError(
                f"{label_path}:{line_number} is not YOLO segmentation format. "
                "Expected: class x1 y1 x2 y2 x3 y3 ..."
            )
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} contains a non-numeric value.") from exc
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"{label_path}:{line_number} has class id {class_id}, expected 0-{class_count - 1}.")
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"{label_path}:{line_number} has coordinates outside the normalized 0-1 range.")
        object_count += 1
        class_ids.add(class_id)
    return object_count, class_ids


def split_groups(grouped_images: dict[str, list[Path]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    groups = list(grouped_images)
    rng = random.Random(seed)
    rng.shuffle(groups)

    total_images = sum(len(paths) for paths in grouped_images.values())
    train_target = total_images * train_ratio
    val_target = total_images * val_ratio

    assignments: dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for group in groups:
        count = len(grouped_images[group])
        if split_counts["train"] + count <= train_target or split_counts["train"] == 0:
            split = "train"
        elif split_counts["val"] + count <= val_target or split_counts["val"] == 0:
            split = "val"
        else:
            split = "test"
        assignments[group] = split
        split_counts[split] += count

    return assignments


def write_yaml(output_dir: Path) -> None:
    dataset_root = output_dir.resolve().as_posix()
    lines = [
        f'path: "{dataset_root}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ratio_sum = args.train + args.val + args.test
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    source = args.source.resolve()
    images_dir = source / "images"
    labels_dir = (args.labels_dir or source / "labels-segment").resolve()
    output_dir = args.output.resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Segmentation label directory not found: {labels_dir}")

    if output_dir.exists():
        if not args.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"{output_dir} already exists. Re-run with --overwrite to recreate it.")
        if args.overwrite:
            shutil.rmtree(output_dir)

    images = collect_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    grouped_images: dict[str, list[Path]] = defaultdict(list)
    class_counts = {class_id: 0 for class_id in range(len(CLASS_NAMES))}
    total_objects = 0

    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(f"Missing segmentation label for {image.name}: {label}")
        object_count, class_ids = validate_segment_label(label, len(CLASS_NAMES))
        total_objects += object_count
        for class_id in class_ids:
            class_counts[class_id] += 1
        grouped_images[image_group(image)].append(image)

    assignments = split_groups(grouped_images, args.train, args.val, args.seed)
    summary = {
        "images": {"train": 0, "val": 0, "test": 0},
        "objects": total_objects,
        "groups": {"train": 0, "val": 0, "test": 0},
        "class_image_presence": {CLASS_NAMES[k]: v for k, v in class_counts.items()},
    }

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for group, paths in grouped_images.items():
        split = assignments[group]
        summary["groups"][split] += 1
        summary["images"][split] += len(paths)
        for image in paths:
            label = labels_dir / f"{image.stem}.txt"
            shutil.copy2(image, output_dir / "images" / split / image.name)
            shutil.copy2(label, output_dir / "labels" / split / label.name)

    write_yaml(output_dir)
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote dataset to {output_dir}")
    print(json.dumps(summary["images"], indent=2))


if __name__ == "__main__":
    main()
