"""Train YOLO26m instance segmentation on the packaged BIMACS dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("yolo26m-seg.pt"), help="Pretrained YOLO segment model.")
    parser.add_argument("--data", type=Path, default=Path("datasets/bimacs-seg/data.yaml"), help="Dataset YAML.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="-1", help="Batch size. Use -1 for Ultralytics auto-batch.")
    parser.add_argument("--device", default="0", help="CUDA device id, e.g. 0.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/bimacs-seg"))
    parser.add_argument("--name", default="yolo26m-seg_640")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM/disk if your machine has room.")
    parser.add_argument("--resume", action="store_true", help="Resume a previous interrupted run.")
    return parser.parse_args()


def coerce_batch(value: str) -> int | float:
    parsed = float(value)
    if parsed.is_integer():
        return int(parsed)
    return parsed


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    project_path = args.project.resolve()

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=coerce_batch(args.batch),
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        patience=args.patience,
        plots=True,
        cos_lr=True,
        deterministic=True,
        cache=args.cache,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
