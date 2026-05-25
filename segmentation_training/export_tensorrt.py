"""Export a trained YOLO segmentation model to TensorRT for NVIDIA GPU deployment."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/bimacs-seg/yolo26m-seg_640/weights/best.pt"),
        help="Trained .pt checkpoint.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fp32", action="store_true", help="Export FP32 instead of the default FP16 engine.")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic input shapes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    exported = model.export(
        format="engine",
        imgsz=args.imgsz,
        device=args.device,
        half=not args.fp32,
        dynamic=args.dynamic,
    )
    print(f"Exported TensorRT engine: {exported}")


if __name__ == "__main__":
    main()
