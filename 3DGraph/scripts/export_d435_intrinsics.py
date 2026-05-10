import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scene_graph.camera import export_d435_color_intrinsics
from scene_graph.config import SceneGraphConfig


def parse_args():
    config = SceneGraphConfig()
    parser = argparse.ArgumentParser(description="Export RealSense D435 color intrinsics to JSON.")
    parser.add_argument("--output", default=config.paths.camera_intrinsics_path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    data, output_path = export_d435_color_intrinsics(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(f"Saved D435 color intrinsics to: {output_path}")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
