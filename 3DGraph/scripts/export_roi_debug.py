import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scene_graph.config import SceneGraphConfig
from scene_graph.roi_debug import export_roi_debug_artifacts


def parse_args():
    config = SceneGraphConfig()
    parser = argparse.ArgumentParser(description="Export ROI/debug artifacts for one RGB-D frame.")
    parser.add_argument("--rgb", default=config.paths.rgb_image_path)
    parser.add_argument("--depth", default=config.paths.depth_image_path)
    parser.add_argument("--model", default=config.paths.yolo_model_path)
    parser.add_argument("--output-dir", default="3DGraph/outputs/roi_debug")
    parser.add_argument("--device", default=config.runtime.device)
    parser.add_argument("--keep-tables", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SceneGraphConfig()
    config.paths.rgb_image_path = args.rgb
    config.paths.depth_image_path = args.depth
    config.paths.yolo_model_path = args.model
    config.runtime.device = args.device
    config.perception.ignore_table_detections = not args.keep_tables
    export_roi_debug_artifacts(
        rgb_path=args.rgb,
        depth_path=args.depth,
        output_dir=args.output_dir,
        config=config,
        clear_existing=not args.keep_existing,
    )


if __name__ == "__main__":
    main()
