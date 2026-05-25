import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scene_graph.config import SceneGraphConfig
from scene_graph.visualization import show_graph_file


def parse_args():
    config = SceneGraphConfig()
    parser = argparse.ArgumentParser(description="Visualize one 3D scene graph JSON file.")
    parser.add_argument(
        "--graph",
        default=f"{config.paths.single_frame_output_dir}/scene_graph_frame_72.json",
    )
    parser.add_argument("--screenshot", default=None)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--hide-boxes", action="store_true")
    parser.add_argument("--hide-edges", action="store_true")
    parser.add_argument("--hide-edge-labels", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-centroids", action="store_true")
    parser.add_argument("--hide-rgb-frame", action="store_true")
    parser.add_argument("--show-rgb-overlays", action="store_true")
    parser.add_argument("--rgb-preview-width", type=int, default=config.visualization.rgb_preview_max_width_px)
    return parser.parse_args()


def main():
    args = parse_args()
    config = SceneGraphConfig()
    config.visualization.show_3d_boxes = not args.hide_boxes
    config.visualization.show_edges = not args.hide_edges
    config.visualization.show_edge_labels = not args.hide_edge_labels
    config.visualization.show_labels = not args.hide_labels
    config.visualization.show_centroids = not args.hide_centroids
    config.visualization.show_rgb_frame = not args.hide_rgb_frame
    config.visualization.show_rgb_overlays = args.show_rgb_overlays
    config.visualization.rgb_preview_max_width_px = args.rgb_preview_width
    show_graph_file(
        args.graph,
        screenshot_path=args.screenshot,
        show_window=not args.no_window,
        config=config,
    )


if __name__ == "__main__":
    main()
