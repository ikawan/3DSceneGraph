import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scene_graph.config import SceneGraphConfig
from scene_graph.visualization import show_stream_from_root


def parse_args():
    config = SceneGraphConfig()
    parser = argparse.ArgumentParser(description="Visualize 3D scene graph JSON frames.")
    parser.add_argument("--root", default=config.paths.stream_graph_root)
    parser.add_argument("--task", default="task_1_k_cooking")
    parser.add_argument("--take", default=None)
    parser.add_argument("--fps", type=int, default=config.visualization.playback_fps)
    parser.add_argument("--max-frames", type=int, default=config.visualization.max_frames)
    parser.add_argument("--hide-boxes", action="store_true")
    parser.add_argument("--hide-edges", action="store_true")
    parser.add_argument("--hide-edge-labels", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-centroids", action="store_true")
    parser.add_argument("--hide-rgb-frame", action="store_true")
    parser.add_argument("--show-rgb-overlays", action="store_true")
    parser.add_argument("--rgb-preview-width", type=int, default=config.visualization.rgb_preview_max_width_px)
    parser.add_argument("--no-loop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SceneGraphConfig()
    config.paths.stream_graph_root = args.root
    config.visualization.playback_fps = args.fps
    config.visualization.max_frames = args.max_frames
    config.visualization.show_3d_boxes = not args.hide_boxes
    config.visualization.show_edges = not args.hide_edges
    config.visualization.show_edge_labels = not args.hide_edge_labels
    config.visualization.show_labels = not args.hide_labels
    config.visualization.show_centroids = not args.hide_centroids
    config.visualization.show_rgb_frame = not args.hide_rgb_frame
    config.visualization.show_rgb_overlays = args.show_rgb_overlays
    config.visualization.rgb_preview_max_width_px = args.rgb_preview_width
    config.visualization.loop_playback = not args.no_loop

    show_stream_from_root(root=args.root, task_name=args.task, take_name=args.take, config=config)


if __name__ == "__main__":
    main()
