import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scene_graph.config import SceneGraphConfig
from scene_graph.io import load_camera_intrinsics, load_depth_image_m, load_rgb_image, save_graph_json
from scene_graph.pipeline import SceneGraphPipeline


def parse_args():
    config = SceneGraphConfig()
    parser = argparse.ArgumentParser(description="Build a 3D scene graph for one RGB-D frame.")
    parser.add_argument("--rgb", default=config.paths.rgb_image_path)
    parser.add_argument("--depth", default=config.paths.depth_image_path)
    parser.add_argument("--model", default=config.paths.yolo_model_path)
    parser.add_argument("--output-dir", default=config.paths.single_frame_output_dir)
    parser.add_argument("--device", default=config.runtime.device)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--keep-tables", action="store_true")
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--tracking-certainty-threshold", type=float, default=config.tracking.certainty_threshold)
    parser.add_argument("--tracking-max-missing-frames", type=int, default=config.tracking.max_missing_frames)
    parser.add_argument("--tracking-max-hand-tracks", type=int, default=config.tracking.max_hand_tracks)
    parser.add_argument("--no-relations", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SceneGraphConfig()
    config.paths.rgb_image_path = args.rgb
    config.paths.depth_image_path = args.depth
    config.paths.yolo_model_path = args.model
    config.paths.single_frame_output_dir = args.output_dir
    config.runtime.device = args.device
    config.hands.enabled = not args.no_hands
    config.perception.ignore_table_detections = not args.keep_tables
    config.tracking.enabled = not args.no_tracking
    config.tracking.certainty_threshold = args.tracking_certainty_threshold
    config.tracking.max_missing_frames = args.tracking_max_missing_frames
    config.tracking.max_hand_tracks = args.tracking_max_hand_tracks
    config.relations.enabled = not args.no_relations

    rgb = load_rgb_image(args.rgb)
    depth_m = load_depth_image_m(args.depth, config)
    intrinsics = load_camera_intrinsics(rgb.shape[:2], config)
    frame_id = args.frame_id or Path(args.rgb).stem

    with SceneGraphPipeline(config) as pipeline:
        graph = pipeline.process_frame(
            rgb,
            depth_m,
            intrinsics,
            frame_id=frame_id,
            metadata={"rgb_path": args.rgb, "depth_path": args.depth},
        )

    output_path = Path(args.output_dir) / f"scene_graph_{frame_id}.json"
    saved_path = save_graph_json(graph, output_path)
    print(f"Saved graph JSON to: {saved_path}")
    print(f"Nodes: {len(graph['nodes'])}, edges: {len(graph['edges'])}")


if __name__ == "__main__":
    main()
