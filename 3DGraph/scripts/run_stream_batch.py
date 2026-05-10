import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scene_graph.config import SceneGraphConfig
from scene_graph.pipeline import process_first_task_takes


def parse_args():
    config = SceneGraphConfig()
    parser = argparse.ArgumentParser(description="Build 3D scene graphs for BIMACS takes.")
    parser.add_argument("--dataset-root", default=config.stream.dataset_root)
    parser.add_argument("--subject", default=config.stream.subject_name)
    parser.add_argument("--num-takes", type=int, default=config.stream.num_takes)
    parser.add_argument("--frame-stride", type=int, default=config.stream.frame_stride)
    parser.add_argument("--max-frames-per-take", type=int, default=config.stream.max_frames_per_take)
    parser.add_argument("--output-dir", default=config.paths.stream_output_dir)
    parser.add_argument("--model", default=config.paths.yolo_model_path)
    parser.add_argument("--device", default=config.runtime.device)
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--keep-tables", action="store_true")
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--tracking-certainty-threshold", type=float, default=config.tracking.certainty_threshold)
    parser.add_argument("--tracking-max-missing-frames", type=int, default=config.tracking.max_missing_frames)
    parser.add_argument("--tracking-max-hand-tracks", type=int, default=config.tracking.max_hand_tracks)
    parser.add_argument("--no-relations", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SceneGraphConfig()
    config.stream.dataset_root = args.dataset_root
    config.stream.subject_name = args.subject
    config.stream.num_takes = args.num_takes
    config.stream.frame_stride = args.frame_stride
    config.stream.max_frames_per_take = args.max_frames_per_take
    config.stream.overwrite_existing = not args.keep_existing
    config.paths.stream_output_dir = args.output_dir
    config.paths.stream_graph_root = args.output_dir
    config.paths.yolo_model_path = args.model
    config.runtime.device = args.device
    config.hands.enabled = not args.no_hands
    config.perception.ignore_table_detections = not args.keep_tables
    config.tracking.enabled = not args.no_tracking
    config.tracking.certainty_threshold = args.tracking_certainty_threshold
    config.tracking.max_missing_frames = args.tracking_max_missing_frames
    config.tracking.max_hand_tracks = args.tracking_max_hand_tracks
    config.relations.enabled = not args.no_relations

    index, index_path = process_first_task_takes(config)
    total_written = sum(take["frames_written"] for take in index["takes"])
    total_failed = sum(take["frames_failed"] for take in index["takes"])
    print(f"Wrote {total_written} frame graphs.")
    print(f"Failed/skipped {total_failed} frames.")
    print(f"Saved stream index to: {index_path}")


if __name__ == "__main__":
    main()
