from __future__ import annotations

import time
from pathlib import Path

from tqdm import tqdm

from .config import DEFAULT_CONFIG
from .graph_builder import build_scene_graph_from_arrays
from .hands import create_hand_processor
from .io import (
    first_task_path,
    iter_bimacs_take_frames,
    load_camera_intrinsics,
    load_depth_image_m,
    load_rgb_image,
    output_path_for_frame,
    save_graph_json,
    selected_take_paths,
    write_stream_index,
)
from .perception import check_compute_device, load_yolo_model
from .tracking import TemporalNodeTracker


class SceneGraphPipeline:
    def __init__(self, config=DEFAULT_CONFIG, yolo_model=None, hands_processor=None):
        self.config = config
        check_compute_device(config.runtime.device)
        self.yolo_model = yolo_model or load_yolo_model(config)
        self.hands_processor = hands_processor
        self.tracker = TemporalNodeTracker(config) if config.tracking.enabled else None
        self._owns_hands = False
        if self.hands_processor is None and config.hands.enabled:
            self.hands_processor = create_hand_processor(config, static_image_mode=False)
            self._owns_hands = True

    def close(self):
        if self._owns_hands and self.hands_processor is not None:
            self.hands_processor.close()
            self.hands_processor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def reset_tracking(self):
        if self.tracker is not None:
            self.tracker.reset()

    def process_frame(self, rgb_image, depth_image, intrinsics, frame_id, timestamp=None, metadata=None, previous_graph=None):
        return build_scene_graph_from_arrays(
            rgb_image=rgb_image,
            depth_image=depth_image,
            intrinsics=intrinsics,
            frame_id=frame_id,
            timestamp=timestamp,
            previous_graph=previous_graph,
            config=self.config,
            yolo_model=self.yolo_model,
            hands_processor=self.hands_processor,
            tracker=self.tracker,
            metadata=metadata,
            verbose=self.config.runtime.verbose_yolo,
        )

    def process_frame_paths(
        self,
        rgb_path,
        depth_path,
        intrinsics=None,
        frame_id=0,
        timestamp=None,
        metadata=None,
        previous_graph=None,
    ):
        rgb = load_rgb_image(rgb_path)
        depth_m = load_depth_image_m(depth_path, self.config, verbose=False)
        intrinsics = intrinsics or load_camera_intrinsics(rgb.shape[:2], self.config)
        return self.process_frame(
            rgb,
            depth_m,
            intrinsics,
            frame_id,
            timestamp=timestamp,
            metadata=metadata,
            previous_graph=previous_graph,
        )

    def process_take(self, take_path, output_root, subject_name, task_name):
        self.reset_tracking()
        frame_items = list(iter_bimacs_take_frames(take_path, self.config))
        sample_rgb = load_rgb_image(frame_items[0][1])
        intrinsics = load_camera_intrinsics(sample_rgb.shape[:2], self.config)
        summary = {
            "take": Path(take_path).name,
            "frames_seen": len(frame_items),
            "frames_written": 0,
            "frames_failed": 0,
            "node_class_counts": {},
            "output_dir": str(Path(output_root) / subject_name / task_name / Path(take_path).name),
        }

        previous_graph = None
        progress = tqdm(frame_items, desc=f"{task_name}/{Path(take_path).name}", unit="frame")
        for frame_id, rgb_path, depth_path, missing_reason in progress:
            output_path = output_path_for_frame(output_root, subject_name, task_name, Path(take_path).name, frame_id)
            if output_path.exists() and not self.config.stream.overwrite_existing:
                summary["frames_written"] += 1
                continue
            if missing_reason is not None:
                summary["frames_failed"] += 1
                continue

            try:
                graph = self.process_frame_paths(
                    rgb_path,
                    depth_path,
                    intrinsics=intrinsics,
                    frame_id=frame_id,
                    metadata={
                        "subject": subject_name,
                        "task": task_name,
                        "take": Path(take_path).name,
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_path),
                    },
                    previous_graph=previous_graph,
                )
                save_graph_json(graph, output_path)
                previous_graph = graph
                summary["frames_written"] += 1
                for node in graph["nodes"]:
                    class_name = node["class_name"]
                    summary["node_class_counts"][class_name] = summary["node_class_counts"].get(class_name, 0) + 1
                progress.set_postfix(nodes=len(graph["nodes"]), edges=len(graph["edges"]))
            except Exception as exc:
                summary["frames_failed"] += 1
                progress.set_postfix(error=type(exc).__name__)

        return summary


def process_first_task_takes(config=DEFAULT_CONFIG):
    output_root = Path(config.paths.stream_output_dir)
    task_path = first_task_path(config)
    takes = selected_take_paths(task_path, config)
    subject_name = config.stream.subject_name
    start_time = time.time()

    index = {
        "subject": subject_name,
        "task": task_path.name,
        "num_takes_requested": config.stream.num_takes,
        "frame_stride": config.stream.frame_stride,
        "max_frames_per_take": config.stream.max_frames_per_take,
        "add_hand_nodes": config.hands.enabled,
        "tracking_enabled": config.tracking.enabled,
        "takes": [],
    }

    with SceneGraphPipeline(config) as pipeline:
        for take_path in takes:
            index["takes"].append(
                pipeline.process_take(
                    take_path=take_path,
                    output_root=output_root,
                    subject_name=subject_name,
                    task_name=task_path.name,
                )
            )

    index["elapsed_seconds"] = time.time() - start_time
    index_path = write_stream_index(index, output_root)
    return index, index_path
