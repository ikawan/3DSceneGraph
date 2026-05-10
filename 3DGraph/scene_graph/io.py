from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_CONFIG
from .geometry import valid_depth_values


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_rgb_image(path):
    path = resolve_path(path)
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")
    return rgb


def decode_color_depth(depth_raw, config=DEFAULT_CONFIG):
    if depth_raw.ndim != 3:
        return depth_raw.astype(np.float32), "raw"

    channels = depth_raw[:, :, :3].astype(np.uint32)
    b = channels[:, :, 0]
    g = channels[:, :, 1]
    r = channels[:, :, 2]

    candidates = {
        "rgb16_little": (r + (g << 8) + (b << 16)).astype(np.float32),
        "bgr16_little": (b + (g << 8) + (r << 16)).astype(np.float32),
        "google_rgb_24bit": (r * 256.0 + g + b / 256.0).astype(np.float32),
        "raw": cv2.cvtColor(depth_raw[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32),
    }

    if config.depth.encoding != "auto":
        if config.depth.encoding not in candidates:
            raise ValueError(f"Unknown depth encoding: {config.depth.encoding}")
        return candidates[config.depth.encoding], config.depth.encoding

    viable = []
    for name, decoded in candidates.items():
        depth_m = decoded * config.depth.scale
        values = valid_depth_values(depth_m, config)
        if values.size > 0:
            viable.append((abs(float(np.median(values)) - 2.5), name, decoded))

    if not viable:
        return candidates["rgb16_little"], "rgb16_little"

    _, chosen_name, chosen_depth = min(viable, key=lambda item: item[0])
    return chosen_depth, chosen_name


def load_depth_image_m(path, config=DEFAULT_CONFIG, verbose=True):
    path = resolve_path(path)
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not load depth image: {path}")

    depth_raw, encoding_used = decode_color_depth(depth_raw, config)
    depth_m = depth_raw.astype(np.float32) * config.depth.scale
    valid = valid_depth_values(depth_m, config)
    if valid.size == 0:
        raise RuntimeError(
            f"No valid depth values found in {path}. "
            f"Check depth encoding/scale. Used encoding={encoding_used}."
        )

    if verbose:
        print(f"Loaded depth with encoding={encoding_used}; median={np.median(valid):.3f} m")
    return depth_m


def load_camera_intrinsics(image_shape_hw, config=DEFAULT_CONFIG):
    path = resolve_path(config.paths.camera_intrinsics_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        required = ("fx", "fy", "cx", "cy")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Camera intrinsics file is missing keys: {missing}")
        return {
            "fx": float(data["fx"]),
            "fy": float(data["fy"]),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
            "source": str(path),
            "model": data.get("model", "realsense_d435"),
            "aligned_depth_to": data.get("aligned_depth_to", "color"),
        }

    h, w = image_shape_hw
    fov_x, fov_y = config.camera.fallback_depth_fov_deg
    fx = w / (2.0 * math.tan(math.radians(fov_x) / 2.0))
    fy = h / (2.0 * math.tan(math.radians(fov_y) / 2.0))
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float((w - 1) / 2.0),
        "cy": float((h - 1) / 2.0),
        "source": "fallback_from_d435_depth_fov",
        "model": config.camera.fallback_model,
        "aligned_depth_to": "depth",
        "warning": "Approximate fallback intrinsics; replace with calibrated D435 intrinsics.",
    }


def save_graph_json(graph, path):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    return path


def load_graph_json(path):
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = {}
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row["value"]
            if row["type"] == "unsigned int":
                value = int(value)
            metadata[row["name"]] = value
    return metadata


def first_task_path(config=DEFAULT_CONFIG):
    subject_dir = resolve_path(config.stream.dataset_root) / config.stream.subject_name
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject folder not found: {subject_dir}")

    tasks = sorted([path for path in subject_dir.iterdir() if path.is_dir()])
    if not tasks:
        raise FileNotFoundError(f"No task folders found in: {subject_dir}")
    return tasks[0]


def selected_take_paths(task_path, config=DEFAULT_CONFIG):
    takes = sorted([path for path in task_path.iterdir() if path.is_dir()])
    if not takes:
        raise FileNotFoundError(f"No take folders found in: {task_path}")
    return takes[: config.stream.num_takes]


def iter_bimacs_take_frames(take_path, config=DEFAULT_CONFIG):
    rgb_root = Path(take_path) / "rgb"
    depth_root = Path(take_path) / "depth"
    rgb_metadata = parse_metadata(rgb_root / "metadata.csv")
    depth_metadata = parse_metadata(depth_root / "metadata.csv")

    frame_count = min(rgb_metadata["frameCount"], depth_metadata["frameCount"])
    frames_per_chunk = rgb_metadata["framesPerChunk"]

    frame_ids = range(0, frame_count, config.stream.frame_stride)
    if config.stream.max_frames_per_take is not None:
        frame_ids = list(frame_ids)[: config.stream.max_frames_per_take]

    for frame_id in frame_ids:
        chunk_id = frame_id // frames_per_chunk
        frame_in_chunk = frame_id % frames_per_chunk
        rgb_path = rgb_root / f"chunk_{chunk_id}" / f"frame_{frame_in_chunk}.png"
        depth_path = depth_root / f"chunk_{chunk_id}" / f"frame_{frame_in_chunk}.png"
        missing_reason = None
        if not rgb_path.exists() or not depth_path.exists():
            missing_reason = "missing rgb/depth frame file"
        yield frame_id, rgb_path, depth_path, missing_reason


def output_path_for_frame(output_root, subject_name, task_name, take_name, frame_id):
    return Path(output_root) / subject_name / task_name / take_name / f"frame_{int(frame_id):06d}.json"


def write_stream_index(index, output_root):
    path = resolve_path(Path(output_root) / "stream_index.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return path
