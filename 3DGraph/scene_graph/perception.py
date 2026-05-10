from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from . import geometry
from .config import DEFAULT_CONFIG
from .schemas import make_node
from .io import resolve_path


def check_compute_device(device):
    if device.lower() == "cpu":
        print("Using CPU for YOLO inference.")
        return

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to use CUDA for YOLO inference.") from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"DEVICE is set to {device}, but torch.cuda.is_available() is False. "
            "Use the project .venv and install a CUDA-enabled PyTorch build, or set device='cpu'."
        )

    device_index = 0
    if ":" in device:
        device_index = int(device.split(":", 1)[1])

    if device.startswith("cuda"):
        print(f"Using GPU for YOLO inference: {torch.cuda.get_device_name(device_index)} ({device})")
    else:
        print(f"Using YOLO device: {device}")


def load_yolo_model(config=DEFAULT_CONFIG):
    return YOLO(str(resolve_path(config.paths.yolo_model_path)))


def node_type_for_class(class_name, config=DEFAULT_CONFIG):
    lower = class_name.lower()
    if lower == config.depth.person_class_name:
        return "person"
    if lower in config.perception.table_classes:
        return "table"
    return "object"


def should_ignore_detection_class(class_name, config=DEFAULT_CONFIG):
    lower = str(class_name).lower()
    return config.perception.ignore_table_detections and lower in config.perception.table_classes


def compute_person_depth_barrier(detections, depth_m, config=DEFAULT_CONFIG):
    person_masks = [
        det["mask"]
        for det in detections
        if det["class_name"].lower() == config.depth.person_class_name
    ]
    if not person_masks:
        raise RuntimeError("No person detected. Cannot filter background behind the person.")

    combined_person_mask = np.any(np.stack(person_masks, axis=0), axis=0)
    person_depth_values = depth_m[combined_person_mask]
    person_depth_values = person_depth_values[
        np.isfinite(person_depth_values)
        & (person_depth_values > config.depth.min_m)
        & (person_depth_values < config.depth.max_m)
    ]
    if person_depth_values.size == 0:
        raise RuntimeError("Person detected, but no valid depth values were found inside the person mask.")

    person_front_depth = float(np.percentile(person_depth_values, config.depth.person_front_percentile))
    roi_depth_limit = person_front_depth + config.depth.person_behind_tolerance_m
    roi_mask = geometry.valid_depth_mask(depth_m, config) & (depth_m <= roi_depth_limit)
    roi_filter = {
        "enabled": True,
        "method": "person_depth_barrier",
        "person_front_percentile": config.depth.person_front_percentile,
        "person_front_depth_m": person_front_depth,
        "roi_depth_limit_m": roi_depth_limit,
        "person_behind_tolerance_m": config.depth.person_behind_tolerance_m,
        "object_in_front_tolerance_m": config.depth.object_in_front_tolerance_m,
    }
    return person_front_depth, roi_depth_limit, roi_mask, roi_filter


def filter_detection_mask(det, depth_m, roi_mask, person_front_depth, config=DEFAULT_CONFIG):
    class_name = det["class_name"].lower()
    mask = det["mask"]

    if class_name == config.depth.person_class_name:
        return mask & geometry.valid_depth_mask(depth_m, config), None

    mask_in_roi = mask & roi_mask
    clean_mask = geometry.clean_mask_by_depth(mask_in_roi, depth_m, config)
    median_depth = geometry.median_depth_in_mask(clean_mask, depth_m, config)
    if median_depth is None:
        return clean_mask, "no valid depth values after person-depth ROI filtering"
    if median_depth > person_front_depth + config.depth.object_in_front_tolerance_m:
        return clean_mask, (
            f"behind person barrier: median_depth={median_depth:.3f} m, "
            f"person_front={person_front_depth:.3f} m"
        )
    return clean_mask, None


def make_yolo_node(node_id, class_name, confidence, bbox_2d, mask, depth_m, intrinsics, config=DEFAULT_CONFIG):
    mask_area = int(mask.sum())
    median_depth_m = geometry.median_depth_in_mask(mask, depth_m, config)
    points_3d = geometry.mask_to_3d_points(mask, depth_m, intrinsics, config)

    if mask_area < config.perception.min_mask_area_pixels:
        return None, f"mask area too small ({mask_area} px)"
    if median_depth_m is None or points_3d is None:
        return None, "no valid depth values inside mask"

    geometry_3d = geometry.summarize_3d_points(points_3d, config)
    return make_node(
        node_id=node_id,
        class_name=class_name,
        node_type=node_type_for_class(class_name, config),
        source_model="yolo_segmentation",
        bbox_2d=bbox_2d,
        center_2d=geometry.mask_center(mask, bbox_2d),
        center_3d_m=geometry_3d["center_3d_m"],
        median_depth_m=median_depth_m,
        bbox_3d_m=geometry_3d["bbox_3d_m"],
        confidence=float(confidence),
        attributes={"mask_area": mask_area},
    ), None


def detection_node_id(node_type, counters):
    counters[node_type] = counters.get(node_type, 0) + 1
    return f"{node_type}_{counters[node_type] - 1}"


def detect_scene_nodes(
    rgb_image,
    depth_image,
    intrinsics,
    config=DEFAULT_CONFIG,
    model=None,
    verbose=False,
    return_diagnostics=False,
):
    h, w = rgb_image.shape[:2]
    if depth_image.shape[:2] != (h, w):
        depth_image = cv2.resize(depth_image, (w, h), interpolation=cv2.INTER_NEAREST)

    model = model or load_yolo_model(config)
    result = model.predict(rgb_image, device=config.runtime.device, verbose=verbose)[0]
    if result.masks is None:
        raise RuntimeError("YOLO produced no segmentation masks for this frame.")

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    class_names = result.names

    detections = []
    for det_index, mask in enumerate(masks):
        cls_id = int(boxes.cls[det_index].item())
        detections.append({
            "det_index": det_index,
            "class_name": class_names[cls_id],
            "confidence": float(boxes.conf[det_index].item()),
            "bbox_2d": geometry.bbox_to_list(boxes.xyxy[det_index].cpu().numpy()),
            "mask": geometry.resize_mask(mask, (h, w)),
        })

    person_front_depth, _, roi_mask, roi_filter = compute_person_depth_barrier(detections, depth_image, config)

    nodes = []
    skipped = []
    counters = {}
    for det in detections:
        if should_ignore_detection_class(det["class_name"], config):
            skipped.append((det["det_index"], det["class_name"], "ignored table detection by config"))
            continue

        filtered_mask, filter_reason = filter_detection_mask(det, depth_image, roi_mask, person_front_depth, config)
        if filter_reason is not None:
            skipped.append((det["det_index"], det["class_name"], filter_reason))
            continue

        node_type = node_type_for_class(det["class_name"], config)
        node, reason = make_yolo_node(
            node_id=detection_node_id(node_type, counters),
            class_name=det["class_name"],
            confidence=det["confidence"],
            bbox_2d=geometry.bbox_from_mask(filtered_mask, det["bbox_2d"]),
            mask=filtered_mask,
            depth_m=depth_image,
            intrinsics=intrinsics,
            config=config,
        )
        if node is None:
            skipped.append((det["det_index"], det["class_name"], reason))
            continue
        nodes.append(node)

    if return_diagnostics:
        return nodes, roi_filter, {"skipped_yolo_detections": skipped}
    return nodes
