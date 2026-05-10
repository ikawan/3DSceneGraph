from __future__ import annotations

import time

from .config import DEFAULT_CONFIG
from .hands import detect_hand_nodes
from .perception import detect_scene_nodes
from .relations import extract_relations
from .schemas import make_scene_graph, validate_scene_graph


def build_scene_graph_from_arrays(
    rgb_image,
    depth_image,
    intrinsics,
    frame_id,
    timestamp=None,
    previous_graph=None,
    config=DEFAULT_CONFIG,
    yolo_model=None,
    hands_processor=None,
    tracker=None,
    metadata=None,
    verbose=False,
):
    timestamp = time.time() if timestamp is None else timestamp
    scene_nodes, roi_filter, perception_diag = detect_scene_nodes(
        rgb_image,
        depth_image,
        intrinsics,
        config=config,
        model=yolo_model,
        verbose=verbose,
        return_diagnostics=True,
    )
    hand_nodes, hand_diag = detect_hand_nodes(
        rgb_image,
        depth_image,
        intrinsics,
        config=config,
        hands_processor=hands_processor,
        return_diagnostics=True,
    )

    nodes = scene_nodes + hand_nodes
    if tracker is not None:
        nodes, tracking_diag = tracker.update(nodes, frame_id=frame_id, timestamp=timestamp)
    else:
        tracking_diag = {"enabled": False}

    edges = extract_relations(nodes, previous_graph=previous_graph, config=config)

    graph_metadata = {
        "models": {
            "yolo_segmentation": config.paths.yolo_model_path,
            "hands": "mediapipe_hands" if config.hands.enabled else None,
        },
        "pipeline_version": config.runtime.pipeline_version,
        "skipped_yolo_detections": len(perception_diag.get("skipped_yolo_detections", [])),
        "skipped_hands": len(hand_diag.get("skipped_hands", [])),
        "tracking": tracking_diag,
    }
    if metadata:
        graph_metadata.update(metadata)

    graph = make_scene_graph(
        frame_id=frame_id,
        timestamp=timestamp,
        camera_intrinsics=intrinsics,
        roi_filter=roi_filter,
        nodes=nodes,
        edges=edges,
        metadata=graph_metadata,
    )

    validation_errors = validate_scene_graph(graph)
    if validation_errors:
        raise RuntimeError("Invalid scene graph: " + "; ".join(validation_errors))

    return graph
