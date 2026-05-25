from __future__ import annotations

import numpy as np

from .config import SceneGraphConfig
from .hands import disambiguate_duplicate_hand_labels
from .perception import should_ignore_detection_class
from .relations import extract_relations
from .schemas import validate_scene_graph
from .tracking import TRACKING_MASK_KEY, TemporalNodeTracker, XMemPrediction


def _test_frame_context(shape=(240, 640)):
    h, w = shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32)
    intrinsics = {
        "fx": 600.0,
        "fy": 600.0,
        "cx": float((w - 1) / 2.0),
        "cy": float((h - 1) / 2.0),
    }
    return rgb, depth, intrinsics


def _rect_mask(center_2d, shape=(240, 640), half_size=12):
    x, y = (int(round(center_2d[0])), int(round(center_2d[1])))
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    x1 = max(0, x - half_size)
    x2 = min(w, x + half_size + 1)
    y1 = max(0, y - half_size)
    y2 = min(h, y + half_size + 1)
    mask[y1:y2, x1:x2] = True
    return mask


class _FakeXMemEngine:
    def __init__(self):
        self.index_mask = None

    def reset(self):
        self.index_mask = None

    def step(self, rgb_image, label_mask=None, valid_labels=None, labels=None):
        h, w = rgb_image.shape[:2]
        labels = labels or []
        if label_mask is not None:
            self.index_mask = np.asarray(label_mask, dtype=np.int32).copy()
        elif self.index_mask is None:
            self.index_mask = np.zeros((h, w), dtype=np.int32)

        probabilities = np.zeros((len(labels) + 1, h, w), dtype=np.float32)
        probabilities[0] = 1.0
        for label in labels:
            mask = self.index_mask == label
            probabilities[0][mask] = 0.05
            probabilities[label][mask] = 0.95
        return XMemPrediction(index_mask=self.index_mask.copy(), probabilities=probabilities)


def _sample_hand(label, node_id, center_2d, center_3d, confidence=0.9):
    x, y = center_2d
    class_name = "hand" if label == "hand" else f"{label}_hand"
    return {
        "id": node_id,
        "class_name": class_name,
        "node_type": "hand",
        "source_model": "mediapipe_hands",
        "bbox_2d": [x - 15.0, y - 20.0, x + 15.0, y + 20.0],
        "center_2d": [float(x), float(y)],
        "center_3d_m": [float(v) for v in center_3d],
        "median_depth_m": float(center_3d[2]),
        "bbox_3d_m": None,
        "confidence": confidence,
        "attributes": {
            "landmarks_2d": [[float(x), float(y)]] + [[float(x), float(y)] for _ in range(20)],
            "normalized_handedness_label": label.title(),
        },
        TRACKING_MASK_KEY: _rect_mask(center_2d),
    }


def check_graph_integrity(graph):
    return validate_scene_graph(graph)


def relation_extraction_handles_missing_depth(config=None):
    nodes = [
        {"id": "object_0", "class_name": "bowl", "node_type": "object", "center_3d_m": None},
        {"id": "object_1", "class_name": "cup", "node_type": "object"},
    ]
    extract_relations(nodes, config=config)
    return True


def perception_ignores_tables_by_default():
    config = SceneGraphConfig()
    return (
        should_ignore_detection_class("dining table", config)
        and should_ignore_detection_class("table", config)
        and should_ignore_detection_class("tv", config)
        and not should_ignore_detection_class("bottle", config)
    )


def relation_extraction_ignores_person_nodes_but_keeps_hands(config=None):
    nodes = [
        {"id": "person_0", "class_name": "person", "node_type": "person", "center_3d_m": [0.0, 0.0, 1.0]},
        {"id": "left_hand", "class_name": "left_hand", "node_type": "hand", "center_3d_m": [0.01, 0.0, 1.0]},
        {"id": "object_0", "class_name": "bowl", "node_type": "object", "center_3d_m": [0.02, 0.0, 1.0]},
    ]
    edges = extract_relations(nodes, config=config)
    has_hand_relation = any(
        edge["relation"] == "hand_near_object"
        and {edge["source"], edge["target"]} == {"left_hand", "object_0"}
        for edge in edges
    )
    has_person_relation = any(
        edge["source"] == "person_0" or edge["target"] == "person_0"
        for edge in edges
    )
    return has_hand_relation and not has_person_relation


def relation_extraction_never_emits_below(config=None):
    nodes = [
        {"id": "object_0", "class_name": "bowl", "node_type": "object", "center_3d_m": [0.0, 0.3, 1.0]},
        {"id": "object_1", "class_name": "cup", "node_type": "object", "center_3d_m": [0.0, 0.0, 1.0]},
    ]
    edges = extract_relations(nodes, config=config)
    relations = {edge["relation"] for edge in edges}
    above_edges = [
        edge for edge in edges
        if edge["source"] == "object_1" and edge["target"] == "object_0" and edge["relation"] == "above"
    ]
    return "below" not in relations and bool(above_edges)


def tracking_carries_forward_missing_nodes():
    config = SceneGraphConfig()
    config.tracking.certainty_threshold = 0.55
    tracker = TemporalNodeTracker(config, xmem_engine=_FakeXMemEngine())
    rgb, depth, intrinsics = _test_frame_context()
    node = {
        "id": "object_0",
        "class_name": "bottle",
        "node_type": "object",
        "source_model": "yolo_segmentation",
        "bbox_2d": [10.0, 20.0, 50.0, 90.0],
        "center_2d": [30.0, 55.0],
        "center_3d_m": [0.1, 0.2, 1.5],
        "median_depth_m": 1.5,
        "bbox_3d_m": {"size_xyz": [0.1, 0.2, 0.1]},
        "confidence": 0.95,
        "attributes": {},
        TRACKING_MASK_KEY: _rect_mask([30.0, 55.0]),
    }
    first_nodes, _ = tracker.update([node], frame_id=0, rgb_image=rgb, depth_image=depth, intrinsics=intrinsics)
    carried_nodes, _ = tracker.update([], frame_id=1, rgb_image=rgb, depth_image=depth, intrinsics=intrinsics)
    return (
        len(first_nodes) == 1
        and len(carried_nodes) == 1
        and carried_nodes[0]["tracking"]["carried_forward"]
        and not carried_nodes[0]["tracking"]["detected_this_frame"]
        and carried_nodes[0]["source_model"] == "xmem_tracking"
        and carried_nodes[0]["center_3d_m"] is not None
    )


def tracking_keeps_hand_nodes():
    config = SceneGraphConfig()
    tracker = TemporalNodeTracker(config, xmem_engine=_FakeXMemEngine())
    rgb, depth, intrinsics = _test_frame_context()
    hand = {
        "id": "hand",
        "class_name": "hand",
        "node_type": "hand",
        "source_model": "mediapipe_hands",
        "bbox_2d": [10.0, 20.0, 50.0, 80.0],
        "center_2d": [30.0, 50.0],
        "center_3d_m": [0.1, 0.2, 1.2],
        "median_depth_m": 1.2,
        "bbox_3d_m": None,
        "confidence": 0.9,
        "attributes": {},
        TRACKING_MASK_KEY: _rect_mask([30.0, 50.0]),
    }
    first_nodes, _ = tracker.update([hand], frame_id=0, rgb_image=rgb, depth_image=depth, intrinsics=intrinsics)
    carried_nodes, _ = tracker.update([], frame_id=1, rgb_image=rgb, depth_image=depth, intrinsics=intrinsics)
    return (
        first_nodes[0]["class_name"] == "right_hand"
        and carried_nodes[0]["node_type"] == "hand"
        and carried_nodes[0]["tracking"]["carried_forward"]
    )


def tracking_matches_hand_when_label_flips():
    config = SceneGraphConfig()
    tracker = TemporalNodeTracker(config, xmem_engine=_FakeXMemEngine())
    rgb, depth, intrinsics = _test_frame_context()
    first_nodes, _ = tracker.update([
        _sample_hand("hand", "hand", [100.0, 160.0], [0.1, 0.2, 1.2])
    ], frame_id=0, rgb_image=rgb, depth_image=depth, intrinsics=intrinsics)
    second_nodes, _ = tracker.update([
        _sample_hand("hand", "hand", [112.0, 166.0], [0.13, 0.2, 1.23])
    ], frame_id=1, rgb_image=rgb, depth_image=depth, intrinsics=intrinsics)
    return (
        len(first_nodes) == 1
        and len(second_nodes) == 1
        and first_nodes[0]["id"] == second_nodes[0]["id"]
        and second_nodes[0]["class_name"] == "right_hand"
        and second_nodes[0]["attributes"]["observed_class_name"] == "hand"
    )


def tracking_caps_hands_to_two():
    config = SceneGraphConfig()
    tracker = TemporalNodeTracker(config, xmem_engine=_FakeXMemEngine())
    rgb, depth, intrinsics = _test_frame_context()
    nodes, diagnostics = tracker.update([
        _sample_hand("hand", "hand", [100.0, 160.0], [0.1, 0.2, 1.2], confidence=0.9),
        _sample_hand("hand", "hand_1", [320.0, 160.0], [0.4, 0.2, 1.2], confidence=0.8),
        _sample_hand("hand", "hand_2", [520.0, 160.0], [0.8, 0.2, 1.2], confidence=0.7),
    ], frame_id=0, rgb_image=rgb, depth_image=depth, intrinsics=intrinsics)
    hand_nodes = [node for node in nodes if node["node_type"] == "hand"]
    return len(hand_nodes) == 2 and diagnostics["dropped_extra_hand_detections"] == 1


def duplicate_hand_labels_are_disambiguated_by_position():
    config = SceneGraphConfig()
    nodes = [
        _sample_hand("left", "left_hand", [100.0, 160.0], [0.1, 0.2, 1.2]),
        _sample_hand("left", "left_hand_1", [340.0, 160.0], [0.4, 0.2, 1.2]),
    ]
    nodes, diagnostics = disambiguate_duplicate_hand_labels(nodes, config)
    by_x = sorted(nodes, key=lambda node: node["center_2d"][0])
    return (
        by_x[0]["class_name"] == "right_hand"
        and by_x[1]["class_name"] == "left_hand"
        and by_x[0]["attributes"]["corrected_handedness_label"] == "Right"
        and bool(diagnostics)
    )
