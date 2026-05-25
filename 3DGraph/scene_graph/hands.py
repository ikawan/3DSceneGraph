from __future__ import annotations

import cv2
import numpy as np

from . import geometry
from .config import DEFAULT_CONFIG
from .schemas import make_node
from .tracking import TRACKING_MASK_KEY


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def create_hand_processor(config=DEFAULT_CONFIG, static_image_mode=True):
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe could not be imported, so hand nodes cannot be created. "
            "Install project dependencies in the active .venv."
        ) from exc

    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=config.hands.max_num_hands,
        min_detection_confidence=config.hands.detection_confidence,
        min_tracking_confidence=config.hands.tracking_confidence,
    )


def hand_landmarks_to_pixels(hand_landmarks, image_shape_hw):
    h, w = image_shape_hw
    points = []
    for landmark in hand_landmarks.landmark:
        x = min(max(landmark.x * (w - 1), 0.0), float(w - 1))
        y = min(max(landmark.y * (h - 1), 0.0), float(h - 1))
        points.append([float(x), float(y)])
    return points


def normalize_handedness_label(label, config=DEFAULT_CONFIG):
    if not config.hands.swap_mediapipe_handedness:
        return label
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def hand_node_id(label, counters):
    label = str(label or "hand").lower()
    if label == "hand" or label.endswith("_hand"):
        base = label
    else:
        base = f"{label}_hand"
    counters[base] = counters.get(base, 0) + 1
    if counters[base] == 1:
        return base
    return f"{base}_{counters[base] - 1}"


def hand_label_from_class_name(class_name):
    lower = str(class_name or "").lower()
    if lower.startswith("left"):
        return "Left"
    if lower.startswith("right"):
        return "Right"
    return None


def hand_reference_x(node):
    landmarks = node.get("attributes", {}).get("landmarks_2d") or node.get("landmarks_2d")
    if landmarks and len(landmarks) > 0 and len(landmarks[0]) >= 2:
        return float(landmarks[0][0])  # Landmark 0 is the wrist.
    return float(node["center_2d"][0])


def set_hand_label(node, label, reason):
    label = str(label)
    old_class_name = node.get("class_name")
    new_class_name = f"{label.lower()}_hand"
    node["id"] = new_class_name
    node["class_name"] = new_class_name
    attrs = node.setdefault("attributes", {})
    attrs["corrected_handedness_label"] = label
    attrs["handedness_disambiguation"] = reason
    attrs["previous_class_name"] = old_class_name
    node["corrected_handedness_label"] = label
    node["handedness_disambiguation"] = reason


def disambiguate_duplicate_hand_labels(nodes, config=DEFAULT_CONFIG):
    if not config.hands.disambiguate_duplicate_handedness or len(nodes) != 2:
        return nodes, []

    labels = [hand_label_from_class_name(node.get("class_name")) for node in nodes]
    if labels[0] is None or labels[1] is None or labels[0] != labels[1]:
        return nodes, []

    sorted_nodes = sorted(nodes, key=hand_reference_x)
    if config.hands.swap_mediapipe_handedness:
        image_left_label, image_right_label = "Right", "Left"
    else:
        image_left_label, image_right_label = "Left", "Right"

    set_hand_label(
        sorted_nodes[0],
        image_left_label,
        "duplicate_mediapipe_label_resolved_by_wrist_x",
    )
    set_hand_label(
        sorted_nodes[1],
        image_right_label,
        "duplicate_mediapipe_label_resolved_by_wrist_x",
    )
    return nodes, [
        (
            "duplicate_handedness",
            f"resolved two {labels[0]} hands by wrist x: image-left={image_left_label}, image-right={image_right_label}",
        )
    ]


def hand_landmark_mask(landmarks_2d, image_shape_hw, config=DEFAULT_CONFIG):
    h, w = image_shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if not landmarks_2d:
        return mask.astype(bool)

    points = np.asarray(landmarks_2d, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return mask.astype(bool)

    points_int = np.round(points).astype(np.int32)
    points_int[:, 0] = np.clip(points_int[:, 0], 0, w - 1)
    points_int[:, 1] = np.clip(points_int[:, 1], 0, h - 1)

    if len(points_int) >= 3:
        hull = cv2.convexHull(points_int)
        cv2.fillConvexPoly(mask, hull, 255)

    thickness = max(1, int(config.hands.hand_mask_connection_thickness_px))
    radius = max(1, int(config.hands.hand_mask_landmark_radius_px))
    for start, end in HAND_CONNECTIONS:
        if start < len(points_int) and end < len(points_int):
            cv2.line(mask, tuple(points_int[start]), tuple(points_int[end]), 255, thickness)

    for point in points_int:
        cv2.circle(mask, tuple(point), radius, 255, -1)

    dilation = max(0, int(config.hands.hand_mask_dilation_px))
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)

    return mask.astype(bool)


def make_hand_node(node_id, hand_landmarks, handedness, image_shape_hw, depth_m, intrinsics, config=DEFAULT_CONFIG):
    raw_label = None
    confidence = 1.0
    if handedness is not None and getattr(handedness, "classification", None):
        raw_label = handedness.classification[0].label
        confidence = float(handedness.classification[0].score)

    if config.hands.use_mediapipe_handedness and raw_label is not None:
        label = normalize_handedness_label(raw_label, config)
        class_name = f"{label.lower()}_hand"
    else:
        label = None
        class_name = "hand"

    landmarks_2d = hand_landmarks_to_pixels(hand_landmarks, image_shape_hw)
    landmarks_np = np.array(landmarks_2d, dtype=np.float32)
    tracking_mask = hand_landmark_mask(landmarks_2d, image_shape_hw, config)

    center_2d = [float(landmarks_np[:, 0].mean()), float(landmarks_np[:, 1].mean())]
    bbox_2d = [
        float(landmarks_np[:, 0].min()),
        float(landmarks_np[:, 1].min()),
        float(landmarks_np[:, 0].max()),
        float(landmarks_np[:, 1].max()),
    ]

    median_depth_m = geometry.median_depth_near_pixel(
        center_2d,
        depth_m,
        config.hands.depth_sample_radius_px,
        config,
    )
    if median_depth_m is None:
        return None, f"{label} hand has no valid depth near landmark center"

    attributes = {
        "landmarks_2d": landmarks_2d,
        "raw_handedness_label": raw_label,
        "normalized_handedness_label": label,
        "handedness_was_swapped": bool(label and config.hands.swap_mediapipe_handedness),
        "handedness_source": "mediapipe" if label else "tracker",
        "tracking_mask_area": int(tracking_mask.sum()),
    }

    node = make_node(
        node_id=node_id,
        class_name=class_name,
        node_type="hand",
        source_model="mediapipe_hands",
        bbox_2d=bbox_2d,
        center_2d=center_2d,
        center_3d_m=geometry.project_pixel_to_3d(center_2d, median_depth_m, intrinsics),
        median_depth_m=median_depth_m,
        bbox_3d_m=None,
        confidence=confidence,
        attributes=attributes,
    )
    node[TRACKING_MASK_KEY] = tracking_mask
    return node, None


def detect_hand_nodes(
    rgb_image,
    depth_image,
    intrinsics,
    config=DEFAULT_CONFIG,
    hands_processor=None,
    return_diagnostics=False,
):
    if not config.hands.enabled:
        return ([], {"skipped_hands": []}) if return_diagnostics else []

    rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    image_shape_hw = rgb_image.shape[:2]

    if hands_processor is None:
        with create_hand_processor(config) as hands:
            results = hands.process(rgb)
    else:
        results = hands_processor.process(rgb)

    nodes = []
    skipped = []
    if not results.multi_hand_landmarks:
        return (nodes, {"skipped_hands": skipped}) if return_diagnostics else nodes

    counters = {}
    handedness_items = getattr(results, "multi_handedness", None) or []
    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = handedness_items[hand_index] if hand_index < len(handedness_items) else None
        if config.hands.use_mediapipe_handedness and handedness is not None:
            normalized_label = normalize_handedness_label(handedness.classification[0].label, config)
        else:
            normalized_label = "hand"
        node, reason = make_hand_node(
            node_id=hand_node_id(normalized_label, counters),
            hand_landmarks=hand_landmarks,
            handedness=handedness,
            image_shape_hw=image_shape_hw,
            depth_m=depth_image,
            intrinsics=intrinsics,
            config=config,
        )
        if node is None:
            skipped.append((hand_index, reason))
            continue
        nodes.append(node)

    max_hands = max(0, int(config.hands.max_num_hands))
    if len(nodes) > max_hands:
        keep_indices = set(sorted(
            range(len(nodes)),
            key=lambda index: float(nodes[index].get("confidence") or 0.0),
            reverse=True,
        )[:max_hands])
        skipped.extend(
            (index, "extra hand detection suppressed by max_num_hands")
            for index in range(len(nodes))
            if index not in keep_indices
        )
        nodes = [node for index, node in enumerate(nodes) if index in keep_indices]

    if config.hands.use_mediapipe_handedness:
        nodes, disambiguation_diag = disambiguate_duplicate_hand_labels(nodes, config)
        skipped.extend(disambiguation_diag)

    if return_diagnostics:
        return nodes, {"skipped_hands": skipped}
    return nodes
