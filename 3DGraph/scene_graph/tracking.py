from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import os
import re
import sys
from typing import Any

import cv2
import numpy as np

from . import geometry
from .config import DEFAULT_CONFIG
from .io import resolve_path
from .schemas import make_node, node_id


TRACKING_MASK_KEY = "_tracking_mask"


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def safe_name(value):
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_")
    return cleaned or "node"


def trailing_int(value):
    match = re.search(r"(\d+)$", str(value or ""))
    if match is None:
        return None
    return int(match.group(1))


def node_type(node):
    return node.get("node_type") or "object"


def node_class(node):
    return str(node.get("class_name", "")).lower()


def normalize_hand_class_name(value):
    value = str(value or "hand").lower()
    if value in {"left", "right"}:
        return f"{value}_hand"
    return value if value.endswith("_hand") else f"{value}_hand"


def is_hand_class(class_name):
    class_name = str(class_name or "").lower()
    return class_name == "hand" or class_name.endswith("_hand")


def is_hand_node(node):
    return node_type(node) == "hand" or is_hand_class(node.get("class_name"))


def is_hand_track(track):
    return track.node_type == "hand" or is_hand_class(track.class_name)


def node_confidence(node, config):
    try:
        confidence = float(node.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    return max(config.tracking.min_observation_confidence, clamp01(confidence))


def center_2d_distance(a, b):
    a_center = a.get("center_2d")
    b_center = b.get("center_2d")
    if not a_center or not b_center:
        return None
    a_arr = np.asarray(a_center, dtype=np.float64)
    b_arr = np.asarray(b_center, dtype=np.float64)
    if a_arr.shape != (2,) or b_arr.shape != (2,):
        return None
    if not np.all(np.isfinite(a_arr)) or not np.all(np.isfinite(b_arr)):
        return None
    return float(np.linalg.norm(a_arr - b_arr))


def distance_score(distance, max_distance):
    if distance is None or max_distance <= 0:
        return None
    return clamp01(1.0 - float(distance) / float(max_distance))


def depth_score(track_node, detection_node, config):
    a = track_node.get("median_depth_m")
    b = detection_node.get("median_depth_m")
    if a is None or b is None or config.tracking.depth_similarity_m <= 0:
        return None
    try:
        delta = abs(float(a) - float(b))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(delta):
        return None
    return clamp01(1.0 - delta / config.tracking.depth_similarity_m)


def mask_iou(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        return None
    union = np.logical_or(a, b).sum()
    if union <= 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def strip_tracking_payload(node):
    output = deepcopy(node)
    output.pop(TRACKING_MASK_KEY, None)
    return output


def strip_tracking_payloads(nodes):
    return [strip_tracking_payload(node) for node in nodes]


def _mask_for_node(node, shape_hw=None):
    mask = node.get(TRACKING_MASK_KEY)
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if shape_hw is not None and mask.shape != tuple(shape_hw):
        mask = geometry.resize_mask(mask, shape_hw)
    return mask


@dataclass
class XMemPrediction:
    index_mask: np.ndarray
    probabilities: np.ndarray | None


class XMemVideoObjectSegmenter:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self._load_runtime()
        self.processor = None
        self.labels: list[int] = []

    def _load_runtime(self):
        if str(self.config.tracking.backend).lower() != "xmem":
            raise RuntimeError(f"Unsupported tracking backend: {self.config.tracking.backend}")

        repo_path = resolve_path(self.config.tracking.xmem_repo_path)
        checkpoint_path = resolve_path(self.config.tracking.xmem_checkpoint_path)
        if not repo_path.exists():
            raise RuntimeError(
                "XMem tracking is enabled, but the XMem repository was not found at "
                f"{repo_path}. Clone https://github.com/hkchengrex/XMem there or set "
                "config.tracking.xmem_repo_path / --xmem-repo."
            )
        if not checkpoint_path.exists():
            raise RuntimeError(
                "XMem tracking is enabled, but the checkpoint was not found at "
                f"{checkpoint_path}. Download XMem.pth from the official XMem v1.0 release "
                "or set config.tracking.xmem_checkpoint_path / --xmem-checkpoint."
            )

        repo_path_text = str(repo_path)
        if repo_path_text not in sys.path:
            sys.path.insert(0, repo_path_text)

        try:
            import torch
            import torch.nn.functional as F
            from inference.inference_core import InferenceCore
            from model.network import XMem
        except Exception as exc:
            raise RuntimeError(
                "Could not import XMem. Install the official XMem repository dependencies "
                "inside the active environment, including PyTorch and torchvision."
            ) from exc

        self.torch = torch
        self.F = F
        self.InferenceCore = InferenceCore
        self.device = torch.device(self.config.runtime.device)
        torch_cache_dir = resolve_path("3DGraph/outputs/torch_cache")
        torch_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(torch_cache_dir))
        torch.hub.set_dir(str(torch_cache_dir / "hub"))
        self.xmem_config = {
            "top_k": self.config.tracking.xmem_top_k,
            "mem_every": self.config.tracking.xmem_mem_every,
            "deep_update_every": self.config.tracking.xmem_deep_update_every,
            "enable_long_term": self.config.tracking.xmem_enable_long_term,
            "enable_long_term_count_usage": self.config.tracking.xmem_enable_long_term_count_usage,
            "num_prototypes": self.config.tracking.xmem_num_prototypes,
            "min_mid_term_frames": self.config.tracking.xmem_min_mid_term_frames,
            "max_mid_term_frames": self.config.tracking.xmem_max_mid_term_frames,
            "max_long_term_elements": self.config.tracking.xmem_max_long_term_elements,
        }
        self.network = XMem(self.xmem_config, str(checkpoint_path), map_location=self.device).eval()
        self.network = self.network.to(self.device)

    def reset(self):
        self.processor = None
        self.labels = []
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def _ensure_processor(self, labels):
        if self.processor is None:
            self.processor = self.InferenceCore(self.network, config=self.xmem_config)
        self.labels = list(labels)
        self.processor.set_all_labels(self.labels)

    def step(self, bgr_image, label_mask=None, valid_labels=None, labels=None):
        labels = list(labels or [])
        if not labels:
            h, w = bgr_image.shape[:2]
            return XMemPrediction(np.zeros((h, w), dtype=np.int32), None)

        if labels != list(range(1, max(labels) + 1)):
            raise ValueError("XMem labels must be contiguous and start at 1.")

        self._ensure_processor(labels)
        frame, original_shape_hw, resized_shape_hw = self._preprocess_frame(bgr_image)
        one_hot_mask = None
        valid_labels = sorted(set(int(label) for label in (valid_labels or [])))
        if label_mask is not None and valid_labels:
            one_hot_mask = self._preprocess_label_mask(label_mask, labels, resized_shape_hw)

        autocast_enabled = bool(self.config.tracking.xmem_amp and self.device.type == "cuda")
        with self.torch.no_grad():
            with self.torch.cuda.amp.autocast(enabled=autocast_enabled):
                prob = self.processor.step(
                    frame,
                    one_hot_mask,
                    valid_labels=valid_labels if one_hot_mask is not None else None,
                )

        prob = self.F.interpolate(
            prob.unsqueeze(1),
            original_shape_hw,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        index_mask = self.torch.argmax(prob, dim=0).detach().cpu().numpy().astype(np.int32)
        probabilities = prob.detach().cpu().numpy().astype(np.float32)
        return XMemPrediction(index_mask=index_mask, probabilities=probabilities)

    def _preprocess_frame(self, bgr_image):
        original_h, original_w = bgr_image.shape[:2]
        resolution = int(self.config.tracking.xmem_size)
        if resolution > 0:
            scale = min(original_w, original_h) / float(resolution)
            resized_w = max(1, int(round(original_w / scale)))
            resized_h = max(1, int(round(original_h / scale)))
        else:
            resized_h, resized_w = original_h, original_w

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        frame = self.torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float()
        frame = self.F.interpolate(frame, (resized_h, resized_w), mode="bilinear", align_corners=False)
        frame = frame.squeeze(0).to(self.device) / 255.0
        mean = self.torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = self.torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        frame = (frame - mean) / std
        return frame, (original_h, original_w), (resized_h, resized_w)

    def _preprocess_label_mask(self, label_mask, labels, resized_shape_hw):
        label_mask = np.asarray(label_mask, dtype=np.int32)
        mask_tensor = self.torch.from_numpy(label_mask).float().view(1, 1, *label_mask.shape)
        mask_tensor = self.F.interpolate(mask_tensor, resized_shape_hw, mode="nearest")
        mask_tensor = mask_tensor.to(self.device).long()[0, 0]

        one_hot = self.torch.zeros(
            (len(labels), resized_shape_hw[0], resized_shape_hw[1]),
            dtype=self.torch.float32,
            device=self.device,
        )
        for label in labels:
            one_hot[label - 1] = (mask_tensor == int(label)).float()
        return one_hot


@dataclass
class NodeTrack:
    label_id: int
    track_id: str
    class_name: str
    node_type: str
    last_node: dict[str, Any]
    certainty: float
    first_frame_id: Any
    last_detected_frame_id: Any
    source_model: str | None = None
    last_mask: np.ndarray | None = None
    age_frames: int = 0
    seen_frames: int = 0
    missing_frames: int = 0
    retired: bool = False
    last_match_score: float | None = None
    last_raw_detection_id: str | None = None
    last_raw_detection_order: int | None = None
    last_xmem_confidence: float | None = None


class TemporalNodeTracker:
    def __init__(self, config=DEFAULT_CONFIG, xmem_engine=None):
        self.config = config
        self.tracks: dict[int, NodeTrack] = {}
        self.class_counters: dict[str, int] = {}
        self.next_label_id = 1
        self.xmem_engine = xmem_engine

    def reset(self):
        self.tracks.clear()
        self.class_counters.clear()
        self.next_label_id = 1
        if self.xmem_engine is not None:
            self.xmem_engine.reset()

    def update(
        self,
        detected_nodes,
        frame_id,
        timestamp=None,
        rgb_image=None,
        depth_image=None,
        intrinsics=None,
    ):
        if not self.config.tracking.enabled:
            return strip_tracking_payloads(detected_nodes), {"enabled": False}
        if rgb_image is None or depth_image is None or intrinsics is None:
            raise RuntimeError("XMem tracking requires rgb_image, depth_image, and intrinsics for every frame.")

        image_shape_hw = rgb_image.shape[:2]
        detected_nodes = [deepcopy(node) for node in detected_nodes]
        detected_nodes, dropped_extra_hand_detections = self._limit_hand_detections(detected_nodes)
        matches = self._match_existing_tracks(detected_nodes, image_shape_hw)
        assignments = {detection_index: (track, score) for track, detection_index, score in matches}

        new_tracks = self._create_tracks_for_unmatched_detections(detected_nodes, assignments, frame_id, image_shape_hw)
        for detection_index, track in new_tracks:
            assignments[detection_index] = (track, 1.0)

        correction_mask, valid_labels = self._build_correction_mask(detected_nodes, assignments, image_shape_hw)
        prediction = self._step_xmem(rgb_image, correction_mask, valid_labels)
        detection_by_label = {
            track.label_id: (detected_nodes[detection_index], score)
            for detection_index, (track, score) in assignments.items()
        }

        outputs = []
        carried_forward = 0
        hidden = 0
        retired = 0
        matched_tracks = 0
        for label_id in self._all_labels():
            track = self.tracks.get(label_id)
            if track is None or track.retired:
                continue

            detection, match_score = detection_by_label.get(label_id, (None, None))
            if detection is not None:
                matched_tracks += 1

            label_mask = prediction.index_mask == label_id
            if int(label_mask.sum()) < self.config.tracking.xmem_min_mask_area_pixels:
                detection_mask = _mask_for_node(detection, image_shape_hw) if detection is not None else None
                if detection_mask is not None and int(detection_mask.sum()) >= self.config.tracking.xmem_min_mask_area_pixels:
                    label_mask = detection_mask

            xmem_confidence = self._label_confidence(prediction, label_id, label_mask)
            self._update_track_from_frame(
                track=track,
                detection=detection,
                mask=label_mask,
                depth_image=depth_image,
                intrinsics=intrinsics,
                frame_id=frame_id,
                match_score=match_score,
                xmem_confidence=xmem_confidence,
            )

            if track.missing_frames > self.config.tracking.max_missing_frames or track.certainty <= 0.01:
                track.retired = True
                retired += 1
                continue

            detected = detection is not None
            carried = not detected
            if detected or track.certainty >= self.config.tracking.certainty_threshold:
                outputs.append(self._make_output_node(track, detected=detected, carried_forward=carried))
                if carried:
                    carried_forward += 1
            else:
                hidden += 1

        diagnostics = {
            "enabled": True,
            "backend": "xmem",
            "tracks_total": sum(1 for track in self.tracks.values() if not track.retired),
            "xmem_labels_total": self.next_label_id - 1,
            "detected_nodes": len(detected_nodes),
            "matched_tracks": matched_tracks,
            "new_tracks": len(new_tracks),
            "carried_forward_nodes": carried_forward,
            "hidden_tracks": hidden,
            "retired_tracks": retired,
            "dropped_extra_hand_detections": dropped_extra_hand_detections,
            "max_hand_tracks": self.config.tracking.max_hand_tracks,
            "certainty_threshold": self.config.tracking.certainty_threshold,
            "max_missing_frames": self.config.tracking.max_missing_frames,
            "xmem_repo_path": str(resolve_path(self.config.tracking.xmem_repo_path)),
            "xmem_checkpoint_path": str(resolve_path(self.config.tracking.xmem_checkpoint_path)),
        }
        return outputs, diagnostics

    def _step_xmem(self, rgb_image, correction_mask, valid_labels):
        if self.next_label_id <= 1:
            h, w = rgb_image.shape[:2]
            return XMemPrediction(np.zeros((h, w), dtype=np.int32), None)
        if self.xmem_engine is None:
            self.xmem_engine = XMemVideoObjectSegmenter(self.config)
        return self.xmem_engine.step(
            rgb_image,
            label_mask=correction_mask,
            valid_labels=valid_labels,
            labels=self._all_labels(),
        )

    def _all_labels(self):
        return list(range(1, self.next_label_id))

    def _limit_hand_detections(self, detected_nodes):
        max_hands = max(0, int(self.config.tracking.max_hand_tracks))
        if max_hands <= 0:
            kept = [node for node in detected_nodes if not is_hand_node(node)]
            return kept, len(detected_nodes) - len(kept)

        hand_indices = [index for index, node in enumerate(detected_nodes) if is_hand_node(node)]
        if len(hand_indices) <= max_hands:
            return detected_nodes, 0

        hand_indices_by_confidence = sorted(
            hand_indices,
            key=lambda index: node_confidence(detected_nodes[index], self.config),
            reverse=True,
        )
        keep_hand_indices = set(hand_indices_by_confidence[:max_hands])
        kept_nodes = [
            node
            for index, node in enumerate(detected_nodes)
            if not is_hand_node(node) or index in keep_hand_indices
        ]
        return kept_nodes, len(hand_indices) - len(keep_hand_indices)

    def _active_hand_track_count(self):
        return sum(
            1
            for track in self.tracks.values()
            if not track.retired
            and is_hand_track(track)
            and track.missing_frames <= self.config.tracking.max_missing_frames
        )

    def _next_track_id(self, class_name):
        base = safe_name(class_name)
        index = self.class_counters.get(base, 0)
        self.class_counters[base] = index + 1
        return f"{base}_track_{index}"

    def _create_track(self, node, frame_id, class_name=None):
        class_name = class_name or node_class(node)
        label_id = self.next_label_id
        self.next_label_id += 1

        raw_id = node_id(node)
        track = NodeTrack(
            label_id=label_id,
            track_id=self._next_track_id(class_name),
            class_name=str(class_name).lower(),
            node_type=node_type(node),
            source_model=node.get("source_model"),
            last_node=strip_tracking_payload(node),
            last_mask=_mask_for_node(node),
            certainty=node_confidence(node, self.config),
            first_frame_id=frame_id,
            last_detected_frame_id=frame_id,
            last_raw_detection_id=raw_id,
            last_raw_detection_order=trailing_int(raw_id),
        )
        if is_hand_track(track):
            track.node_type = "hand"
        self.tracks[label_id] = track
        return track

    def _match_existing_tracks(self, detected_nodes, image_shape_hw):
        candidates = []
        for track in self.tracks.values():
            if track.retired or track.missing_frames > self.config.tracking.max_missing_frames:
                continue
            for detection_index, detection in enumerate(detected_nodes):
                score = self._match_score(track, detection, image_shape_hw)
                if score is None or score < self._minimum_match_score(track, detection):
                    continue
                candidates.append((score, track.certainty, track.label_id, detection_index))

        candidates.sort(reverse=True)
        used_tracks = set()
        used_detections = set()
        matches = []
        for score, _, label_id, detection_index in candidates:
            if label_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(label_id)
            used_detections.add(detection_index)
            matches.append((self.tracks[label_id], detection_index, score))
        return matches

    def _minimum_match_score(self, track, detection):
        if is_hand_track(track) and is_hand_node(detection):
            return self.config.tracking.hand_min_match_score
        return self.config.tracking.min_match_score

    def _match_score(self, track, detection, image_shape_hw):
        hand_match = is_hand_track(track) and is_hand_node(detection)
        if not hand_match:
            if track.node_type != node_type(detection) or track.class_name != node_class(detection):
                return None

        signals = []
        self._add_signal(signals, 0.40, mask_iou(track.last_mask, _mask_for_node(detection, image_shape_hw)))
        self._add_signal(signals, 0.20, geometry.bbox_2d_iou(track.last_node.get("bbox_2d"), detection.get("bbox_2d")))

        max_2d_distance = (
            self.config.tracking.hand_max_2d_center_distance_px
            if hand_match
            else self.config.tracking.max_2d_center_distance_px
        )
        self._add_signal(
            signals,
            0.17,
            distance_score(center_2d_distance(track.last_node, detection), max_2d_distance),
        )

        max_3d_distance = (
            self.config.tracking.hand_max_3d_match_distance_m
            if hand_match
            else self.config.tracking.max_3d_match_distance_m
        )
        self._add_signal(
            signals,
            0.15,
            distance_score(
                geometry.distance_3d(track.last_node.get("center_3d_m"), detection.get("center_3d_m")),
                max_3d_distance,
            ),
        )
        self._add_signal(signals, 0.08, depth_score(track.last_node, detection, self.config))

        if not signals:
            return None
        weight_sum = sum(weight for weight, _ in signals)
        return sum(weight * score for weight, score in signals) / weight_sum

    @staticmethod
    def _add_signal(signals, weight, score):
        if score is None:
            return
        signals.append((float(weight), clamp01(score)))

    def _create_tracks_for_unmatched_detections(self, detected_nodes, assignments, frame_id, image_shape_hw):
        matched_indices = set(assignments)
        new_tracks = []

        for detection_index, detection in enumerate(detected_nodes):
            if detection_index in matched_indices or is_hand_node(detection):
                continue
            new_tracks.append((detection_index, self._create_track(detection, frame_id)))

        hand_indices = [
            index
            for index, node in enumerate(detected_nodes)
            if index not in matched_indices and is_hand_node(node)
        ]
        if not hand_indices:
            return new_tracks

        hand_indices = sorted(hand_indices, key=lambda index: self._hand_reference_x(detected_nodes[index]))
        available_slots = max(0, int(self.config.tracking.max_hand_tracks) - self._active_hand_track_count())
        for detection_index in hand_indices[:available_slots]:
            class_name = self._class_for_new_hand(detected_nodes[detection_index], image_shape_hw)
            if class_name is None:
                continue
            new_tracks.append((detection_index, self._create_track(detected_nodes[detection_index], frame_id, class_name)))
        return new_tracks

    def _class_for_new_hand(self, node, image_shape_hw):
        observed = node_class(node)
        if observed in {"left_hand", "right_hand"} and not self._has_active_hand_class(observed):
            return observed

        image_left_label = normalize_hand_class_name(self.config.hands.image_left_hand_label)
        image_right_label = normalize_hand_class_name(self.config.hands.image_right_hand_label)
        preferred = image_left_label
        center = node.get("center_2d") or [self._hand_reference_x(node), 0.0]
        if center and float(center[0]) >= image_shape_hw[1] / 2.0:
            preferred = image_right_label

        for candidate in (preferred, image_left_label, image_right_label):
            if not self._has_active_hand_class(candidate):
                return candidate
        return None

    def _has_active_hand_class(self, class_name):
        class_name = normalize_hand_class_name(class_name)
        return any(
            not track.retired
            and is_hand_track(track)
            and track.class_name == class_name
            and track.missing_frames <= self.config.tracking.max_missing_frames
            for track in self.tracks.values()
        )

    @staticmethod
    def _hand_reference_x(node):
        landmarks = node.get("attributes", {}).get("landmarks_2d") or node.get("landmarks_2d")
        if landmarks and len(landmarks) > 0 and len(landmarks[0]) >= 2:
            return float(landmarks[0][0])
        center = node.get("center_2d") or [0.0, 0.0]
        return float(center[0])

    def _build_correction_mask(self, detected_nodes, assignments, image_shape_hw):
        if not assignments:
            return None, []

        correction_mask = np.zeros(image_shape_hw, dtype=np.int32)
        items = []
        for detection_index, (track, _) in assignments.items():
            mask = _mask_for_node(detected_nodes[detection_index], image_shape_hw)
            if mask is None:
                continue
            area = int(mask.sum())
            if area <= 0:
                continue
            items.append((area, track.label_id, mask))

        if not items:
            return None, []

        valid_labels = []
        for _, label_id, mask in sorted(items, key=lambda item: item[0], reverse=True):
            correction_mask[mask] = int(label_id)
            valid_labels.append(int(label_id))
        return correction_mask, sorted(set(valid_labels))

    def _label_confidence(self, prediction, label_id, label_mask):
        if prediction.probabilities is None:
            return None
        if label_id >= prediction.probabilities.shape[0] or not np.any(label_mask):
            return 0.0
        values = prediction.probabilities[label_id][label_mask]
        if values.size == 0:
            return 0.0
        return float(np.mean(values))

    def _update_track_from_frame(
        self,
        track,
        detection,
        mask,
        depth_image,
        intrinsics,
        frame_id,
        match_score,
        xmem_confidence,
    ):
        detected = detection is not None
        track.age_frames += 1
        track.last_xmem_confidence = xmem_confidence

        valid_mask = mask is not None and int(np.asarray(mask, dtype=bool).sum()) >= self.config.tracking.xmem_min_mask_area_pixels
        if detected:
            observation = node_confidence(detection, self.config)
            boost = self.config.tracking.detected_update_rate * observation * (1.0 - track.certainty)
            track.certainty = clamp01(track.certainty + boost)
            track.seen_frames += 1
            track.missing_frames = 0
            track.last_detected_frame_id = frame_id
            track.last_match_score = None if match_score is None else float(match_score)
            track.last_raw_detection_id = node_id(detection)
            track.last_raw_detection_order = trailing_int(track.last_raw_detection_id)
        else:
            track.certainty = clamp01(track.certainty * self.config.tracking.missing_decay)
            track.missing_frames += 1
            track.last_match_score = None

        if valid_mask:
            track.last_mask = np.asarray(mask, dtype=bool)
            track.last_node = self._node_from_mask(
                track=track,
                mask=track.last_mask,
                depth_image=depth_image,
                intrinsics=intrinsics,
                detection=detection,
                detected=detected,
                xmem_confidence=xmem_confidence,
            )
        elif detected:
            track.last_node = self._node_from_detection(track, detection, xmem_confidence)
        else:
            attrs = track.last_node.setdefault("attributes", {})
            attrs["xmem_mask_area"] = 0
            attrs["xmem_mask_confidence"] = xmem_confidence

    def _node_from_detection(self, track, detection, xmem_confidence):
        output = strip_tracking_payload(detection)
        output["class_name"] = track.class_name
        output["node_type"] = track.node_type
        attrs = output.setdefault("attributes", {})
        attrs["stable_class_name"] = track.class_name
        attrs["observed_class_name"] = detection.get("class_name")
        attrs["xmem_label_id"] = int(track.label_id)
        attrs["xmem_mask_confidence"] = xmem_confidence
        if is_hand_track(track):
            output["class_name"] = track.class_name
            attrs["hand_label_source"] = "xmem_stable_track"
        return output

    def _node_from_mask(self, track, mask, depth_image, intrinsics, detection, detected, xmem_confidence):
        fallback = detection if detection is not None else track.last_node
        bbox_2d = geometry.bbox_from_mask(mask, fallback.get("bbox_2d"))
        center_2d = geometry.mask_center(mask, bbox_2d)
        median_depth_m = geometry.median_depth_in_mask(mask, depth_image, self.config)

        center_3d_m = fallback.get("center_3d_m")
        bbox_3d_m = fallback.get("bbox_3d_m")
        if median_depth_m is not None:
            center_3d_m = geometry.project_pixel_to_3d(center_2d, median_depth_m, intrinsics)
            if track.node_type != "hand":
                points_3d = geometry.mask_to_3d_points(mask, depth_image, intrinsics, self.config)
                if points_3d is not None:
                    bbox_3d_m = geometry.summarize_3d_points(points_3d, self.config)["bbox_3d_m"]

        attributes = deepcopy(fallback.get("attributes") or {})
        attributes.update(
            {
                "stable_class_name": track.class_name,
                "observed_class_name": fallback.get("class_name"),
                "xmem_label_id": int(track.label_id),
                "xmem_mask_area": int(mask.sum()),
                "xmem_mask_confidence": xmem_confidence,
                "xmem_corrected_by_detector": bool(detected),
            }
        )
        if is_hand_track(track):
            attributes["hand_label_source"] = "xmem_stable_track"
            bbox_3d_m = None

        source_model = fallback.get("source_model")
        if not detected:
            source_model = "xmem_tracking"

        return make_node(
            node_id=track.track_id,
            class_name=track.class_name,
            node_type=track.node_type,
            source_model=source_model,
            bbox_2d=bbox_2d,
            center_2d=center_2d,
            center_3d_m=center_3d_m,
            median_depth_m=median_depth_m if median_depth_m is not None else fallback.get("median_depth_m"),
            bbox_3d_m=bbox_3d_m,
            confidence=fallback.get("confidence"),
            attributes=attributes,
        )

    def _make_output_node(self, track, detected, carried_forward):
        output = strip_tracking_payload(track.last_node)
        output["id"] = track.track_id
        output["class_name"] = track.class_name
        output["node_type"] = track.node_type
        output.pop("tracking", None)

        output["tracking"] = {
            "track_id": track.track_id,
            "xmem_label_id": int(track.label_id),
            "certainty": round(float(track.certainty), 4),
            "detected_this_frame": bool(detected),
            "carried_forward": bool(carried_forward),
            "age_frames": int(track.age_frames),
            "seen_frames": int(track.seen_frames),
            "missing_frames": int(track.missing_frames),
            "first_frame_id": track.first_frame_id,
            "last_detected_frame_id": track.last_detected_frame_id,
            "match_score": None if track.last_match_score is None else round(float(track.last_match_score), 4),
            "last_raw_detection_id": track.last_raw_detection_id,
            "xmem_mask_confidence": (
                None if track.last_xmem_confidence is None else round(float(track.last_xmem_confidence), 4)
            ),
        }
        return output
