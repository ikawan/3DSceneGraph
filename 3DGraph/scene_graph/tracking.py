from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

from . import geometry
from .config import DEFAULT_CONFIG
from .schemas import node_id


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


def is_hand_class(class_name):
    return str(class_name or "").lower().endswith("_hand")


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


def bbox_size_from_node(node):
    bbox_3d = node.get("bbox_3d_m")
    if isinstance(bbox_3d, dict) and bbox_3d.get("size_xyz") is not None:
        size = np.asarray(bbox_3d["size_xyz"], dtype=np.float64)
        if size.shape == (3,) and np.all(np.isfinite(size)):
            return size

    bbox_2d = node.get("bbox_2d")
    if bbox_2d:
        x1, y1, x2, y2 = bbox_2d
        size = np.asarray([max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float64)
        if np.all(np.isfinite(size)):
            return size

    return None


def bbox_size_score(track_node, detection_node):
    a = bbox_size_from_node(track_node)
    b = bbox_size_from_node(detection_node)
    if a is None or b is None or a.shape != b.shape:
        return None
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    if scale <= 0:
        return None
    return clamp01(1.0 - float(np.linalg.norm(a - b)) / scale)


def order_score(previous_order, detection_node):
    current_order = trailing_int(node_id(detection_node))
    if previous_order is None or current_order is None:
        return None
    return 1.0 / (1.0 + abs(previous_order - current_order))


@dataclass
class NodeTrack:
    track_id: str
    class_name: str
    node_type: str
    last_node: dict[str, Any]
    certainty: float
    first_frame_id: Any
    last_detected_frame_id: Any
    age_frames: int = 1
    seen_frames: int = 1
    missing_frames: int = 0
    last_match_score: float | None = None
    last_raw_detection_id: str | None = None
    last_raw_detection_order: int | None = None


class TemporalNodeTracker:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.tracks: dict[str, NodeTrack] = {}
        self.class_counters: dict[str, int] = {}

    def reset(self):
        self.tracks.clear()
        self.class_counters.clear()

    def update(self, detected_nodes, frame_id, timestamp=None):
        if not self.config.tracking.enabled:
            return detected_nodes, {"enabled": False}

        detected_nodes = [deepcopy(node) for node in detected_nodes]
        detected_nodes, dropped_extra_hand_detections = self._limit_hand_detections(detected_nodes)
        matches = self._match_detections(detected_nodes)
        matched_track_ids = set()
        matched_detection_indices = set()
        outputs = []

        for track_id, detection_index, match_score in matches:
            track = self.tracks[track_id]
            detection = detected_nodes[detection_index]
            self._update_matched_track(track, detection, frame_id, match_score)
            outputs.append(self._make_output_node(track, detected=True, carried_forward=False))
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

        new_tracks = 0
        skipped_hand_tracks = 0
        for detection_index, detection in enumerate(detected_nodes):
            if detection_index in matched_detection_indices:
                continue
            if is_hand_node(detection) and self._active_hand_track_count() >= self.config.tracking.max_hand_tracks:
                skipped_hand_tracks += 1
                continue
            track = self._create_track(detection, frame_id)
            outputs.append(self._make_output_node(track, detected=True, carried_forward=False))
            matched_track_ids.add(track.track_id)
            matched_detection_indices.add(detection_index)
            new_tracks += 1

        carried_forward = 0
        hidden = 0
        retired = 0
        for track_id in list(self.tracks):
            if track_id in matched_track_ids:
                continue
            track = self.tracks[track_id]
            self._update_missing_track(track)
            if track.missing_frames > self.config.tracking.max_missing_frames or track.certainty <= 0.01:
                del self.tracks[track_id]
                retired += 1
                continue
            if track.certainty >= self.config.tracking.certainty_threshold:
                outputs.append(self._make_output_node(track, detected=False, carried_forward=True))
                carried_forward += 1
            else:
                hidden += 1

        pruned_hand_tracks, kept_hand_track_ids = self._enforce_hand_track_limit(matched_track_ids)
        if pruned_hand_tracks:
            outputs = self._filter_outputs_to_hand_tracks(outputs, kept_hand_track_ids)
            retired += pruned_hand_tracks

        diagnostics = {
            "enabled": True,
            "tracks_total": len(self.tracks),
            "detected_nodes": len(detected_nodes),
            "matched_tracks": len(matches),
            "new_tracks": new_tracks,
            "carried_forward_nodes": carried_forward,
            "hidden_tracks": hidden,
            "retired_tracks": retired,
            "dropped_extra_hand_detections": dropped_extra_hand_detections,
            "skipped_new_hand_tracks": skipped_hand_tracks,
            "max_hand_tracks": self.config.tracking.max_hand_tracks,
            "certainty_threshold": self.config.tracking.certainty_threshold,
            "max_missing_frames": self.config.tracking.max_missing_frames,
        }
        return outputs, diagnostics

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
            if is_hand_track(track) and track.missing_frames <= self.config.tracking.max_missing_frames
        )

    def _enforce_hand_track_limit(self, preferred_track_ids):
        max_hands = max(0, int(self.config.tracking.max_hand_tracks))
        hand_tracks = [track for track in self.tracks.values() if is_hand_track(track)]
        if len(hand_tracks) <= max_hands:
            return 0, {track.track_id for track in hand_tracks}

        def rank(track):
            return (
                1 if track.track_id in preferred_track_ids else 0,
                float(track.certainty),
                -int(track.missing_frames),
                int(track.seen_frames),
                -int(track.age_frames),
            )

        sorted_tracks = sorted(hand_tracks, key=rank, reverse=True)
        keep_ids = {track.track_id for track in sorted_tracks[:max_hands]}
        pruned = 0
        for track in sorted_tracks[max_hands:]:
            if track.track_id in self.tracks:
                del self.tracks[track.track_id]
                pruned += 1
        return pruned, keep_ids

    @staticmethod
    def _filter_outputs_to_hand_tracks(outputs, keep_hand_track_ids):
        filtered = []
        for node in outputs:
            if not is_hand_node(node):
                filtered.append(node)
                continue
            tracking = node.get("tracking") or {}
            if tracking.get("track_id") in keep_hand_track_ids:
                filtered.append(node)
        return filtered

    def _next_track_id(self, node):
        base = safe_name(node.get("class_name") or node_type(node))
        index = self.class_counters.get(base, 0)
        self.class_counters[base] = index + 1
        return f"{base}_track_{index}"

    def _create_track(self, node, frame_id):
        raw_id = node_id(node)
        track_id = self._next_track_id(node)
        track = NodeTrack(
            track_id=track_id,
            class_name=node_class(node),
            node_type=node_type(node),
            last_node=deepcopy(node),
            certainty=node_confidence(node, self.config),
            first_frame_id=frame_id,
            last_detected_frame_id=frame_id,
            last_raw_detection_id=raw_id,
            last_raw_detection_order=trailing_int(raw_id),
        )
        track.last_node = self._node_for_track_storage(track, node)
        self.tracks[track_id] = track
        return track

    def _update_matched_track(self, track, node, frame_id, match_score):
        observation = node_confidence(node, self.config)
        boost = self.config.tracking.detected_update_rate * observation * (1.0 - track.certainty)
        track.certainty = clamp01(track.certainty + boost)
        track.age_frames += 1
        track.seen_frames += 1
        track.missing_frames = 0
        track.last_detected_frame_id = frame_id
        track.last_match_score = float(match_score)
        track.last_raw_detection_id = node_id(node)
        track.last_raw_detection_order = trailing_int(track.last_raw_detection_id)
        track.last_node = self._node_for_track_storage(track, node)

    def _update_missing_track(self, track):
        track.age_frames += 1
        track.missing_frames += 1
        track.certainty = clamp01(track.certainty * self.config.tracking.missing_decay)
        track.last_match_score = None

    def _make_output_node(self, track, detected, carried_forward):
        output = deepcopy(track.last_node)
        output["id"] = track.track_id
        output.pop("tracking", None)

        tracking = {
            "track_id": track.track_id,
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
        }
        output["tracking"] = tracking
        return output

    @staticmethod
    def _node_for_track_storage(track, node):
        output = deepcopy(node)
        if is_hand_track(track):
            observed_class_name = output.get("class_name")
            attributes = output.setdefault("attributes", {})
            attributes["observed_class_name"] = observed_class_name
            attributes["stable_class_name"] = track.class_name
            output["class_name"] = track.class_name
        return output

    def _match_detections(self, detected_nodes):
        candidates = []
        for track_id, track in self.tracks.items():
            if track.missing_frames > self.config.tracking.max_missing_frames:
                continue
            for detection_index, detection in enumerate(detected_nodes):
                score = self._match_score(track, detection)
                if score is None or score < self._minimum_match_score(track, detection):
                    continue
                candidates.append((score, track.certainty, track_id, detection_index))

        candidates.sort(reverse=True)
        used_tracks = set()
        used_detections = set()
        matches = []
        for score, _, track_id, detection_index in candidates:
            if track_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            matches.append((track_id, detection_index, score))
        return matches

    def _minimum_match_score(self, track, detection):
        if is_hand_track(track) and is_hand_node(detection):
            return self.config.tracking.hand_min_match_score
        return self.config.tracking.min_match_score

    def _match_score(self, track, detection):
        if track.node_type != node_type(detection):
            return None

        hand_match = is_hand_track(track) and is_hand_node(detection)
        if not hand_match and track.class_name != node_class(detection):
            return None

        last_node = track.last_node
        signals = []

        dist_3d = geometry.distance_3d(last_node.get("center_3d_m"), detection.get("center_3d_m"))
        max_3d_distance = (
            self.config.tracking.hand_max_3d_match_distance_m
            if hand_match
            else self.config.tracking.max_3d_match_distance_m
        )
        self._add_signal(signals, 0.30, distance_score(dist_3d, max_3d_distance))

        dist_2d = center_2d_distance(last_node, detection)
        max_2d_distance = (
            self.config.tracking.hand_max_2d_center_distance_px
            if hand_match
            else self.config.tracking.max_2d_center_distance_px
        )
        self._add_signal(signals, 0.25, distance_score(dist_2d, max_2d_distance))

        self._add_signal(signals, 0.20, geometry.bbox_2d_iou(last_node.get("bbox_2d"), detection.get("bbox_2d")))
        self._add_signal(signals, 0.15, depth_score(last_node, detection, self.config))
        self._add_signal(signals, 0.07, bbox_size_score(last_node, detection))
        if hand_match:
            label_score = 1.0 if track.class_name == node_class(detection) else 0.65
            self._add_signal(signals, 0.03, label_score)
        else:
            self._add_signal(signals, 0.03, order_score(track.last_raw_detection_order, detection))

        if not signals:
            return None
        weight_sum = sum(weight for weight, _ in signals)
        return sum(weight * score for weight, score in signals) / weight_sum

    @staticmethod
    def _add_signal(signals, weight, score):
        if score is None:
            return
        signals.append((float(weight), clamp01(score)))
