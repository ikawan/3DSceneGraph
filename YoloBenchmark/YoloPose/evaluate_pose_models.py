"""
YOLO Pose Model Evaluation Script

This script compares YOLO pose model outputs that were previously generated and saved to disk.

KEY CONCEPTS:
  - Reference Model: The heaviest/slowest model, treated as pseudo-ground-truth (NOT true labeled ground truth).
  - Evaluation measures: Detection agreement, bounding-box IoU, keypoint accuracy, temporal stability, and inference speed.
  - Temporal stability is measured by frame-to-frame consistency within each model independently.
  - Speed targets: Models should achieve ~33.33 ms/frame mean (30 Hz) to support real-time processing.

OUTPUT FILES GENERATED:
  - pose_evaluation_summary.csv: Per-model summary metrics and scores
  - pose_evaluation_summary.json: Detailed JSON report with all metrics
  - pose_per_keypoint_metrics.csv: Per-keypoint comparison metrics
  - pose_per_frame_metrics.csv: Optional per-frame debugging metrics

AUTHOR: Generated for YOLO benchmark evaluation
"""

import json
import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import warnings

import numpy as np

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================

# Path to the benchmark results directory
BENCHMARK_BASE_DIR = Path(__file__).parent / "benchmarks"

# Name of the reference model (the pseudo-ground-truth)
# This should be the slowest/heaviest model, e.g., "yolo26x-pose"
REFERENCE_MODEL_NAME = "yolo26x-pose"

# IoU threshold for matching bounding boxes between candidate and reference
MATCH_IOU_THRESHOLD = 0.5

# IoU threshold for matching instances between consecutive frames (temporal stability)
TEMPORAL_MATCH_IOU_THRESHOLD = 0.5

# Keypoint confidence threshold: ignore keypoints with confidence below this
KEYPOINT_CONF_THRESHOLD = 0.1

# Method for normalizing keypoint distances:
#   "bbox_diagonal": distance / bbox_diagonal_length
#   "bbox_sqrt_area": distance / sqrt(bbox_area)
KEYPOINT_DISTANCE_NORMALIZATION = "bbox_diagonal"

# Weighting for final ranking score (should sum to 1.0)
SCORE_WEIGHTS = {
    "f1": 0.30,
    "keypoint_accuracy": 0.25,
    "bbox_iou": 0.15,
    "speed": 0.15,
    "stability": 0.15,
}

# Output directory for results
OUTPUT_DIR = Path(__file__).parent

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class KeypointData:
    """Represents a single keypoint."""
    kp_id: int
    x: float
    y: float
    conf: Optional[float] = None

    def position(self) -> Tuple[float, float]:
        """Return (x, y) position."""
        return (self.x, self.y)


@dataclass
class InstanceData:
    """Represents a pose detection instance in a frame."""
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None  # [x1, y1, x2, y2]
    bbox_centroid_xy: Optional[List[float]] = None  # [cx, cy]
    num_keypoints: Optional[int] = None
    num_visible_keypoints: Optional[int] = None
    mean_keypoint_confidence: Optional[float] = None
    keypoints: List[KeypointData] = field(default_factory=list)

    def bbox_area(self) -> float:
        """Compute bounding box area."""
        if self.bbox_xyxy is None:
            return 0.0
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, (x2 - x1) * (y2 - y1))

    def get_keypoint_map(self) -> Dict[int, KeypointData]:
        """Return keypoints indexed by ID."""
        return {kp.kp_id: kp for kp in self.keypoints}


@dataclass
class FramePredictions:
    """Represents all detections in a single frame."""
    frame_idx: int
    timestamp_sec: float
    inference_ms: Optional[float] = None
    instances: List[InstanceData] = field(default_factory=list)


@dataclass
class ModelMetadata:
    """Represents metadata for a model."""
    model_name: str
    model_path: str
    mean_inference_ms: float
    median_inference_ms: float
    p95_inference_ms: float
    effective_fps: float
    num_frames_processed: int
    target_fps: float = 30.0
    target_ms_per_frame: float = 33.3333


@dataclass
class DetectionMetrics:
    """Detection-level agreement metrics."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class BboxMetrics:
    """Bounding box agreement metrics."""
    mean_iou: float = 0.0
    median_iou: float = 0.0
    p95_iou: float = 0.0
    all_ious: List[float] = field(default_factory=list)


@dataclass
class KeypointMetrics:
    """Keypoint agreement metrics."""
    mean_distance_px: float = 0.0
    median_distance_px: float = 0.0
    mean_normalized_distance: float = 0.0
    comparable_keypoint_fraction: float = 0.0
    avg_detected_keypoints_per_match: float = 0.0
    avg_visible_keypoints_per_match: float = 0.0
    mean_keypoint_confidence: float = 0.0
    all_distances_px: List[float] = field(default_factory=list)
    all_normalized_distances: List[float] = field(default_factory=list)


@dataclass
class TemporalMetrics:
    """Temporal stability metrics."""
    mean_abs_count_delta: float = 0.0
    mean_bbox_centroid_jitter: float = 0.0
    mean_bbox_area_delta: float = 0.0
    mean_keypoint_jitter: float = 0.0
    empty_frame_fraction: float = 0.0


@dataclass
class TimingMetrics:
    """Inference timing metrics."""
    mean_inference_ms: float = 0.0
    median_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    effective_fps: float = 0.0
    meets_30hz_mean: bool = False
    meets_30hz_p95: bool = False


@dataclass
class ModelEvaluation:
    """Complete evaluation results for one model."""
    model_name: str
    reference_model_name: str
    num_frames_compared: int
    detection_metrics: DetectionMetrics
    bbox_metrics: BboxMetrics
    keypoint_metrics: KeypointMetrics
    temporal_metrics: TemporalMetrics
    timing_metrics: TimingMetrics
    final_score: float = 0.0


@dataclass
class PerKeypointMetrics:
    """Metrics for a specific keypoint ID across frames."""
    model_name: str
    keypoint_id: int
    mean_distance_px: float = 0.0
    median_distance_px: float = 0.0
    mean_normalized_distance: float = 0.0
    num_compared: int = 0


# ============================================================================
# HELPER FUNCTIONS: BBOX OPERATIONS
# ============================================================================

def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute Intersection-over-Union (IoU) between two bounding boxes.
    
    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]
    
    Returns:
        IoU in range [0, 1]
    """
    if box_a is None or box_b is None:
        return 0.0
    
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    
    # Ensure coordinates are valid (x1 <= x2, y1 <= y2)
    x1_a, x2_a = min(x1_a, x2_a), max(x1_a, x2_a)
    y1_a, y2_a = min(y1_a, y2_a), max(y1_a, y2_a)
    x1_b, x2_b = min(x1_b, x2_b), max(x1_b, x2_b)
    y1_b, y2_b = min(y1_b, y2_b), max(y1_b, y2_b)
    
    # Compute intersection area
    x1_inter = max(x1_a, x1_b)
    y1_inter = max(y1_a, y1_b)
    x2_inter = min(x2_a, x2_b)
    y2_inter = min(y2_a, y2_b)
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Compute union area
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area
    
    if union_area == 0.0:
        return 0.0
    
    return inter_area / union_area


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalize_keypoint_distance(
    distance_px: float, ref_bbox: List[float], method: str = "bbox_diagonal"
) -> float:
    """
    Normalize keypoint distance by reference bounding box size.
    
    Args:
        distance_px: Raw distance in pixels
        ref_bbox: Reference bounding box [x1, y1, x2, y2]
        method: "bbox_diagonal" or "bbox_sqrt_area"
    
    Returns:
        Normalized distance (unit-less)
    """
    if ref_bbox is None or distance_px == 0.0:
        return distance_px
    
    x1, y1, x2, y2 = ref_bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    if method == "bbox_diagonal":
        diag = np.sqrt(width ** 2 + height ** 2)
        return distance_px / diag if diag > 0 else distance_px
    elif method == "bbox_sqrt_area":
        area = width * height
        sqrt_area = np.sqrt(area)
        return distance_px / sqrt_area if sqrt_area > 0 else distance_px
    else:
        return distance_px


# ============================================================================
# HELPER FUNCTIONS: FILE I/O AND PARSING
# ============================================================================

def load_metadata(metadata_path: Path) -> Optional[ModelMetadata]:
    """Load and parse model metadata from JSON."""
    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return ModelMetadata(
            model_name=data.get("model_name", "unknown"),
            model_path=data.get("model_path", ""),
            mean_inference_ms=float(data.get("mean_inference_ms", 0.0)),
            median_inference_ms=float(data.get("median_inference_ms", 0.0)),
            p95_inference_ms=float(data.get("p95_inference_ms", 0.0)),
            effective_fps=float(data.get("effective_fps", 0.0)),
            num_frames_processed=int(data.get("num_frames_processed", 0)),
            target_fps=float(data.get("target_fps", 30.0)),
            target_ms_per_frame=float(data.get("target_ms_per_frame", 33.3333)),
        )
    except Exception as e:
        warnings.warn(f"Failed to load metadata from {metadata_path}: {e}")
        return None


def normalize_instance(raw_instance: Dict) -> InstanceData:
    """
    Parse and normalize a raw instance dict into standardized InstanceData.
    
    Tolerates missing fields gracefully.
    """
    keypoints_raw = raw_instance.get("keypoints", [])
    keypoints = [
        KeypointData(
            kp_id=int(kp.get("id", i)),
            x=float(kp.get("x", 0.0)),
            y=float(kp.get("y", 0.0)),
            conf=float(kp.get("conf")) if "conf" in kp else None,
        )
        for i, kp in enumerate(keypoints_raw)
    ]
    
    bbox_xyxy = raw_instance.get("bbox_xyxy")
    if bbox_xyxy:
        bbox_xyxy = [float(x) for x in bbox_xyxy]
    
    bbox_centroid = raw_instance.get("bbox_centroid_xy")
    if bbox_centroid:
        bbox_centroid = [float(x) for x in bbox_centroid]
    
    return InstanceData(
        class_id=raw_instance.get("class_id"),
        class_name=raw_instance.get("class_name"),
        confidence=float(raw_instance.get("confidence")) if "confidence" in raw_instance else None,
        bbox_xyxy=bbox_xyxy,
        bbox_centroid_xy=bbox_centroid,
        num_keypoints=raw_instance.get("num_keypoints"),
        num_visible_keypoints=raw_instance.get("num_visible_keypoints"),
        mean_keypoint_confidence=float(raw_instance.get("mean_keypoint_confidence")) if "mean_keypoint_confidence" in raw_instance else None,
        keypoints=keypoints,
    )


def load_predictions_jsonl(predictions_path: Path) -> Dict[int, FramePredictions]:
    """
    Load predictions from a JSONL file, keyed by frame_idx.
    
    Returns:
        Dict mapping frame_idx -> FramePredictions
    """
    frames = {}
    try:
        with open(predictions_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                frame_idx = int(data.get("frame_idx", -1))
                if frame_idx < 0:
                    continue
                
                instances = [
                    normalize_instance(inst) for inst in data.get("instances", [])
                ]
                
                frames[frame_idx] = FramePredictions(
                    frame_idx=frame_idx,
                    timestamp_sec=float(data.get("timestamp_sec", 0.0)),
                    inference_ms=float(data.get("inference_ms")) if "inference_ms" in data else None,
                    instances=instances,
                )
    except Exception as e:
        warnings.warn(f"Failed to load predictions from {predictions_path}: {e}")
    
    return frames


# ============================================================================
# MATCHING AND COMPARISON FUNCTIONS
# ============================================================================

def match_instances_greedy(
    candidate_instances: List[InstanceData],
    reference_instances: List[InstanceData],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[InstanceData, InstanceData]], List[InstanceData], List[InstanceData]]:
    """
    Perform one-to-one greedy matching of candidate instances to reference instances.
    
    Strategy: Match by highest IoU, skip matches below iou_threshold.
    
    Args:
        candidate_instances: Candidate model detections
        reference_instances: Reference model detections
        iou_threshold: Minimum IoU to accept a match
    
    Returns:
        (matched_pairs, unmatched_candidates, unmatched_references)
        where matched_pairs is a list of (candidate, reference) tuples
    """
    matched_pairs = []
    used_ref_indices = set()
    
    # Compute all pairwise IoUs
    ious = []
    for c_idx, cand in enumerate(candidate_instances):
        for r_idx, ref in enumerate(reference_instances):
            # Only match same class if both have class_id
            if (cand.class_id is not None and ref.class_id is not None and
                    cand.class_id != ref.class_id):
                continue
            
            iou = bbox_iou(cand.bbox_xyxy, ref.bbox_xyxy)
            ious.append((iou, c_idx, r_idx))
    
    # Sort by IoU descending
    ious.sort(reverse=True, key=lambda x: x[0])
    
    # Greedy matching: highest IoU first, one-to-one
    used_cand_indices = set()
    for iou, c_idx, r_idx in ious:
        if iou < iou_threshold:
            break
        if c_idx in used_cand_indices or r_idx in used_ref_indices:
            continue
        
        matched_pairs.append((candidate_instances[c_idx], reference_instances[r_idx]))
        used_cand_indices.add(c_idx)
        used_ref_indices.add(r_idx)
    
    unmatched_cands = [
        cand for i, cand in enumerate(candidate_instances) if i not in used_cand_indices
    ]
    unmatched_refs = [
        ref for i, ref in enumerate(reference_instances) if i not in used_ref_indices
    ]
    
    return matched_pairs, unmatched_cands, unmatched_refs


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def compute_detection_metrics(
    all_matched: List[Tuple[InstanceData, InstanceData]],
    all_unmatched_cands: List[InstanceData],
    all_unmatched_refs: List[InstanceData],
) -> DetectionMetrics:
    """Compute TP, FP, FN, precision, recall, F1."""
    tp = len(all_matched)
    fp = len(all_unmatched_cands)
    fn = len(all_unmatched_refs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return DetectionMetrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def compute_bbox_metrics(
    all_matched: List[Tuple[InstanceData, InstanceData]],
) -> BboxMetrics:
    """Compute bounding box IoU metrics for matched pairs."""
    ious = []
    for cand, ref in all_matched:
        iou = bbox_iou(cand.bbox_xyxy, ref.bbox_xyxy)
        ious.append(iou)
    
    if not ious:
        return BboxMetrics()
    
    ious_arr = np.array(ious)
    return BboxMetrics(
        mean_iou=float(np.mean(ious_arr)),
        median_iou=float(np.median(ious_arr)),
        p95_iou=float(np.percentile(ious_arr, 95)),
        all_ious=ious,
    )


def compute_keypoint_metrics(
    all_matched: List[Tuple[InstanceData, InstanceData]],
    normalization_method: str = "bbox_diagonal",
    conf_threshold: float = 0.1,
) -> KeypointMetrics:
    """
    Compute keypoint distance metrics for matched pairs.
    
    Only compares keypoints that:
    1. Exist in both candidate and reference
    2. Have confidence >= conf_threshold (if confidence is available)
    """
    all_distances_px = []
    all_norm_distances = []
    total_possible_kp_pairs = 0
    all_detected_kp_counts = []
    all_visible_kp_counts = []
    all_kp_confs = []
    
    for cand, ref in all_matched:
        cand_kp_map = cand.get_keypoint_map()
        ref_kp_map = ref.get_keypoint_map()
        
        # Find common keypoint IDs
        common_ids = set(cand_kp_map.keys()) & set(ref_kp_map.keys())
        
        if not common_ids:
            continue
        
        for kp_id in common_ids:
            cand_kp = cand_kp_map[kp_id]
            ref_kp = ref_kp_map[kp_id]
            
            # Check confidence threshold
            if conf_threshold > 0:
                if (cand_kp.conf is not None and cand_kp.conf < conf_threshold):
                    continue
                if (ref_kp.conf is not None and ref_kp.conf < conf_threshold):
                    continue
            
            # Compute distance
            dist = euclidean_distance(cand_kp.position(), ref_kp.position())
            all_distances_px.append(dist)
            
            # Compute normalized distance
            norm_dist = normalize_keypoint_distance(
                dist, ref.bbox_xyxy, method=normalization_method
            )
            all_norm_distances.append(norm_dist)
            
            # Track keypoint confidence
            if ref_kp.conf is not None:
                all_kp_confs.append(ref_kp.conf)
        
        # Track detected and visible keypoint counts
        if cand.num_keypoints:
            all_detected_kp_counts.append(cand.num_keypoints)
        if cand.num_visible_keypoints:
            all_visible_kp_counts.append(cand.num_visible_keypoints)
        
        total_possible_kp_pairs += len(common_ids)
    
    if not all_distances_px:
        return KeypointMetrics()
    
    distances_arr = np.array(all_distances_px)
    norm_distances_arr = np.array(all_norm_distances)
    
    return KeypointMetrics(
        mean_distance_px=float(np.mean(distances_arr)),
        median_distance_px=float(np.median(distances_arr)),
        mean_normalized_distance=float(np.mean(norm_distances_arr)) if len(norm_distances_arr) > 0 else 0.0,
        comparable_keypoint_fraction=len(all_distances_px) / (len(all_matched) * 17) if all_matched else 0.0,
        avg_detected_keypoints_per_match=float(np.mean(all_detected_kp_counts)) if all_detected_kp_counts else 0.0,
        avg_visible_keypoints_per_match=float(np.mean(all_visible_kp_counts)) if all_visible_kp_counts else 0.0,
        mean_keypoint_confidence=float(np.mean(all_kp_confs)) if all_kp_confs else 0.0,
        all_distances_px=all_distances_px,
        all_normalized_distances=all_norm_distances,
    )


def compute_per_keypoint_metrics(
    all_matched: List[Tuple[InstanceData, InstanceData]],
    normalization_method: str = "bbox_diagonal",
    conf_threshold: float = 0.1,
) -> Dict[int, List[float]]:
    """
    Compute per-keypoint metrics: keyed by keypoint_id.
    
    Returns:
        Dict mapping keypoint_id -> list of per-keypoint statistics
    """
    per_kp_distances = defaultdict(list)
    per_kp_norm_distances = defaultdict(list)
    
    for cand, ref in all_matched:
        cand_kp_map = cand.get_keypoint_map()
        ref_kp_map = ref.get_keypoint_map()
        
        common_ids = set(cand_kp_map.keys()) & set(ref_kp_map.keys())
        
        for kp_id in common_ids:
            cand_kp = cand_kp_map[kp_id]
            ref_kp = ref_kp_map[kp_id]
            
            # Check confidence threshold
            if conf_threshold > 0:
                if (cand_kp.conf is not None and cand_kp.conf < conf_threshold):
                    continue
                if (ref_kp.conf is not None and ref_kp.conf < conf_threshold):
                    continue
            
            dist = euclidean_distance(cand_kp.position(), ref_kp.position())
            per_kp_distances[kp_id].append(dist)
            
            norm_dist = normalize_keypoint_distance(
                dist, ref.bbox_xyxy, method=normalization_method
            )
            per_kp_norm_distances[kp_id].append(norm_dist)
    
    return per_kp_distances, per_kp_norm_distances


def compute_temporal_stability_metrics(
    frames: Dict[int, FramePredictions],
    match_iou_threshold: float = 0.5,
) -> TemporalMetrics:
    """
    Compute temporal stability metrics by matching consecutive frames.
    
    For each model, match frame t to frame t+1 using bbox IoU, then compute:
    - Count deltas
    - Centroid jitter
    - Area stability
    - Keypoint jitter
    - Empty frame fraction
    """
    frame_indices = sorted(frames.keys())
    if len(frame_indices) < 2:
        return TemporalMetrics()
    
    count_deltas = []
    centroid_jitters = []
    area_deltas = []
    keypoint_jitters = []
    empty_frame_count = 0
    
    for i in range(len(frame_indices) - 1):
        frame_t = frames[frame_indices[i]]
        frame_t1 = frames[frame_indices[i + 1]]
        
        # Count delta
        count_deltas.append(abs(len(frame_t.instances) - len(frame_t1.instances)))
        
        if len(frame_t.instances) == 0:
            empty_frame_count += 1
            continue
        if len(frame_t1.instances) == 0:
            empty_frame_count += 1
            continue
        
        # Match instances between frames
        matched, _, _ = match_instances_greedy(
            frame_t.instances, frame_t1.instances, iou_threshold=match_iou_threshold
        )
        
        # Centroid jitter
        for inst_t, inst_t1 in matched:
            if inst_t.bbox_centroid_xy and inst_t1.bbox_centroid_xy:
                jitter = euclidean_distance(
                    inst_t.bbox_centroid_xy, inst_t1.bbox_centroid_xy
                )
                centroid_jitters.append(jitter)
            
            # Area delta
            area_t = inst_t.bbox_area()
            area_t1 = inst_t1.bbox_area()
            area_deltas.append(abs(area_t - area_t1))
            
            # Keypoint jitter
            inst_t_kp_map = inst_t.get_keypoint_map()
            inst_t1_kp_map = inst_t1.get_keypoint_map()
            common_ids = set(inst_t_kp_map.keys()) & set(inst_t1_kp_map.keys())
            
            for kp_id in common_ids:
                kp_t = inst_t_kp_map[kp_id]
                kp_t1 = inst_t1_kp_map[kp_id]
                kp_jitter = euclidean_distance(kp_t.position(), kp_t1.position())
                keypoint_jitters.append(kp_jitter)
    
    total_frames = len(frame_indices)
    empty_frame_fraction = empty_frame_count / total_frames if total_frames > 0 else 0.0
    
    return TemporalMetrics(
        mean_abs_count_delta=float(np.mean(count_deltas)) if count_deltas else 0.0,
        mean_bbox_centroid_jitter=float(np.mean(centroid_jitters)) if centroid_jitters else 0.0,
        mean_bbox_area_delta=float(np.mean(area_deltas)) if area_deltas else 0.0,
        mean_keypoint_jitter=float(np.mean(keypoint_jitters)) if keypoint_jitters else 0.0,
        empty_frame_fraction=empty_frame_fraction,
    )


def compute_timing_metrics(metadata: ModelMetadata) -> TimingMetrics:
    """Extract timing metrics from metadata."""
    return TimingMetrics(
        mean_inference_ms=metadata.mean_inference_ms,
        median_inference_ms=metadata.median_inference_ms,
        p95_inference_ms=metadata.p95_inference_ms,
        effective_fps=metadata.effective_fps,
        meets_30hz_mean=metadata.mean_inference_ms <= metadata.target_ms_per_frame,
        meets_30hz_p95=metadata.p95_inference_ms <= metadata.target_ms_per_frame,
    )


def compute_final_score(evaluation: ModelEvaluation, weights: Dict[str, float]) -> float:
    """
    Compute a weighted final score for ranking models.
    
    Components:
    - f1: detection F1 score (directly)
    - keypoint_accuracy: 1 - (normalized_keypoint_distance / some_reference)
    - bbox_iou: mean bbox IoU (directly)
    - speed: inverse of mean inference time, normalized
    - stability: inverse of centroid jitter, normalized
    
    All sub-scores normalized to [0, 1] range.
    """
    # F1 is already in [0, 1]
    f1_score = evaluation.detection_metrics.f1
    
    # Keypoint accuracy: reward lower distance
    # Invert and normalize: assume reasonable max distance is 100 pixels
    kp_distance = evaluation.keypoint_metrics.mean_distance_px
    keypoint_accuracy_score = max(0, 1 - (kp_distance / 100.0)) if kp_distance > 0 else 1.0
    
    # BBox IoU is already in [0, 1]
    bbox_score = evaluation.bbox_metrics.mean_iou
    
    # Speed: penalize slow models, reward fast ones
    # Reference: 33.33 ms for 30 Hz
    # If < 33.33 ms, score = 1.0
    # If > 100 ms, score = 0.0
    mean_inf_ms = evaluation.timing_metrics.mean_inference_ms
    if mean_inf_ms <= 33.33:
        speed_score = 1.0
    elif mean_inf_ms >= 100:
        speed_score = 0.0
    else:
        speed_score = 1 - ((mean_inf_ms - 33.33) / (100 - 33.33))
    
    # Stability: reward low centroid jitter
    # Reference: < 10 pixels is good
    centroid_jitter = evaluation.temporal_metrics.mean_bbox_centroid_jitter
    if centroid_jitter < 5:
        stability_score = 1.0
    elif centroid_jitter > 50:
        stability_score = 0.0
    else:
        stability_score = 1 - ((centroid_jitter - 5) / (50 - 5))
    
    # Compute weighted score
    final_score = (
        weights.get("f1", 0) * f1_score +
        weights.get("keypoint_accuracy", 0) * keypoint_accuracy_score +
        weights.get("bbox_iou", 0) * bbox_score +
        weights.get("speed", 0) * speed_score +
        weights.get("stability", 0) * stability_score
    )
    
    return final_score


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_all_models(
    reference_model_name: str,
    benchmark_dir: Path,
    iou_threshold: float = 0.5,
    kp_conf_threshold: float = 0.1,
    normalization_method: str = "bbox_diagonal",
    score_weights: Dict[str, float] = None,
) -> Tuple[Dict[str, ModelEvaluation], Dict[int, List[PerKeypointMetrics]]]:
    """
    Evaluate all candidate models against a reference model.
    
    Returns:
        (evaluations_by_model, per_keypoint_metrics_by_model)
    """
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    
    if score_weights is None:
        score_weights = SCORE_WEIGHTS
    
    # Find all model folders
    model_folders = {d.name: d for d in benchmark_dir.iterdir() if d.is_dir()}
    
    if reference_model_name not in model_folders:
        raise ValueError(f"Reference model '{reference_model_name}' not found in {benchmark_dir}")
    
    print(f"\n{'=' * 80}")
    print(f"YOLO POSE MODEL EVALUATION")
    print(f"{'=' * 80}")
    print(f"Reference model: {reference_model_name}")
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Number of candidate models: {len(model_folders) - 1}")
    print(f"{'=' * 80}\n")
    
    # Load reference model
    ref_folder = model_folders[reference_model_name]
    ref_metadata = load_metadata(ref_folder / "metadata.json")
    ref_predictions = load_predictions_jsonl(ref_folder / "predictions.jsonl")
    
    if not ref_metadata or not ref_predictions:
        raise RuntimeError(f"Failed to load reference model from {ref_folder}")
    
    print(f"✓ Loaded reference model: {len(ref_predictions)} frames")
    
    # Evaluate each candidate model
    evaluations = {}
    per_keypoint_metrics_all = defaultdict(list)
    
    for model_name in sorted(model_folders.keys()):
        if model_name == reference_model_name:
            continue
        
        print(f"\nEvaluating {model_name}...", end=" ")
        
        cand_folder = model_folders[model_name]
        cand_metadata = load_metadata(cand_folder / "metadata.json")
        cand_predictions = load_predictions_jsonl(cand_folder / "predictions.jsonl")
        
        if not cand_metadata or not cand_predictions:
            warnings.warn(f"  ✗ Skipped (missing files)")
            continue
        
        # Find common frames
        common_frames = set(cand_predictions.keys()) & set(ref_predictions.keys())
        if not common_frames:
            warnings.warn(f"  ✗ Skipped (no overlapping frames)")
            continue
        
        # Compare frame by frame
        all_matched = []
        all_unmatched_cands = []
        all_unmatched_refs = []
        
        for frame_idx in sorted(common_frames):
            cand_frame = cand_predictions[frame_idx]
            ref_frame = ref_predictions[frame_idx]
            
            matched, unmatched_c, unmatched_r = match_instances_greedy(
                cand_frame.instances, ref_frame.instances, iou_threshold=iou_threshold
            )
            
            all_matched.extend(matched)
            all_unmatched_cands.extend(unmatched_c)
            all_unmatched_refs.extend(unmatched_r)
        
        # Compute all metrics
        detection_metrics = compute_detection_metrics(
            all_matched, all_unmatched_cands, all_unmatched_refs
        )
        bbox_metrics = compute_bbox_metrics(all_matched)
        keypoint_metrics = compute_keypoint_metrics(
            all_matched, normalization_method=normalization_method,
            conf_threshold=kp_conf_threshold
        )
        temporal_metrics = compute_temporal_stability_metrics(cand_predictions)
        timing_metrics = compute_timing_metrics(cand_metadata)
        
        # Per-keypoint metrics
        per_kp_distances, per_kp_norm_distances = compute_per_keypoint_metrics(
            all_matched, normalization_method=normalization_method,
            conf_threshold=kp_conf_threshold
        )
        
        for kp_id in per_kp_distances.keys():
            distances = per_kp_distances[kp_id]
            norm_distances = per_kp_norm_distances[kp_id]
            
            per_keypoint_metrics_all[kp_id].append(
                PerKeypointMetrics(
                    model_name=model_name,
                    keypoint_id=kp_id,
                    mean_distance_px=float(np.mean(distances)) if distances else 0.0,
                    median_distance_px=float(np.median(distances)) if distances else 0.0,
                    mean_normalized_distance=float(np.mean(norm_distances)) if norm_distances else 0.0,
                    num_compared=len(distances),
                )
            )
        
        # Create evaluation object
        evaluation = ModelEvaluation(
            model_name=model_name,
            reference_model_name=reference_model_name,
            num_frames_compared=len(common_frames),
            detection_metrics=detection_metrics,
            bbox_metrics=bbox_metrics,
            keypoint_metrics=keypoint_metrics,
            temporal_metrics=temporal_metrics,
            timing_metrics=timing_metrics,
            final_score=0.0,  # Will compute below
        )
        
        # Compute final score
        evaluation.final_score = compute_final_score(evaluation, score_weights)
        evaluations[model_name] = evaluation
        
        print(f"✓ (F1={detection_metrics.f1:.3f}, Score={evaluation.final_score:.3f})")
    
    print(f"\n{'=' * 80}")
    print(f"Completed evaluation of {len(evaluations)} models")
    print(f"{'=' * 80}\n")
    
    return evaluations, dict(per_keypoint_metrics_all)


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def generate_summary_csv(
    evaluations: Dict[str, ModelEvaluation],
    output_path: Path,
) -> None:
    """Generate summary CSV with per-model metrics."""
    rows = []
    
    for model_name in sorted(evaluations.keys()):
        ev = evaluations[model_name]
        rows.append({
            "model_name": model_name,
            "reference_model": ev.reference_model_name,
            "num_frames_compared": ev.num_frames_compared,
            "tp": ev.detection_metrics.tp,
            "fp": ev.detection_metrics.fp,
            "fn": ev.detection_metrics.fn,
            "precision": round(ev.detection_metrics.precision, 4),
            "recall": round(ev.detection_metrics.recall, 4),
            "f1": round(ev.detection_metrics.f1, 4),
            "mean_bbox_iou": round(ev.bbox_metrics.mean_iou, 4),
            "median_bbox_iou": round(ev.bbox_metrics.median_iou, 4),
            "p95_bbox_iou": round(ev.bbox_metrics.p95_iou, 4),
            "mean_keypoint_distance_px": round(ev.keypoint_metrics.mean_distance_px, 2),
            "median_keypoint_distance_px": round(ev.keypoint_metrics.median_distance_px, 2),
            "mean_normalized_keypoint_distance": round(ev.keypoint_metrics.mean_normalized_distance, 4),
            "comparable_keypoint_fraction": round(ev.keypoint_metrics.comparable_keypoint_fraction, 4),
            "avg_detected_keypoints_per_match": round(ev.keypoint_metrics.avg_detected_keypoints_per_match, 2),
            "avg_visible_keypoints_per_match": round(ev.keypoint_metrics.avg_visible_keypoints_per_match, 2),
            "mean_keypoint_confidence": round(ev.keypoint_metrics.mean_keypoint_confidence, 4),
            "mean_abs_count_delta": round(ev.temporal_metrics.mean_abs_count_delta, 2),
            "mean_bbox_centroid_jitter": round(ev.temporal_metrics.mean_bbox_centroid_jitter, 2),
            "mean_bbox_area_delta": round(ev.temporal_metrics.mean_bbox_area_delta, 0),
            "mean_keypoint_jitter": round(ev.temporal_metrics.mean_keypoint_jitter, 2),
            "empty_frame_fraction": round(ev.temporal_metrics.empty_frame_fraction, 4),
            "mean_inference_ms": round(ev.timing_metrics.mean_inference_ms, 2),
            "median_inference_ms": round(ev.timing_metrics.median_inference_ms, 2),
            "p95_inference_ms": round(ev.timing_metrics.p95_inference_ms, 2),
            "effective_fps": round(ev.timing_metrics.effective_fps, 2),
            "meets_30hz_mean": ev.timing_metrics.meets_30hz_mean,
            "meets_30hz_p95": ev.timing_metrics.meets_30hz_p95,
            "final_score": round(ev.final_score, 4),
        })
    
    # Sort by final_score descending
    rows.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"✓ Summary CSV: {output_path}")


def generate_summary_json(
    evaluations: Dict[str, ModelEvaluation],
    output_path: Path,
) -> None:
    """Generate detailed JSON report."""
    report = {
        "evaluation_results": {},
        "summary": {
            "num_models": len(evaluations),
            "best_model": None,
            "worst_model": None,
        }
    }
    
    sorted_models = sorted(evaluations.items(), key=lambda x: x[1].final_score, reverse=True)
    
    if sorted_models:
        report["summary"]["best_model"] = sorted_models[0][0]
        report["summary"]["worst_model"] = sorted_models[-1][0]
    
    for model_name, ev in sorted_models:
        report["evaluation_results"][model_name] = {
            "reference_model": ev.reference_model_name,
            "num_frames_compared": ev.num_frames_compared,
            "detection": {
                "tp": ev.detection_metrics.tp,
                "fp": ev.detection_metrics.fp,
                "fn": ev.detection_metrics.fn,
                "precision": round(ev.detection_metrics.precision, 4),
                "recall": round(ev.detection_metrics.recall, 4),
                "f1": round(ev.detection_metrics.f1, 4),
            },
            "bounding_boxes": {
                "mean_iou": round(ev.bbox_metrics.mean_iou, 4),
                "median_iou": round(ev.bbox_metrics.median_iou, 4),
                "p95_iou": round(ev.bbox_metrics.p95_iou, 4),
            },
            "keypoints": {
                "mean_distance_px": round(ev.keypoint_metrics.mean_distance_px, 2),
                "median_distance_px": round(ev.keypoint_metrics.median_distance_px, 2),
                "mean_normalized_distance": round(ev.keypoint_metrics.mean_normalized_distance, 4),
                "comparable_keypoint_fraction": round(ev.keypoint_metrics.comparable_keypoint_fraction, 4),
                "avg_detected_keypoints_per_match": round(ev.keypoint_metrics.avg_detected_keypoints_per_match, 2),
                "avg_visible_keypoints_per_match": round(ev.keypoint_metrics.avg_visible_keypoints_per_match, 2),
                "mean_keypoint_confidence": round(ev.keypoint_metrics.mean_keypoint_confidence, 4),
            },
            "temporal_stability": {
                "mean_abs_count_delta": round(ev.temporal_metrics.mean_abs_count_delta, 2),
                "mean_bbox_centroid_jitter": round(ev.temporal_metrics.mean_bbox_centroid_jitter, 2),
                "mean_bbox_area_delta": round(ev.temporal_metrics.mean_bbox_area_delta, 0),
                "mean_keypoint_jitter": round(ev.temporal_metrics.mean_keypoint_jitter, 2),
                "empty_frame_fraction": round(ev.temporal_metrics.empty_frame_fraction, 4),
            },
            "timing": {
                "mean_inference_ms": round(ev.timing_metrics.mean_inference_ms, 2),
                "median_inference_ms": round(ev.timing_metrics.median_inference_ms, 2),
                "p95_inference_ms": round(ev.timing_metrics.p95_inference_ms, 2),
                "effective_fps": round(ev.timing_metrics.effective_fps, 2),
                "meets_30hz_mean": ev.timing_metrics.meets_30hz_mean,
                "meets_30hz_p95": ev.timing_metrics.meets_30hz_p95,
            },
            "final_score": round(ev.final_score, 4),
        }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Summary JSON: {output_path}")


def generate_per_keypoint_csv(
    per_keypoint_metrics: Dict[int, List[PerKeypointMetrics]],
    output_path: Path,
) -> None:
    """Generate per-keypoint metrics CSV."""
    rows = []
    
    for kp_id in sorted(per_keypoint_metrics.keys()):
        for metric in per_keypoint_metrics[kp_id]:
            rows.append({
                "keypoint_id": metric.keypoint_id,
                "model_name": metric.model_name,
                "mean_distance_px": round(metric.mean_distance_px, 2),
                "median_distance_px": round(metric.median_distance_px, 2),
                "mean_normalized_distance": round(metric.mean_normalized_distance, 4),
                "num_compared": metric.num_compared,
            })
    
    if rows:
        # Sort by keypoint_id, then model_name
        rows.sort(key=lambda x: (x["keypoint_id"], x["model_name"]))
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ Per-keypoint CSV: {output_path}")
    else:
        print(f"⚠ No per-keypoint data to export")


def print_ranking_table(evaluations: Dict[str, ModelEvaluation]) -> None:
    """Print a nicely formatted ranking table."""
    sorted_models = sorted(evaluations.items(), key=lambda x: x[1].final_score, reverse=True)
    
    print(f"\n{'=' * 100}")
    print(f"MODEL RANKING (by Final Score)")
    print(f"{'=' * 100}")
    print(f"{'Rank':<6} {'Model Name':<25} {'F1':<8} {'Keypoint Dist':<15} {'Speed':<10} {'Score':<8}")
    print(f"{'-' * 100}")
    
    for rank, (model_name, ev) in enumerate(sorted_models, 1):
        f1 = ev.detection_metrics.f1
        kp_dist = ev.keypoint_metrics.mean_distance_px
        mean_ms = ev.timing_metrics.mean_inference_ms
        score = ev.final_score
        
        print(f"{rank:<6} {model_name:<25} {f1:<8.4f} {kp_dist:<15.2f} px {mean_ms:<10.2f} ms {score:<8.4f}")
    
    print(f"{'=' * 100}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    try:
        # Run evaluation
        evaluations, per_keypoint_metrics = evaluate_all_models(
            reference_model_name=REFERENCE_MODEL_NAME,
            benchmark_dir=BENCHMARK_BASE_DIR,
            iou_threshold=MATCH_IOU_THRESHOLD,
            kp_conf_threshold=KEYPOINT_CONF_THRESHOLD,
            normalization_method=KEYPOINT_DISTANCE_NORMALIZATION,
            score_weights=SCORE_WEIGHTS,
        )
        
        if not evaluations:
            print("⚠ No models were evaluated. Check the benchmark directory and configuration.")
            return
        
        # Print ranking
        print_ranking_table(evaluations)
        
        # Generate outputs
        print(f"\nGenerating output files in {OUTPUT_DIR}...")
        generate_summary_csv(evaluations, OUTPUT_DIR / "pose_evaluation_summary.csv")
        generate_summary_json(evaluations, OUTPUT_DIR / "pose_evaluation_summary.json")
        generate_per_keypoint_csv(per_keypoint_metrics, OUTPUT_DIR / "pose_per_keypoint_metrics.csv")
        
        print(f"\n{'=' * 80}")
        print(f"✓ Evaluation complete!")
        print(f"{'=' * 80}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
