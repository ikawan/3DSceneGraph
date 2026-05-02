#!/usr/bin/env python3
"""
Evaluate YOLO segmentation model outputs against a pseudo-ground-truth reference run.

This script compares saved model outputs in a results directory to a chosen reference model.
The reference model is treated as pseudo-ground-truth, so the metrics measure
agreement with the reference, plus timing and temporal stability.

Outputs:
- evaluation_summary.csv
- evaluation_summary.json
- evaluation_per_frame_metrics.csv

The script is intentionally tolerant of missing fields and schema variations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# Configuration
# When running from the repository root folder `Yolo`, this points to the existing benchmark outputs.
EVAL_ROOT = Path("YoloBenchmark/YoloSeg/benchmarks")
FALLBACK_ROOT = Path("runs_eval")
REFERENCE_MODEL = "yolo26x-seg"  # Change this to your chosen reference model folder or metadata model_name
MATCH_IOU_THRESHOLD = 0.5
TEMPORAL_IOU_THRESHOLD = 0.3
TARGET_FPS = 30.0
TARGET_MS_PER_FRAME = 1000.0 / TARGET_FPS
OUTPUT_SUMMARY_CSV = "evaluation_summary.csv"
OUTPUT_SUMMARY_JSON = "evaluation_summary.json"
OUTPUT_PER_FRAME_CSV = "evaluation_per_frame_metrics.csv"

# Weighted ranking formula (higher is better). Adjust as needed.
SCORE_WEIGHTS = {
    "f1": 0.35,
    "mean_mask_iou": 0.20,
    "mean_dice": 0.15,
    "speed_score": 0.15,
    "stability_score": 0.15,
}


def safe_divide(numerator: float, denominator: float, default: Optional[float] = None) -> Optional[float]:
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator


def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        warnings.warn(f"Missing JSON file: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        warnings.warn(f"Invalid JSON in {path}: {exc}")
        return None


def load_metadata(path: Path) -> Dict[str, Any]:
    raw = load_json_file(path)
    return raw if raw is not None else {}


def normalize_bbox(raw_bbox: Any) -> Optional[List[float]]:
    if not raw_bbox or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    except (TypeError, ValueError):
        return None
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    return [x_min, y_min, x_max, y_max]


def normalize_centroid(raw_centroid: Any) -> Optional[List[float]]:
    if not raw_centroid or len(raw_centroid) < 2:
        return None
    try:
        return [float(raw_centroid[0]), float(raw_centroid[1])]
    except (TypeError, ValueError):
        return None


def normalize_polygon(raw_polygon: Any) -> List[List[int]]:
    if not raw_polygon or not isinstance(raw_polygon, list):
        return []
    points: List[List[int]] = []
    for point in raw_polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except (TypeError, ValueError):
            continue
        points.append([x, y])
    if len(points) < 3:
        return []
    return points


def normalize_instance(raw_instance: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instance_idx": raw_instance.get("instance_idx"),
        "class_id": int(raw_instance["class_id"]) if raw_instance.get("class_id") is not None else None,
        "class_name": str(raw_instance.get("class_name")) if raw_instance.get("class_name") is not None else None,
        "confidence": float(raw_instance["confidence"]) if raw_instance.get("confidence") is not None else None,
        "bbox_xyxy": normalize_bbox(raw_instance.get("bbox_xyxy")),
        "centroid_xy": normalize_centroid(raw_instance.get("centroid_xy")),
        "mask_area_px": float(raw_instance["mask_area_px"]) if raw_instance.get("mask_area_px") is not None else None,
        "polygon_xy": normalize_polygon(raw_instance.get("polygon_xy")),
    }


def normalize_frame(raw_frame: Dict[str, Any]) -> Dict[str, Any]:
    frame_idx = raw_frame.get("frame_idx")
    if frame_idx is None:
        frame_idx = raw_frame.get("timestamp_sec")
    try:
        frame_idx = int(frame_idx)
    except (TypeError, ValueError):
        frame_idx = int(raw_frame.get("frame_idx", 0) or 0)

    instances_raw = raw_frame.get("instances") or []
    instances = [normalize_instance(inst) for inst in instances_raw if isinstance(inst, dict)]

    return {
        "frame_idx": frame_idx,
        "timestamp_sec": float(raw_frame.get("timestamp_sec")) if raw_frame.get("timestamp_sec") is not None else None,
        "inference_ms": float(raw_frame.get("inference_ms")) if raw_frame.get("inference_ms") is not None else None,
        "num_instances": int(raw_frame.get("num_instances")) if raw_frame.get("num_instances") is not None else len(instances),
        "instances": instances,
    }


def load_predictions_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    frames: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        warnings.warn(f"Missing predictions file: {path}")
        return frames

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_frame = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(f"Skipping invalid JSON line {line_idx} in {path}: {exc}")
                continue
            frame = normalize_frame(raw_frame)
            frame_idx = frame["frame_idx"]
            if frame_idx in frames:
                warnings.warn(f"Duplicate frame_idx {frame_idx} in {path}; overwriting previous frame")
            frames[frame_idx] = frame
    return dict(sorted(frames.items()))


def valid_polygon(polygon: List[List[int]]) -> bool:
    return len(polygon) >= 3


def bbox_area(bbox: List[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: List[float], b: List[float]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def rasterize_polygons(poly_a: List[List[int]], poly_b: List[List[int]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not valid_polygon(poly_a) or not valid_polygon(poly_b):
        return None

    all_points = np.array(poly_a + poly_b, dtype=np.int32)
    x_min = int(max(0, all_points[:, 0].min()))
    y_min = int(max(0, all_points[:, 1].min()))
    x_max = int(all_points[:, 0].max())
    y_max = int(all_points[:, 1].max())

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if width <= 0 or height <= 0:
        return None

    if width > 5000 or height > 5000:
        return None

    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros((height, width), dtype=np.uint8)

    shifted_a = np.array([[x - x_min, y - y_min] for x, y in poly_a], dtype=np.int32)
    shifted_b = np.array([[x - x_min, y - y_min] for x, y in poly_b], dtype=np.int32)
    cv2.fillPoly(mask_a, [shifted_a], 1)
    cv2.fillPoly(mask_b, [shifted_b], 1)
    return mask_a, mask_b


def compute_mask_overlap(poly_a: List[List[int]], poly_b: List[List[int]]) -> Tuple[Optional[float], Optional[float]]:
    raster = rasterize_polygons(poly_a, poly_b)
    if raster is None:
        return None, None
    mask_a, mask_b = raster
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return None, None
    iou = intersection / union
    area_a = float(mask_a.sum())
    area_b = float(mask_b.sum())
    dice = 2.0 * intersection / (area_a + area_b) if (area_a + area_b) > 0 else None
    return iou, dice


def compute_pair_overlap(candidate: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Optional[float]]:
    bbox_a = candidate.get("bbox_xyxy")
    bbox_b = reference.get("bbox_xyxy")
    poly_a = candidate.get("polygon_xy")
    poly_b = reference.get("polygon_xy")

    bbox_overlap = bbox_iou(bbox_a, bbox_b) if bbox_a and bbox_b else 0.0
    mask_iou = None
    dice = None
    if valid_polygon(poly_a) and valid_polygon(poly_b):
        mask_iou, dice = compute_mask_overlap(poly_a, poly_b)

    best_overlap = mask_iou if mask_iou is not None else bbox_overlap
    return {
        "best_overlap": best_overlap,
        "bbox_iou": bbox_overlap if bbox_a and bbox_b else None,
        "mask_iou": mask_iou,
        "dice": dice,
    }


def greedy_match_instances(
    candidates: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, Dict[str, Optional[float]]]], List[int], List[int]]:
    pairs: List[Tuple[float, int, int, Dict[str, Optional[float]]]] = []
    for c_idx, cand in enumerate(candidates):
        for r_idx, ref in enumerate(references):
            cand_class = cand.get("class_id")
            ref_class = ref.get("class_id")
            if cand_class is None or ref_class is None or cand_class != ref_class:
                continue
            overlap_info = compute_pair_overlap(cand, ref)
            score = overlap_info["best_overlap"]
            if score is None or score < iou_threshold:
                continue
            pairs.append((score, c_idx, r_idx, overlap_info))

    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_candidates = set()
    matched_references = set()
    matches: List[Tuple[int, int, Dict[str, Optional[float]]]] = []
    for score, c_idx, r_idx, overlap_info in pairs:
        if c_idx in matched_candidates or r_idx in matched_references:
            continue
        matched_candidates.add(c_idx)
        matched_references.add(r_idx)
        matches.append((c_idx, r_idx, overlap_info))
    unmatched_candidates = [i for i in range(len(candidates)) if i not in matched_candidates]
    unmatched_references = [i for i in range(len(references)) if i not in matched_references]
    return matches, unmatched_candidates, unmatched_references


def build_timing_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean_inference_ms": None,
            "median_inference_ms": None,
            "p95_inference_ms": None,
            "effective_fps": None,
            "meets_30hz_mean": None,
            "meets_30hz_p95": None,
        }
    mean_ms = statistics.mean(values)
    median_ms = statistics.median(values)
    p95_ms = float(np.percentile(np.array(values, dtype=np.float64), 95))
    effective_fps = 1000.0 / mean_ms if mean_ms > 0 else None
    return {
        "mean_inference_ms": mean_ms,
        "median_inference_ms": median_ms,
        "p95_inference_ms": p95_ms,
        "effective_fps": effective_fps,
        "meets_30hz_mean": mean_ms <= TARGET_MS_PER_FRAME,
        "meets_30hz_p95": p95_ms <= TARGET_MS_PER_FRAME,
    }


def compute_temporal_stability(frames: Dict[int, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    sorted_frame_indices = sorted(frames)
    if len(sorted_frame_indices) < 2:
        return {
            "mean_abs_count_delta": None,
            "mean_centroid_jitter": None,
            "mean_mask_area_delta": None,
            "empty_frame_fraction": 0.0 if frames else None,
        }

    count_deltas: List[float] = []
    centroid_displacements: List[float] = []
    mask_area_deltas: List[float] = []
    empty_frames = 0
    total_frames = len(sorted_frame_indices)

    for frame_idx in sorted_frame_indices:
        if not frames[frame_idx].get("instances"):
            empty_frames += 1

    for prev_idx, next_idx in zip(sorted_frame_indices, sorted_frame_indices[1:]):
        prev_instances = frames[prev_idx]["instances"]
        next_instances = frames[next_idx]["instances"]
        count_deltas.append(abs(len(next_instances) - len(prev_instances)))
        matches, _, _ = greedy_match_instances(prev_instances, next_instances, TEMPORAL_IOU_THRESHOLD)
        for prev_i, next_i, _info in matches:
            prev_inst = prev_instances[prev_i]
            next_inst = next_instances[next_i]
            prev_centroid = prev_inst.get("centroid_xy")
            next_centroid = next_inst.get("centroid_xy")
            if prev_centroid and next_centroid:
                dx = prev_centroid[0] - next_centroid[0]
                dy = prev_centroid[1] - next_centroid[1]
                centroid_displacements.append(math.hypot(dx, dy))
            prev_area = prev_inst.get("mask_area_px")
            next_area = next_inst.get("mask_area_px")
            if prev_area is not None and next_area is not None:
                mask_area_deltas.append(abs(prev_area - next_area))

    return {
        "mean_abs_count_delta": statistics.mean(count_deltas) if count_deltas else None,
        "mean_centroid_jitter": statistics.mean(centroid_displacements) if centroid_displacements else None,
        "mean_mask_area_delta": statistics.mean(mask_area_deltas) if mask_area_deltas else None,
        "empty_frame_fraction": empty_frames / total_frames if total_frames else None,
    }


def evaluate_model_against_reference(
    candidate_frames: Dict[int, Dict[str, Any]],
    reference_frames: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    frame_indices = sorted(set(candidate_frames) & set(reference_frames))
    tp = 0
    fp = 0
    fn = 0
    bbox_iou_values: List[float] = []
    mask_iou_values: List[float] = []
    dice_values: List[float] = []
    per_frame_rows: List[Dict[str, Any]] = []

    for frame_idx in frame_indices:
        candidate_frame = candidate_frames[frame_idx]
        reference_frame = reference_frames[frame_idx]
        matches, unmatched_cands, unmatched_refs = greedy_match_instances(
            candidate_frame["instances"], reference_frame["instances"], MATCH_IOU_THRESHOLD
        )
        frame_tp = len(matches)
        frame_fp = len(unmatched_cands)
        frame_fn = len(unmatched_refs)
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn

        for _, _, overlap_info in matches:
            if overlap_info["bbox_iou"] is not None:
                bbox_iou_values.append(overlap_info["bbox_iou"])
            if overlap_info["mask_iou"] is not None:
                mask_iou_values.append(overlap_info["mask_iou"])
            if overlap_info["dice"] is not None:
                dice_values.append(overlap_info["dice"])

        per_frame_rows.append(
            {
                "frame_idx": frame_idx,
                "tp": frame_tp,
                "fp": frame_fp,
                "fn": frame_fn,
                "matched_pairs": frame_tp,
                "candidate_count": len(candidate_frame["instances"]),
                "reference_count": len(reference_frame["instances"]),
                "mean_bbox_iou": statistics.mean(bbox_iou_values[-frame_tp:]) if frame_tp and bbox_iou_values else None,
            }
        )

    precision = safe_divide(tp, tp + fp, 0.0)
    recall = safe_divide(tp, tp + fn, 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    overlap_metrics = {
        "mean_bbox_iou": statistics.mean(bbox_iou_values) if bbox_iou_values else None,
        "mean_mask_iou": statistics.mean(mask_iou_values) if mask_iou_values else None,
        "mean_dice": statistics.mean(dice_values) if dice_values else None,
    }

    result = {
        "frames_compared": len(frame_indices),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_agreement_rate": 1.0 if tp > 0 else None,
        **overlap_metrics,
    }
    return result, per_frame_rows


def score_model(metrics: Dict[str, Any]) -> float:
    f1 = metrics.get("f1") or 0.0
    mean_mask_iou = metrics.get("mean_mask_iou") or 0.0
    mean_dice = metrics.get("mean_dice") or 0.0
    mean_inference_ms = metrics.get("mean_inference_ms") or float("inf")
    speed_score = min(1.0, TARGET_MS_PER_FRAME / mean_inference_ms) if mean_inference_ms > 0 else 0.0

    count_delta = metrics.get("mean_abs_count_delta") or 0.0
    centroid_jitter = metrics.get("mean_centroid_jitter") or 0.0
    mask_area_delta = metrics.get("mean_mask_area_delta") or 0.0
    stability_penalty = count_delta + centroid_jitter / 100.0 + mask_area_delta / 1000.0
    stability_score = 1.0 / (1.0 + stability_penalty)

    weighted_score = (
        SCORE_WEIGHTS["f1"] * f1
        + SCORE_WEIGHTS["mean_mask_iou"] * mean_mask_iou
        + SCORE_WEIGHTS["mean_dice"] * mean_dice
        + SCORE_WEIGHTS["speed_score"] * speed_score
        + SCORE_WEIGHTS["stability_score"] * stability_score
    )
    return weighted_score


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_model_registry(root_dir: Path) -> List[Dict[str, Any]]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Evaluation root directory does not exist: {root_dir}")

    models: List[Dict[str, Any]] = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        metadata_path = child / "metadata.json"
        predictions_path = child / "predictions.jsonl"
        if not metadata_path.exists() or not predictions_path.exists():
            continue
        metadata = load_metadata(metadata_path)
        models.append(
            {
                "folder_name": child.name,
                "root_path": child,
                "metadata_path": metadata_path,
                "predictions_path": predictions_path,
                "metadata": metadata,
            }
        )
    return models


def find_reference_model(models: List[Dict[str, Any]], reference_name: str) -> Optional[Dict[str, Any]]:
    for model in models:
        if model["folder_name"] == reference_name:
            return model
        metadata_name = model["metadata"].get("model_name")
        if metadata_name == reference_name:
            return model
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YOLO segmentation runs against a reference model")
    parser.add_argument("--root", type=Path, default=EVAL_ROOT, help="Root directory containing model run folders")
    parser.add_argument("--reference", type=str, default=REFERENCE_MODEL, help="Reference model folder or metadata model_name")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write evaluation reports")
    args = parser.parse_args()

    root_dir = args.root
    if not root_dir.exists() and FALLBACK_ROOT.exists():
        warnings.warn(f"Root {root_dir} not found; falling back to {FALLBACK_ROOT}")
        root_dir = FALLBACK_ROOT

    models = build_model_registry(root_dir)
    if not models:
        raise RuntimeError(f"No valid model result directories found under {root_dir}")

    reference_model = find_reference_model(models, args.reference)
    if reference_model is None:
        available = [m["folder_name"] for m in models]
        raise ValueError(
            f"Reference model '{args.reference}' not found under {root_dir}. Available models: {available}"
        )

    output_dir = args.output_dir or root_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / OUTPUT_SUMMARY_CSV
    summary_json_path = output_dir / OUTPUT_SUMMARY_JSON
    per_frame_csv_path = output_dir / OUTPUT_PER_FRAME_CSV

    print(f"Loading reference model: {reference_model['folder_name']}")
    reference_frames = load_predictions_jsonl(reference_model["predictions_path"])
    if not reference_frames:
        raise RuntimeError(f"No reference predictions loaded from {reference_model['predictions_path']}")
    reference_timing = [frame["inference_ms"] for frame in reference_frames.values() if frame.get("inference_ms") is not None]
    reference_timing_metrics = build_timing_stats(reference_timing)
    reference_stability = compute_temporal_stability(reference_frames)

    results: List[Dict[str, Any]] = []
    per_frame_metrics: List[Dict[str, Any]] = []
    for model in models:
        if model is reference_model:
            continue
        print(f"Evaluating candidate: {model['folder_name']}")
        candidate_frames = load_predictions_jsonl(model["predictions_path"])
        if not candidate_frames:
            warnings.warn(f"Skipping {model['folder_name']} because predictions could not be loaded")
            continue

        detection_metrics, frame_rows = evaluate_model_against_reference(candidate_frames, reference_frames)
        timing_values = [frame["inference_ms"] for frame in candidate_frames.values() if frame.get("inference_ms") is not None]
        if not timing_values:
            metadata_timing = model["metadata"].get("mean_inference_ms")
            if metadata_timing is not None:
                timing_values = [float(metadata_timing)] * len(candidate_frames)
        timing_metrics = build_timing_stats(timing_values)
        stability_metrics = compute_temporal_stability(candidate_frames)

        summary: Dict[str, Any] = {
            "model_name": model["metadata"].get("model_name") or model["folder_name"],
            "folder_name": model["folder_name"],
            "reference_model": reference_model["folder_name"],
            "frames_compared": detection_metrics["frames_compared"],
            "tp": detection_metrics["tp"],
            "fp": detection_metrics["fp"],
            "fn": detection_metrics["fn"],
            "precision": detection_metrics["precision"],
            "recall": detection_metrics["recall"],
            "f1": detection_metrics["f1"],
            "mean_bbox_iou": detection_metrics.get("mean_bbox_iou"),
            "mean_mask_iou": detection_metrics.get("mean_mask_iou"),
            "mean_dice": detection_metrics.get("mean_dice"),
            "class_agreement_rate": detection_metrics.get("class_agreement_rate"),
            "mean_abs_count_delta": stability_metrics.get("mean_abs_count_delta"),
            "mean_centroid_jitter": stability_metrics.get("mean_centroid_jitter"),
            "mean_mask_area_delta": stability_metrics.get("mean_mask_area_delta"),
            "empty_frame_fraction": stability_metrics.get("empty_frame_fraction"),
            "mean_inference_ms": timing_metrics.get("mean_inference_ms"),
            "median_inference_ms": timing_metrics.get("median_inference_ms"),
            "p95_inference_ms": timing_metrics.get("p95_inference_ms"),
            "effective_fps": timing_metrics.get("effective_fps"),
            "meets_30hz_mean": timing_metrics.get("meets_30hz_mean"),
            "meets_30hz_p95": timing_metrics.get("meets_30hz_p95"),
            "score": None,
        }
        summary["score"] = score_model(summary)
        results.append(summary)

        for row in frame_rows:
            row_copy = {
                "model_name": summary["model_name"],
                "reference_model": summary["reference_model"],
                **row,
            }
            per_frame_metrics.append(row_copy)

    results.sort(key=lambda x: x["score"], reverse=True)
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    summary_fields = [
        "rank",
        "model_name",
        "folder_name",
        "reference_model",
        "frames_compared",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_bbox_iou",
        "mean_mask_iou",
        "mean_dice",
        "class_agreement_rate",
        "mean_abs_count_delta",
        "mean_centroid_jitter",
        "mean_mask_area_delta",
        "empty_frame_fraction",
        "mean_inference_ms",
        "median_inference_ms",
        "p95_inference_ms",
        "effective_fps",
        "meets_30hz_mean",
        "meets_30hz_p95",
        "score",
    ]

    write_csv(summary_csv_path, results, summary_fields)
    write_csv(per_frame_csv_path, per_frame_metrics, [
        "model_name",
        "reference_model",
        "frame_idx",
        "candidate_count",
        "reference_count",
        "tp",
        "fp",
        "fn",
        "matched_pairs",
        "mean_bbox_iou",
    ])

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump({"reference_model": reference_model["folder_name"], "models": results}, f, indent=2)

    print("\nEvaluation complete")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Per-frame CSV: {per_frame_csv_path}")
    print("\nModel ranking:")
    for row in results:
        print(
            f"{row['rank']:>2}. {row['model_name']} - score={row['score']:.4f}, f1={row['f1']:.4f}, mean_mask_iou={row.get('mean_mask_iou'):.4f}, mean_inference_ms={row.get('mean_inference_ms'):.2f}"
        )


if __name__ == "__main__":
    main()
