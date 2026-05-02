# YOLO Pose Model Evaluation Guide

## Overview

The `evaluate_pose_models.py` script compares YOLO pose model outputs that have been pre-generated and saved to disk. It uses the **heaviest/slowest model as a pseudo-ground-truth baseline** and evaluates all other models against it.

**Key Point**: This is *not* human-labeled ground truth. The reference model serves only as a proxy baseline for comparing relative model performance.

---

## Quick Start

### Running the Script

```bash
python evaluate_pose_models.py
```

The script will:
1. Load all model folders from `benchmarks/`
2. Compare each candidate model to the reference model (configurable)
3. Compute metrics across detection, bounding boxes, keypoints, temporal stability, and timing
4. Generate three output files (CSV and JSON reports)
5. Print a ranking table to the console

### Output Files Generated

1. **`pose_evaluation_summary.csv`** - Per-model summary with all metrics, sorted by final score
2. **`pose_evaluation_summary.json`** - Detailed JSON report with hierarchical structure
3. **`pose_per_keypoint_metrics.csv`** - Per-keypoint breakdown of mean distances

---

## Configuration Variables

Edit the top of `evaluate_pose_models.py` to customize behavior:

```python
# Path to benchmark results
BENCHMARK_BASE_DIR = Path(__file__).parent / "benchmarks"

# Reference model (pseudo-ground-truth)
REFERENCE_MODEL_NAME = "yolo26l-pose"

# Detection matching threshold (IoU)
MATCH_IOU_THRESHOLD = 0.5

# Temporal frame-to-frame matching threshold
TEMPORAL_MATCH_IOU_THRESHOLD = 0.5

# Keypoint confidence filter (ignore very low-confidence keypoints)
KEYPOINT_CONF_THRESHOLD = 0.1

# Keypoint distance normalization: "bbox_diagonal" or "bbox_sqrt_area"
KEYPOINT_DISTANCE_NORMALIZATION = "bbox_diagonal"

# Ranking score weights (must sum to ~1.0)
SCORE_WEIGHTS = {
    "f1": 0.30,                      # Detection agreement
    "keypoint_accuracy": 0.25,       # Keypoint distance accuracy
    "bbox_iou": 0.15,                # Bounding box overlap
    "speed": 0.15,                   # Inference speed (30 Hz target)
    "stability": 0.15,               # Temporal stability
}
```

---

## Metrics Explained

### A. Detection Metrics
- **TP** (True Positives): Matched detections
- **FP** (False Positives): Candidate detections with no match
- **FN** (False Negatives): Reference detections with no match
- **Precision**: TP / (TP + FP) — how many detected instances are correct
- **Recall**: TP / (TP + FN) — how many ground-truth instances were found
- **F1**: Harmonic mean of precision and recall

**Matching Strategy**: Greedily match by highest IoU, one-to-one per frame, threshold ≥ 0.5 IoU

### B. Bounding Box Metrics
- **Mean/Median/P95 IoU**: Intersection-over-Union statistics across all matched pairs
- **Higher is better** (1.0 = perfect overlap)

### C. Keypoint Metrics
- **Mean/Median Keypoint Distance (pixels)**: Euclidean distance between corresponding keypoints
- **Mean Normalized Distance**: Distance normalized by reference bbox diagonal or sqrt(area)
- **Comparable Keypoint Fraction**: Percentage of keypoints that could be compared
- **Avg Detected/Visible Keypoints**: Average keypoint counts per matched instance
- **Mean Keypoint Confidence**: Average keypoint confidence from reference model

**Keypoint Comparison**: Only compares keypoints present in both models, confidence ≥ threshold

### D. Temporal Stability Metrics
- **Mean Abs Count Delta**: Average frame-to-frame change in detection count
- **Mean Centroid Jitter**: Average pixel displacement of bounding box centers between frames
- **Mean Area Delta**: Average change in bounding box area between frames
- **Mean Keypoint Jitter**: Average keypoint displacement between frames
- **Empty Frame Fraction**: Percentage of frames with zero detections

**How it works**: Matches instances between frame t and t+1 using IoU, computes consistency metrics

### E. Timing Metrics
- **Mean/Median/P95 Inference Time**: Processing time per frame (milliseconds)
- **Effective FPS**: Computed from mean inference time
- **Meets 30 Hz Mean/P95**: Whether the model meets the 33.33 ms target budget

**Target**: ~33.33 ms per frame (30 Hz real-time video processing)

### F. Final Score
Weighted combination of sub-scores (configurable):
- Higher F1 = higher final score
- Faster inference (closer to 30 Hz) = higher speed score
- Lower keypoint distance = higher keypoint accuracy score
- Lower temporal jitter = higher stability score
- Lower bbox IoU mismatch = higher bbox score

**Formula**: Normalizes each component to [0, 1] and applies weights

---

## Understanding the Results

### Example Output

```
MODEL RANKING (by Final Score)
====================================================================================
Rank   Model Name         F1       Keypoint Dist   Speed      Score     
---
1      yolo26x-pose       0.9968   11.92 px        14.50 ms   0.9665
2      yolo26m-pose       0.9954   13.65 px        10.34 ms   0.9610
3      yolo26s-pose       0.9968   14.77 px        11.86 ms   0.9574
4      yolo26n-pose       0.9304   23.60 px        10.15 ms   0.9112
```

**Interpretation**:
- **yolo26x-pose** (rank 1): Best overall. Excellent detection (F1=0.997), very accurate keypoints (11.92 px), within 30 Hz budget (14.5 ms)
- **yolo26m-pose** (rank 2): Slightly smaller model, fastest inference (10.3 ms), still high quality
- **yolo26s-pose** (rank 3): Good detection but slightly worse keypoint accuracy
- **yolo26n-pose** (rank 4): Smallest/fastest but significantly lower detection quality (F1=0.93, 207 false positives)

**Trade-off**: Smaller models are faster but lose accuracy. Choose based on your latency vs. accuracy requirements.

---

## Advanced Usage

### Changing the Reference Model

If you want a different reference model:

```python
REFERENCE_MODEL_NAME = "yolo26m-pose"  # Use smaller model as reference
```

### Adjusting Matching Thresholds

For stricter matching (fewer false positives accepted):
```python
MATCH_IOU_THRESHOLD = 0.7  # Higher threshold = stricter matching
```

### Tweaking the Ranking Weights

To prioritize speed over accuracy:
```python
SCORE_WEIGHTS = {
    "f1": 0.20,
    "keypoint_accuracy": 0.15,
    "bbox_iou": 0.10,
    "speed": 0.35,           # Increased
    "stability": 0.20,
}
```

### Using Different Keypoint Normalization

To normalize by bounding box area instead of diagonal:
```python
KEYPOINT_DISTANCE_NORMALIZATION = "bbox_sqrt_area"
```

---

## Troubleshooting

### Missing Models
If a model folder is missing `metadata.json` or `predictions.jsonl`, the script warns and skips it.

### No Overlapping Frames
If candidate and reference models were run on different video segments, they won't have matching frames. Check that all models were evaluated on the same input video.

### Extreme Keypoint Distances
If some keypoint distances are very large, it may indicate:
- Matching threshold is too low (matches wrong instances)
- Keypoint confidence filtering is too aggressive
- Model output formats differ unexpectedly

Check the per-keypoint CSV for outliers and adjust `MATCH_IOU_THRESHOLD` or `KEYPOINT_CONF_THRESHOLD`.

---

## Data Format Reference

### Input: Folder Structure
```
benchmarks/
├── yolo26l-pose/
│   ├── metadata.json
│   └── predictions.jsonl
├── yolo26m-pose/
│   ├── metadata.json
│   └── predictions.jsonl
... (more models)
```

### Input: metadata.json
```json
{
  "model_name": "yolo26l-pose",
  "mean_inference_ms": 13.98,
  "median_inference_ms": 13.64,
  "p95_inference_ms": 15.96,
  "effective_fps": 71.51,
  "num_frames_processed": 1417
}
```

### Input: predictions.jsonl (one JSON object per line)
```json
{
  "frame_idx": 0,
  "timestamp_sec": 0.0,
  "inference_ms": 52.1,
  "num_instances": 1,
  "instances": [{
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.958,
    "bbox_xyxy": [542.9, 4.0, 1366.3, 985.1],
    "bbox_centroid_xy": [954.6, 494.5],
    "num_keypoints": 17,
    "num_visible_keypoints": 17,
    "mean_keypoint_confidence": 0.481,
    "keypoints": [
      {"id": 0, "x": 939.65, "y": 26.77, "conf": 0.0138},
      {"id": 1, "x": 970.13, "y": 0.0, "conf": 0.0019},
      ... (more keypoints)
    ]
  }]
}
```

### Output: pose_evaluation_summary.csv
Per-model metrics in CSV format, one row per model, sorted by final_score descending.

### Output: pose_evaluation_summary.json
Same metrics in structured JSON, with hierarchical organization by metric category.

### Output: pose_per_keypoint_metrics.csv
Metrics broken down by keypoint ID:
- `keypoint_id`: COCO keypoint ID (0-16 for pose, varies for other tasks)
- `model_name`: Which model
- `mean_distance_px`: Avg Euclidean distance in pixels
- `median_distance_px`: Median distance
- `mean_normalized_distance`: Normalized by bbox size
- `num_compared`: How many keypoint comparisons contributed

---

## For Production Use

- **Real-time processing**: Choose a model that consistently meets 30 Hz (meets_30hz_mean = True)
- **High accuracy**: Prioritize models with F1 > 0.99 and low keypoint distance
- **Balanced**: Start with yolo26m-pose (good speed/accuracy trade-off)
- **Edge devices**: Use yolo26n-pose or smaller, but verify F1 is acceptable for your use case

---

## Questions or Issues?

- Check the console output for any warning messages
- Verify model folders and file paths are correct
- Review the JSON report for detailed per-model breakdown
- Adjust configuration variables and re-run to test different settings
