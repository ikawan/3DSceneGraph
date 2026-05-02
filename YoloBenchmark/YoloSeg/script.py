import json
import statistics
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# Configuration
# =========================
VIDEO_PATH = "videos/BenchmarkVideo.mp4"

# Change this manually before each run
MODEL_PATH = "yolo26x-seg.pt"

OUTPUT_ROOT = "YoloBenchmark/YoloSeg/benchmarks"

IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = "cuda:0"   # use "cpu" if needed
TARGET_FPS = 30.0

# Keep every Nth polygon point to reduce file size.
# Use 1 to keep all points.
POLYGON_STRIDE = 2

# Round float values in output JSON
ROUND_DIGITS = 4


def round_float(x, digits=ROUND_DIGITS):
    return round(float(x), digits)


def polygon_area(points_xy):
    """
    Compute polygon area using the shoelace formula.
    """
    if len(points_xy) < 3:
        return 0.0
    pts = np.asarray(points_xy, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def bbox_area_xyxy(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def contour_centroid(points_xy):
    """
    Compute centroid from contour moments.
    Falls back to mean of contour points if needed.
    """
    if len(points_xy) == 0:
        return [None, None]

    cnt = np.asarray(points_xy, dtype=np.int32).reshape((-1, 1, 2))
    m = cv2.moments(cnt)

    if abs(m["m00"]) > 1e-8:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        return [round_float(cx), round_float(cy)]

    pts = np.asarray(points_xy, dtype=np.float32)
    return [round_float(np.mean(pts[:, 0])), round_float(np.mean(pts[:, 1]))]


def simplify_polygon(points_xy, stride=1):
    if stride <= 1 or len(points_xy) <= 3:
        return points_xy
    simplified = points_xy[::stride]
    if len(simplified) < 3:
        return points_xy[:3]
    return simplified


def ensure_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def main():
    model_name = Path(MODEL_PATH).stem
    out_dir = Path(OUTPUT_ROOT) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.jsonl"
    metadata_path = out_dir / "metadata.json"

    video_info = ensure_video_info(VIDEO_PATH)
    video_fps = video_info["fps"]
    frame_width = video_info["width"]
    frame_height = video_info["height"]

    model = YOLO(MODEL_PATH)
    class_names = model.names if hasattr(model, "names") else {}

    inference_times_ms = []
    frames_processed = 0

    results_generator = model.predict(
        source=VIDEO_PATH,
        stream=True,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        verbose=False,
    )

    with predictions_path.open("w", encoding="utf-8") as f_out:
        for frame_idx, result in enumerate(results_generator):
            frames_processed += 1

            speed = getattr(result, "speed", {}) or {}
            inference_ms = float(speed.get("inference", 0.0))
            inference_times_ms.append(inference_ms)

            timestamp_sec = frame_idx / video_fps if video_fps > 0 else None

            frame_record = {
                "frame_idx": frame_idx,
                "timestamp_sec": round_float(timestamp_sec) if timestamp_sec is not None else None,
                "frame_size": [frame_width, frame_height],
                "inference_ms": round_float(inference_ms),
                "preprocess_ms": round_float(speed.get("preprocess", 0.0)),
                "postprocess_ms": round_float(speed.get("postprocess", 0.0)),
                "num_instances": 0,
                "instances": [],
            }

            boxes = result.boxes
            masks = result.masks

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                # List of contours in pixel coordinates
                mask_polygons = masks.xy if masks is not None else [None] * len(xyxy)

                frame_record["num_instances"] = len(xyxy)

                for inst_idx in range(len(xyxy)):
                    bbox = xyxy[inst_idx].tolist()
                    class_id = int(cls[inst_idx])
                    confidence = float(confs[inst_idx])
                    class_name = class_names.get(class_id, str(class_id))

                    polygon = []
                    mask_area_px = None
                    centroid_xy = [None, None]

                    if mask_polygons is not None and inst_idx < len(mask_polygons):
                        poly = np.asarray(mask_polygons[inst_idx], dtype=np.float32)
                        if poly.ndim == 2 and poly.shape[1] == 2 and len(poly) >= 3:
                            poly_list = [[int(p[0]), int(p[1])] for p in poly]
                            poly_list = simplify_polygon(poly_list, stride=POLYGON_STRIDE)
                            polygon = poly_list
                            mask_area_px = round_float(polygon_area(poly_list))
                            centroid_xy = contour_centroid(poly_list)

                    instance_record = {
                        "instance_idx": inst_idx,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round_float(confidence),
                        "bbox_xyxy": [round_float(v) for v in bbox],
                        "bbox_area_px": round_float(bbox_area_xyxy(bbox)),
                        "centroid_xy": centroid_xy,
                        "mask_area_px": mask_area_px,
                        "polygon_xy": polygon,
                    }

                    frame_record["instances"].append(instance_record)

            f_out.write(json.dumps(frame_record, ensure_ascii=False) + "\n")

    if inference_times_ms:
        mean_ms = statistics.mean(inference_times_ms)
        median_ms = statistics.median(inference_times_ms)
        p95_ms = float(np.percentile(np.array(inference_times_ms, dtype=np.float32), 95))
        effective_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    else:
        mean_ms = median_ms = p95_ms = effective_fps = 0.0

    metadata = {
        "model_path": MODEL_PATH,
        "model_name": model_name,
        "video_path": VIDEO_PATH,
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "device": DEVICE,
        "video_info": video_info,
        "target_fps": TARGET_FPS,
        "target_ms_per_frame": round_float(1000.0 / TARGET_FPS),
        "num_frames_processed": frames_processed,
        "mean_inference_ms": round_float(mean_ms),
        "median_inference_ms": round_float(median_ms),
        "p95_inference_ms": round_float(p95_ms),
        "effective_fps": round_float(effective_fps),
        "meets_30hz_mean_budget": mean_ms <= (1000.0 / TARGET_FPS),
    }

    with metadata_path.open("w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2, ensure_ascii=False)

    print("\nFinished run")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()