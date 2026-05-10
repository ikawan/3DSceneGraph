from __future__ import annotations

import cv2
import numpy as np


def valid_depth_values(depth_m, config):
    values = depth_m[np.isfinite(depth_m)]
    values = values[(values > config.depth.min_m) & (values < config.depth.max_m)]
    return values


def valid_depth_mask(depth_m, config):
    return np.isfinite(depth_m) & (depth_m > config.depth.min_m) & (depth_m < config.depth.max_m)


def resize_mask(mask, shape_hw):
    h, w = shape_hw
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def bbox_to_list(box_xyxy):
    return [float(v) for v in box_xyxy]


def bbox_from_mask(mask, fallback_bbox):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return fallback_bbox
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def mask_center(mask, fallback_bbox):
    ys, xs = np.where(mask)
    if xs.size > 0:
        return [float(xs.mean()), float(ys.mean())]

    x1, y1, x2, y2 = fallback_bbox
    return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]


def median_depth_in_mask(mask, depth_m, config):
    values = depth_m[mask]
    values = values[np.isfinite(values)]
    values = values[(values > config.depth.min_m) & (values < config.depth.max_m)]
    if values.size == 0:
        return None
    return float(np.median(values))


def median_depth_near_pixel(center_2d, depth_m, radius_px, config):
    h, w = depth_m.shape[:2]
    cx, cy = center_2d
    x = int(round(cx))
    y = int(round(cy))

    x1 = max(0, x - radius_px)
    x2 = min(w, x + radius_px + 1)
    y1 = max(0, y - radius_px)
    y2 = min(h, y + radius_px + 1)

    values = depth_m[y1:y2, x1:x2].reshape(-1)
    values = values[np.isfinite(values)]
    values = values[(values > config.depth.min_m) & (values < config.depth.max_m)]
    if values.size == 0:
        return None

    return float(np.median(values))


def project_pixel_to_3d(center_2d, depth_m_value, intrinsics):
    u, v = center_2d
    z = float(depth_m_value)
    x = (float(u) - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (float(v) - intrinsics["cy"]) * z / intrinsics["fy"]
    return [float(x), float(y), z]


def mask_to_3d_points(mask, depth_m, intrinsics, config):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None

    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > config.depth.min_m) & (z < config.depth.max_m)
    if not np.any(valid):
        return None

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (xs - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (ys - intrinsics["cy"]) * z / intrinsics["fy"]
    return np.column_stack((x, y, z))


def summarize_3d_points(points_3d, config):
    center = np.median(points_3d, axis=0)
    bbox_min = np.percentile(points_3d, config.perception.bbox_3d_percentile_low, axis=0)
    bbox_max = np.percentile(points_3d, config.perception.bbox_3d_percentile_high, axis=0)
    size = bbox_max - bbox_min

    return {
        "center_3d_m": [float(v) for v in center],
        "bbox_3d_m": {
            "min_xyz": [float(v) for v in bbox_min],
            "max_xyz": [float(v) for v in bbox_max],
            "size_xyz": [float(v) for v in size],
            "coordinate_frame": "camera_xyz_m",
            "percentile_range": [
                config.perception.bbox_3d_percentile_low,
                config.perception.bbox_3d_percentile_high,
            ],
        },
    }


def clean_mask_by_depth(mask, depth_m, config):
    values = depth_m[mask]
    values = values[np.isfinite(values)]
    values = values[(values > config.depth.min_m) & (values < config.depth.max_m)]
    if values.size == 0:
        return mask

    median_depth = np.median(values)
    return (
        mask
        & valid_depth_mask(depth_m, config)
        & (depth_m >= median_depth - config.depth.object_depth_tolerance_m)
        & (depth_m <= median_depth + config.depth.object_depth_tolerance_m)
    )


def distance_3d(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    return float(np.linalg.norm(a - b))


def bbox_2d_iou(a, b):
    if not a or not b:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return None
    return float(inter / union)


def bbox_2d_overlap_ratio(source_bbox, target_bbox):
    if not source_bbox or not target_bbox:
        return None
    sx1, sy1, sx2, sy2 = source_bbox
    tx1, ty1, tx2, ty2 = target_bbox
    ix1, iy1 = max(sx1, tx1), max(sy1, ty1)
    ix2, iy2 = min(sx2, tx2), min(sy2, ty2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    source_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
    if source_area <= 0:
        return None
    return float(inter / source_area)


def bbox_3d_distance(a, b):
    if not a or not b:
        return None
    a_min = np.asarray(a.get("min_xyz"), dtype=np.float64)
    a_max = np.asarray(a.get("max_xyz"), dtype=np.float64)
    b_min = np.asarray(b.get("min_xyz"), dtype=np.float64)
    b_max = np.asarray(b.get("max_xyz"), dtype=np.float64)
    if any(arr.shape != (3,) or not np.all(np.isfinite(arr)) for arr in (a_min, a_max, b_min, b_max)):
        return None

    separation = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(separation))


def horizontal_distance_xz(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        return None
    return float(np.linalg.norm([a[0] - b[0], a[2] - b[2]]))
