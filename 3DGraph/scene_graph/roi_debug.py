from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import geometry
from .config import DEFAULT_CONFIG
from .io import load_depth_image_m, load_rgb_image, resolve_path
from .perception import (
    check_compute_device,
    compute_person_depth_barrier,
    filter_detection_mask,
    load_yolo_model,
    should_ignore_detection_class,
)


def colorize_depth(depth_m, config=DEFAULT_CONFIG):
    valid = geometry.valid_depth_mask(depth_m, config)
    vis = np.zeros_like(depth_m, dtype=np.uint8)
    if np.any(valid):
        depth_clipped = np.clip(depth_m, config.depth.min_m, config.depth.max_m)
        normalized = cv2.normalize(depth_clipped[valid], None, 0, 255, cv2.NORM_MINMAX)
        vis[valid] = normalized.astype(np.uint8).reshape(-1)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def write_mask(path, mask):
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def clear_roi_debug_outputs(output_dir):
    for pattern in ("*.png", "metadata.json"):
        for path in Path(output_dir).glob(pattern):
            if path.is_file():
                path.unlink()


def safe_name(name):
    return "".join(c if c.isalnum() else "_" for c in str(name).lower()).strip("_")


def overlay_mask(image, mask, color, alpha=0.45):
    out = image.copy()
    color = np.array(color, dtype=np.float32)
    out[mask] = ((1.0 - alpha) * out[mask] + alpha * color).astype(np.uint8)
    return out


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def yolo_detections_from_result(result, image_shape_hw, config=DEFAULT_CONFIG):
    if result.masks is None:
        raise RuntimeError("YOLO produced no segmentation masks for this frame.")

    detections = []
    h, w = image_shape_hw
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    class_names = result.names

    for det_index, mask in enumerate(masks):
        cls_id = int(boxes.cls[det_index].item())
        detections.append({
            "det_index": det_index,
            "class_name": class_names[cls_id],
            "confidence": float(boxes.conf[det_index].item()),
            "bbox_2d": geometry.bbox_to_list(boxes.xyxy[det_index].cpu().numpy()),
            "mask": geometry.resize_mask(mask, (h, w)),
        })
    return detections


def export_roi_debug_artifacts(
    rgb_path=None,
    depth_path=None,
    output_dir="3DGraph/outputs/roi_debug",
    config=DEFAULT_CONFIG,
    model=None,
    clear_existing=True,
    verbose=True,
):
    rgb_path = rgb_path or config.paths.rgb_image_path
    depth_path = depth_path or config.paths.depth_image_path
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        clear_roi_debug_outputs(output_dir)

    check_compute_device(config.runtime.device)
    rgb = load_rgb_image(rgb_path)
    depth_m = load_depth_image_m(depth_path, config)
    h, w = rgb.shape[:2]
    if depth_m.shape[:2] != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

    model = model or load_yolo_model(config)
    result = model.predict(rgb, device=config.runtime.device, verbose=config.runtime.verbose_yolo)[0]
    detections = yolo_detections_from_result(result, rgb.shape[:2], config)
    person_front_depth, roi_depth_limit, roi_mask, roi_filter = compute_person_depth_barrier(
        detections,
        depth_m,
        config,
    )

    person_mask = np.any(
        np.stack([
            det["mask"]
            for det in detections
            if det["class_name"].lower() == config.depth.person_class_name
        ], axis=0),
        axis=0,
    )
    filtered_person_mask = person_mask & roi_mask
    semantic_roi_mask = filtered_person_mask.copy()
    overlay = overlay_mask(rgb.copy(), filtered_person_mask, (255, 0, 0))
    roi_rgb = rgb.copy()
    roi_rgb[~roi_mask] = 0

    cv2.imwrite(str(output_dir / "roi_rgb.png"), roi_rgb)
    write_mask(output_dir / "roi_mask.png", roi_mask)
    write_mask(output_dir / "person_mask.png", filtered_person_mask)
    cv2.imwrite(str(output_dir / "depth_colorized.png"), colorize_depth(depth_m, config))

    kept = []
    skipped = []
    metadata = {
        "rgb_image": str(resolve_path(rgb_path)),
        "depth_image": str(resolve_path(depth_path)),
        "model": str(resolve_path(config.paths.yolo_model_path)),
        "roi_filter": roi_filter,
        "detections": [],
        "skipped_detections": [],
    }

    for det in detections:
        class_name = det["class_name"]
        if class_name.lower() == config.depth.person_class_name:
            continue
        if should_ignore_detection_class(class_name, config):
            skipped.append((det["det_index"], class_name, "ignored by perception config"))
            continue

        clean_mask, reason = filter_detection_mask(det, depth_m, roi_mask, person_front_depth, config)
        if reason is not None:
            skipped.append((det["det_index"], class_name, reason))
            continue
        if int(clean_mask.sum()) < config.perception.min_mask_area_pixels:
            skipped.append((det["det_index"], class_name, "mask area too small"))
            continue

        semantic_roi_mask |= clean_mask
        overlay = overlay_mask(overlay, clean_mask, (0, 0, 255))
        out_name = f"mask_{det['det_index']:02d}_{safe_name(class_name)}.png"
        write_mask(output_dir / out_name, clean_mask)
        median_depth = geometry.median_depth_in_mask(clean_mask, depth_m, config)
        kept.append(det)
        metadata["detections"].append({
            "index": int(det["det_index"]),
            "class_name": class_name,
            "confidence": float(det["confidence"]),
            "mask_file": out_name,
            "mask_pixels": int(clean_mask.sum()),
            "bbox_xyxy": mask_bbox(clean_mask),
            "median_depth_m": median_depth,
        })

    semantic_roi_rgb = rgb.copy()
    semantic_roi_rgb[~semantic_roi_mask] = 0
    write_mask(output_dir / "semantic_roi_mask.png", semantic_roi_mask)
    cv2.imwrite(str(output_dir / "semantic_roi_rgb.png"), semantic_roi_rgb)
    cv2.imwrite(str(output_dir / "overlay_kept_objects.png"), overlay)

    metadata["kept_detection_count"] = len(kept)
    metadata["skipped_detection_count"] = len(skipped)
    metadata["skipped_detections"] = [
        {"index": int(index), "class_name": class_name, "reason": reason}
        for index, class_name, reason in skipped
    ]
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Estimated person front depth: {person_front_depth:.3f} m")
        print(f"ROI depth limit: {roi_depth_limit:.3f} m")
        print(f"Kept {len(kept)} non-person detections inside the depth ROI.")
        print(f"Skipped {len(skipped)} detections.")
        print(f"Saved ROI debug outputs to: {output_dir}")

    return metadata, output_dir
