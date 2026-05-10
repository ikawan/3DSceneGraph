from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_CONFIG
from .io import resolve_path


def export_d435_color_intrinsics(
    output_path=None,
    width=640,
    height=480,
    fps=30,
):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required to export RealSense intrinsics. "
            "Install the RealSense Python package and connect the D435 camera."
        ) from exc

    output_path = resolve_path(output_path or DEFAULT_CONFIG.paths.camera_intrinsics_path)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        data = {
            "model": "realsense_d435",
            "stream": "color",
            "aligned_depth_to": "color",
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "distortion_model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }
    finally:
        pipeline.stop()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data, output_path
