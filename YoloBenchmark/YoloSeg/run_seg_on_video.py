"""
YOLO Segmentation Model Inference Script with Video Output

This script runs a specified YOLO segmentation model on a reference video and saves
the output video with segmentation predictions visualized.

USAGE:
    python run_seg_on_video.py
    
    Then modify the configuration variables at the top to change:
    - INPUT_VIDEO: Path to the input video
    - MODEL_PATH: Path to the YOLO segmentation model
    - OUTPUT_VIDEO: Name for the output video (saved in OutputVids/)
"""

import cv2
import statistics
from pathlib import Path
from ultralytics import YOLO


# =========================
# Configuration
# =========================

# Path to input video (relative or absolute)
# Examples: "videos/BenchmarkVideo.mp4", "../../videos/sample.mp4"
INPUT_VIDEO = "videos/BenchmarkVideo.mp4"

# Path to YOLO segmentation model
# Available models in root: yolo26n-seg.pt, yolo26s-seg.pt, yolo26m-seg.pt, yolo26l-seg.pt, yolo26x-seg.pt
MODEL_PATH = "yolo26m-seg.pt"

# Output video filename (will be saved in OutputVids/)
OUTPUT_VIDEO_NAME = "yolo26m.mp4"

# Model parameters
IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = "cuda:0"  # use "cpu" if needed

# Video parameters
TARGET_FPS = 30.0


# =========================
# Helper Functions
# =========================

def get_video_info(video_path):
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": frame_width,
        "height": frame_height,
    }


def main():
    """Main inference function."""
    
    # Resolve paths
    model_path = Path(MODEL_PATH)
    input_video = Path(INPUT_VIDEO)
    output_dir = Path(__file__).parent / "OutputVids"
    output_video = output_dir / OUTPUT_VIDEO_NAME
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check input video exists
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return
    
    # Check model exists
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"Reading video: {input_video}")
    video_info = get_video_info(str(input_video))
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    frame_count = video_info["frame_count"]
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {frame_count}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    if not out_writer.isOpened():
        print("Error: Could not open video writer")
        return
    
    # Open input video
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        print("Error: Could not open input video")
        return
    
    print(f"\nProcessing video...")
    inference_times_ms = []
    frames_processed = 0
    
    try:
        # Run inference with streaming
        results = model.predict(
            source=str(input_video),
            stream=True,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False,
        )
        
        for frame_idx, result in enumerate(results):
            frames_processed += 1
            
            # Get inference time
            speed = getattr(result, "speed", {}) or {}
            inference_ms = float(speed.get("inference", 0.0))
            inference_times_ms.append(inference_ms)
            
            # Get frame from result
            frame = result.orig_img
            
            # Plot results on frame (YOLO provides this method)
            annotated_frame = result.plot()
            
            # Write frame to output video
            out_writer.write(annotated_frame)
            
            # Progress
            if (frame_idx + 1) % 30 == 0:
                print(f"  Processed {frame_idx + 1}/{frame_count} frames", end="\r")
        
        print(f"  Processed {frames_processed}/{frame_count} frames")
        
    finally:
        out_writer.release()
        cap.release()
    
    # Calculate statistics
    if inference_times_ms:
        mean_ms = statistics.mean(inference_times_ms)
        median_ms = statistics.median(inference_times_ms)
        effective_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    else:
        mean_ms = median_ms = effective_fps = 0.0
    
    print(f"\n✓ Output saved to: {output_video}")
    print(f"\nStatistics:")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Mean inference time: {mean_ms:.2f} ms")
    print(f"  Median inference time: {median_ms:.2f} ms")
    print(f"  Effective FPS: {effective_fps:.2f}")
    print(f"  Meets 30 Hz target: {'Yes' if mean_ms <= 33.33 else 'No'}")


if __name__ == "__main__":
    main()
