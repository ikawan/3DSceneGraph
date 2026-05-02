from pathlib import Path
import argparse

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


# =========================
# Configuration
# =========================
BASE_PATH = "bimacs_rgbd/bimacs_rgbd_data"

DEFAULT_SUBJECT = "subject_2"
DEFAULT_TASK = "task_1_k_cooking"
DEFAULT_TAKE = "take_0"

POSE_MODEL_PATH = "yolo26m-pose.pt"
SEG_MODEL_PATH = "yolo26m-seg.pt"

OUTPUT_DIR = "OutputVids"

IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = "cuda:0"   # use "cpu" if needed

# COCO person class is usually 0
EXCLUDE_SEG_CLASS_IDS = {0}


def get_bimacs_frame_path(subject, task, take):
    """Construct the path to BIMACS RGB frames."""
    return Path(BASE_PATH) / subject / task / take / "rgb"


def get_metadata(frame_path):
    """Load metadata from the BIMACS dataset."""
    metadata_file = frame_path / "metadata.csv"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    metadata = {}
    with open(metadata_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                key, type_val, value = parts
                # Parse value based on type
                if type_val == "unsigned int":
                    metadata[key] = int(value)
                elif type_val == "string":
                    metadata[key] = value
    
    return metadata


def load_frames_from_chunks(frame_path, metadata):
    """Load all frames from chunk directories."""
    frames = []
    frame_count = metadata.get('frameCount', 0)
    frames_per_chunk = metadata.get('framesPerChunk', 100)
    
    chunk_idx = 0
    frame_idx = 0
    
    while frame_idx < frame_count:
        chunk_dir = frame_path / f"chunk_{chunk_idx}"
        
        if not chunk_dir.exists():
            break
        
        # Load frames from this chunk
        for i in range(frames_per_chunk):
            if frame_idx >= frame_count:
                break
            
            frame_file = chunk_dir / f"frame_{i}.png"
            if frame_file.exists():
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    frames.append(frame)
            
            frame_idx += 1
        
        chunk_idx += 1
    
    return frames


def ensure_output_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_allowed_seg_classes(model, excluded_ids):
    """
    Returns all segmentation class IDs except excluded ones.
    This avoids drawing/person-detecting the class we do not want.
    """
    names = model.names if hasattr(model, "names") else {}
    all_class_ids = set(names.keys())

    return sorted(list(all_class_ids - excluded_ids))


def process_frames(frames, pose_model, seg_model, allowed_seg_classes, device, imgsz, conf, iou):
    """Process frames through YOLO Pose, YOLO Segmentation, and MediaPipe Hands."""
    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    processed_frames = []

    for frame_idx, frame in enumerate(frames):
        annotated = frame.copy()

        # =========================
        # MediaPipe Hands
        # =========================
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # =========================
        # YOLO Pose
        # =========================
        pose_results = pose_model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        
        if pose_results:
            pose_overlay = pose_results[0].plot()
            annotated = cv2.addWeighted(annotated, 0.65, pose_overlay, 0.35, 0)

        # =========================
        # YOLO Segmentation
        # person class excluded above
        # =========================
        seg_results = seg_model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            classes=allowed_seg_classes,
            verbose=False,
        )
        
        if seg_results:
            seg_overlay = seg_results[0].plot()
            annotated = cv2.addWeighted(annotated, 0.75, seg_overlay, 0.25, 0)

        # Frame counter
        cv2.putText(
            annotated,
            f"Frame {frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        processed_frames.append(annotated)

        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames")

    hands.close()
    return processed_frames


def main():
    parser = argparse.ArgumentParser(description="Process BIMACS RGBD frames with YOLO and MediaPipe")
    parser.add_argument("--subject", type=str, default=DEFAULT_SUBJECT,
                        help="Subject folder name (default: subject_2)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Task folder name (e.g., task_1_k_cooking)")
    parser.add_argument("--take", type=str, default=DEFAULT_TAKE,
                        help="Take folder name (e.g., take_0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: OutputVids/{task}_{take}.mp4)")
    
    args = parser.parse_args()

    # Validate inputs
    frame_path = get_bimacs_frame_path(args.subject, args.task, args.take)
    
    if not frame_path.exists():
        raise FileNotFoundError(f"Data path not found: {frame_path}")
    
    print(f"Loading frames from: {frame_path}")
    
    # Load metadata
    metadata = get_metadata(frame_path)
    print(f"Metadata: {metadata}")
    
    # Load frames
    print("Loading frames...")
    frames = load_frames_from_chunks(frame_path, metadata)
    print(f"Loaded {len(frames)} frames")
    
    if not frames:
        raise ValueError("No frames loaded")
    
    # Get video properties from first frame
    height, width = frames[0].shape[:2]
    fps = metadata.get('fps', 30)
    
    # Setup output
    if args.output is None:
        output_path = Path(OUTPUT_DIR) / f"{args.task}_{args.take}.mp4"
    else:
        output_path = Path(args.output)
    
    ensure_output_dir(output_path)
    
    # Initialize models
    print("Loading YOLO models...")
    pose_model = YOLO(POSE_MODEL_PATH)
    seg_model = YOLO(SEG_MODEL_PATH)

    allowed_seg_classes = get_allowed_seg_classes(
        seg_model,
        EXCLUDE_SEG_CLASS_IDS,
    )

    print("Segmentation model classes:")
    print(seg_model.names)
    print(f"Excluding segmentation class IDs: {EXCLUDE_SEG_CLASS_IDS}")
    print(f"Running segmentation on class IDs: {allowed_seg_classes}")

    # Create video writer
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Process frames
    print("Processing frames...")
    processed_frames = process_frames(
        frames,
        pose_model,
        seg_model,
        allowed_seg_classes,
        DEVICE,
        IMGSZ,
        CONF,
        IOU
    )

    # Write output video
    print("Writing output video...")
    for frame in processed_frames:
        writer.write(frame)

    writer.release()

    print("Finished processing")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()