from pathlib import Path
import argparse
import time

import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO


# =========================
# Configuration
# =========================
BASE_PATH = "bimacs_rgbd/bimacs_rgbd_data"
DEFAULT_OUTPUT = "OutputVids/all_tasks.mp4"
MAX_TAKES_PER_TASK = 4

SEG_MODEL_PATH = "yolo26m-seg.pt"

IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = "cuda:0"

# COCO classes excluded from segmentation overlay
EXCLUDE_SEG_CLASS_NAMES = {"person", "tv", "mouse", "chair", "keyboard"}


def get_bimacs_frame_path(subject, task, take):
    """Construct the path to BIMACS RGB frames."""
    return Path(BASE_PATH) / subject / task / take / "rgb"


def find_task_take_paths(base_path, max_takes_per_task=MAX_TAKES_PER_TASK):
    """Discover the first N takes for every task across all subjects."""
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    task_take_paths = []
    for subject_dir in sorted(base.iterdir()):
        if not subject_dir.is_dir():
            continue

        for task_dir in sorted(subject_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            takes = sorted([take_dir.name for take_dir in task_dir.iterdir() if take_dir.is_dir()])
            for take in takes[:max_takes_per_task]:
                task_take_paths.append((subject_dir.name, task_dir.name, take))

    return task_take_paths


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


def check_gpu_available(device):
    if "cuda" not in device.lower():
        raise ValueError("Configured device is not GPU. Set DEVICE to a CUDA device such as 'cuda:0'.")
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. GPU execution is required.")

    print(f"Using GPU device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"Primary GPU: {torch.cuda.get_device_name(0)}")


def create_hand_processor():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def annotate_frame(frame, seg_model, allowed_seg_classes, device, imgsz, conf, iou, hands, label_text=None):
    """Annotate a single frame with segmentation and MediaPipe hand landmarks."""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    annotated = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)

    if hand_results and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

    # YOLO Segmentation
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

    label = label_text or ""
    cv2.putText(
        annotated,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),  # red in BGR
        2,
        cv2.LINE_AA,
    )

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Process BIMACS RGBD frames with YOLO segmentation and MediaPipe")
    parser.add_argument("--base-path", type=str, default=BASE_PATH,
                        help="Root BIMACS path containing subject/task/take folders")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output video path")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Computation device (must be a CUDA device)")
    parser.add_argument("--max-takes", type=int, default=MAX_TAKES_PER_TASK,
                        help="Maximum number of takes to process per task")

    args = parser.parse_args()

    check_gpu_available(args.device)

    task_take_paths = find_task_take_paths(args.base_path, args.max_takes)
    if not task_take_paths:
        raise ValueError(f"No task/take folders found under {args.base_path}")

    print(f"Found {len(task_take_paths)} takes across subjects/tasks")

    # Initialize segmentation model
    print("Loading YOLO segmentation model...")
    seg_model = YOLO(SEG_MODEL_PATH)

    excluded_seg_class_ids = {
        class_id
        for class_id, class_name in seg_model.names.items()
        if class_name in EXCLUDE_SEG_CLASS_NAMES
    }

    allowed_seg_classes = get_allowed_seg_classes(seg_model, excluded_seg_class_ids)

    print("Segmentation model classes:")
    print(seg_model.names)
    print(f"Excluding segmentation classes: {EXCLUDE_SEG_CLASS_NAMES}")
    print(f"Excluding segmentation class IDs: {excluded_seg_class_ids}")
    print(f"Running segmentation on class IDs: {allowed_seg_classes}")

    # Determine video properties from the first valid frame
    first_frame = None
    first_metadata = None
    for subject, task, take in task_take_paths:
        frame_path = get_bimacs_frame_path(subject, task, take)
        if frame_path.exists():
            metadata = get_metadata(frame_path)
            frames = load_frames_from_chunks(frame_path, metadata)
            if frames:
                first_frame = frames[0]
                first_metadata = metadata
                break

    if first_frame is None:
        raise ValueError("No frames found in the selected takes")

    height, width = first_frame.shape[:2]
    fps = first_metadata.get('fps', 30)

    output_path = Path(args.output)
    ensure_output_dir(output_path)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    total_frames = 0
    total_process_time = 0.0

    print("Processing selected takes...")
    with create_hand_processor() as hands:
        for subject, task, take in task_take_paths:
            frame_path = get_bimacs_frame_path(subject, task, take)
            if not frame_path.exists():
                print(f"Skipping missing frame path: {frame_path}")
                continue

            metadata = get_metadata(frame_path)
            frames = load_frames_from_chunks(frame_path, metadata)
            if not frames:
                print(f"Skipping empty take: {subject}/{task}/{take}")
                continue

            print(f"Processing {len(frames)} frames from {subject}/{task}/{take}")
            for frame_idx, frame in enumerate(frames):
                start_time = time.time()
                label_text = f"{subject}/{task}/{take} | Frame {frame_idx}"
                annotated = annotate_frame(
                    frame,
                    seg_model,
                    allowed_seg_classes,
                    args.device,
                    IMGSZ,
                    CONF,
                    IOU,
                    hands,
                    label_text=label_text,
                )
                elapsed = time.time() - start_time
                total_process_time += elapsed
                total_frames += 1
                writer.write(annotated)

                if total_frames % 30 == 0:
                    print(f"Written {total_frames} frames so far")

    writer.release()

    if total_frames == 0:
        raise ValueError("No frames were processed")

    average_time = total_process_time / total_frames
    print("Finished processing")
    print(f"Saved output to: {output_path}")
    print(f"Total frames processed: {total_frames}")
    print(f"Average process time per frame: {average_time:.4f} seconds")


if __name__ == "__main__":
    main()
