from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO


# =========================
# Configuration
# =========================
VIDEO_PATH = "videos/signlang.mp4"

POSE_MODEL_PATH = "yolo26m-pose.pt"
SEG_MODEL_PATH = "yolo26m-seg.pt"

OUTPUT_PATH = "videos/signs.mp4"

IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = "cuda:0"   # use "cpu" if needed

# COCO person class is usually 0
EXCLUDE_SEG_CLASS_IDS = {0}


def ensure_output_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return fps, width, height


def get_allowed_seg_classes(model, excluded_ids):
    """
    Returns all segmentation class IDs except excluded ones.
    This avoids drawing/person-detecting the class we do not want.
    """
    names = model.names if hasattr(model, "names") else {}
    all_class_ids = set(names.keys())

    return sorted(list(all_class_ids - excluded_ids))


def main():
    ensure_output_dir(OUTPUT_PATH)

    fps, width, height = ensure_video_info(VIDEO_PATH)

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

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # =========================
    # YOLO streams
    # =========================
    pose_stream = pose_model.predict(
        source=VIDEO_PATH,
        stream=True,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        verbose=False,
    )

    seg_stream = seg_model.predict(
        source=VIDEO_PATH,
        stream=True,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        classes=allowed_seg_classes,
        verbose=False,
    )

    for frame_idx, (pose_result, seg_result) in enumerate(zip(pose_stream, seg_stream)):
        frame = pose_result.orig_img
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
        pose_overlay = pose_result.plot()
        annotated = cv2.addWeighted(annotated, 0.65, pose_overlay, 0.35, 0)

        # =========================
        # YOLO Segmentation
        # person class excluded above
        # =========================
        seg_overlay = seg_result.plot()
        annotated = cv2.addWeighted(annotated, 0.75, seg_overlay, 0.25, 0)

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

        writer.write(annotated)

        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames")

    writer.release()
    hands.close()

    print("Finished processing")
    print(f"Saved output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()