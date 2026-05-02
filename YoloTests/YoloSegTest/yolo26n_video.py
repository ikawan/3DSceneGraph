import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def segment_video(
    video_path,
    model_path="yolo26n-seg.pt",
    output_path="segmented_output.mp4",
    conf=0.25,
    imgsz=640,
    display=True
):
    """
    Run YOLO instance segmentation on a video file.

    Args:
        video_path: Path to input video.
        model_path: YOLO segmentation model path.
        output_path: Path to save annotated video.
        conf: Confidence threshold.
        imgsz: Inference image size.
        display: Whether to show frames during processing.
    """

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if input_fps <= 0:
        input_fps = 30.0

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        input_fps,
        (frame_width, frame_height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print("Press 'q' to stop early.")

    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index += 1

        start = time.perf_counter()

        results = model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            save=False,
            verbose=False
        )

        end = time.perf_counter()

        result = results[0]
        annotated_frame = result.plot()

        inference_ms = (end - start) * 1000
        fps = 1.0 / (end - start) if (end - start) > 0 else 0.0
        timestamp = time.strftime("%H:%M:%S")

        # Overlay timing info
        cv2.putText(
            annotated_frame,
            f"Inference: {inference_ms:.1f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Time: {timestamp}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_index}/{total_frames}",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        writer.write(annotated_frame)

        if display:
            cv2.imshow("YOLO Video Segmentation", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early by user.")
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    segment_video(
        video_path="videos/BenchmarkVideo.mp4",      # change this
        model_path="yolo26n-seg.pt",       # or yolo11s-seg.pt
        output_path="Yolo/YoloTests/YoloSegTest/VidResult/yolo26n_output.mp4",
        conf=0.25,
        imgsz=640,
        display=True
    )