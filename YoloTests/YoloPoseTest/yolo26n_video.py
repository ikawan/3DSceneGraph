from ultralytics import YOLO
import cv2
from pathlib import Path


def run_pose_on_video(
    video_path: str,
    model_path: str = "yolo26n-pose.pt",
    output_path: str = "pose_output.mp4",
    conf: float = 0.25
):
    """
    Run YOLO pose estimation on a recorded video and save the annotated result.

    Args:
        video_path: Path to input video file.
        model_path: Path to YOLO pose model weights.
        output_path: Path to save annotated video.
        conf: Confidence threshold.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load pose model
    model = YOLO(model_path)

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video. Press 'q' to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose estimation on the current frame
        results = model.predict(
            source=frame,
            conf=conf,
            verbose=False
        )

        # Draw annotations
        annotated_frame = results[0].plot()

        # Save frame to output video
        writer.write(annotated_frame)

        # Show preview window
        cv2.imshow("YOLO Pose Video", annotated_frame)

        # Quit early with q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    run_pose_on_video(
        video_path="videos/BenchmarkVideo.mp4",
        model_path="yolo26n-pose.pt",
        output_path="YoloTests/YoloPoseTest/videos/yolo26_output.mp4",
        conf=0.25
    )