from ultralytics import YOLO
import cv2


def run_pose_on_webcam(
    model_path: str = "yolo26n-pose.pt",
    camera_index: int = 0,
    conf: float = 0.25
):
    """
    Run YOLO pose estimation on a live webcam feed.

    Args:
        model_path: Path to YOLO pose model weights.
        camera_index: Webcam index, usually 0 for the default camera.
        conf: Confidence threshold.
    """
    # Load pose model
    model = YOLO(model_path)

    # Open webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {camera_index}")

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        # Run inference on the current frame
        results = model.predict(
            source=frame,
            conf=conf,
            verbose=False
        )

        # Get first result and draw annotations
        result = results[0]
        annotated_frame = result.plot()

        # Show the annotated live feed
        cv2.imshow("YOLO Pose Webcam", annotated_frame)

        # Optional: print detected keypoints for the first person
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            first_person = result.keypoints.xy[0]
            print("First person keypoints:", first_person.tolist())

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pose_on_webcam(
        model_path="yolo26n-pose.pt",
        camera_index=0,
        conf=0.25
    )