import time
import cv2
from ultralytics import YOLO


def live_segmentation(
    model_path="yolo11n-seg.pt",
    camera_index=0,
    conf=0.25,
    imgsz=640
):
    """
    Run live webcam instance segmentation with YOLO segmentation model.

    Press 'q' to quit.
    """

    # Load model once
    model = YOLO(model_path)

    # Open webcam
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {camera_index}")

    print("Webcam opened successfully.")
    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read frame from webcam.")
            break

        # Start timing
        start_time = time.perf_counter()

        # Run segmentation on current frame
        results = model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            save=False,
            verbose=False
        )

        # End timing
        end_time = time.perf_counter()

        inference_time = end_time - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0.0
        inference_ms = inference_time * 1000

        # Get first result
        result = results[0]

        # Plot annotated frame
        # This includes boxes, class names, masks, and confidence scores
        annotated_frame = result.plot()

        # Add timing text
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

        # Optional: show current clock time on frame
        current_time_text = time.strftime("%H:%M:%S")
        cv2.putText(
            annotated_frame,
            f"Time: {current_time_text}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Display result
        cv2.imshow("YOLO11 Live Detection", annotated_frame)

        # Quit on q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_segmentation(
        model_path="yolo11n.pt",   # or "yolo11s-seg.pt"
        camera_index=0,                # usually 0 for default webcam
        conf=0.25,
        imgsz=640
    )