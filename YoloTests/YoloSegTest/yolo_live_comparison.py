import time
import cv2
import numpy as np
from ultralytics import YOLO


def resize_to_same_height(img1, img2, target_height=720):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    scale1 = target_height / h1
    scale2 = target_height / h2

    new_w1 = int(w1 * scale1)
    new_w2 = int(w2 * scale2)

    img1_resized = cv2.resize(img1, (new_w1, target_height))
    img2_resized = cv2.resize(img2, (new_w2, target_height))

    return img1_resized, img2_resized


def process_with_model(frame, model, model_name, conf=0.25, imgsz=640):
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
    annotated = result.plot()

    inference_ms = (end - start) * 1000
    fps = 1.0 / (end - start) if (end - start) > 0 else 0.0

    cv2.putText(
        annotated,
        f"{model_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Inference: {inference_ms:.1f} ms",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(
        annotated,
        f"Time: {timestamp}",
        (10, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return annotated


def main():
    model1 = YOLO("yolo26n.pt")
    model2 = YOLO("yolo26n-seg.pt")   # change to your second model

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        annotated1 = process_with_model(frame, model1, "YOLO26n")
        annotated2 = process_with_model(frame, model2, "YOLO26n-seg")

        annotated1, annotated2 = resize_to_same_height(annotated1, annotated2, target_height=600)
        combined = np.hstack((annotated1, annotated2))

        cv2.imshow("Model Comparison", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()