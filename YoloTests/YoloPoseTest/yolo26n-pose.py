from ultralytics import YOLO
import cv2
from pathlib import Path


def run_pose_on_image(
    image_path: str,
    model_path: str = "yolo26n-pose.pt",
    output_path: str = "pose_result.jpg",
    conf: float = 0.25
):
    """
    Run pose estimation on a single image and save the annotated result.

    Args:
        image_path: Path to input image.
        model_path: YOLO pose model weights.
        output_path: Path to save annotated image.
        conf: Confidence threshold.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load pose model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=False,
        verbose=False
    )

    if not results:
        print("No results returned.")
        return

    result = results[0]

    # Draw pose annotations on image
    annotated = result.plot()

    # Save annotated image
    cv2.imwrite(output_path, annotated)
    print(f"Annotated image saved to: {output_path}")

    # Print keypoints for each detected person
    if result.keypoints is not None:
        xy = result.keypoints.xy  # shape: (num_people, num_keypoints, 2)
        confs = result.keypoints.conf  # shape: (num_people, num_keypoints)

        for person_idx in range(len(xy)):
            print(f"\nPerson {person_idx + 1}:")
            for kp_idx, (point, score) in enumerate(zip(xy[person_idx], confs[person_idx])):
                x, y = point.tolist()
                s = float(score) if score is not None else None
                print(f"  Keypoint {kp_idx:02d}: x={x:.1f}, y={y:.1f}, conf={s:.3f}")
    else:
        print("No keypoints detected.")


if __name__ == "__main__":
    run_pose_on_image(
        image_path="pictures/cut.jpg",
        model_path="yolo26n-pose.pt",
        output_path="YoloTests/YoloPoseTest/PicResult/cut_result26.jpg",
        conf=0.25
    )