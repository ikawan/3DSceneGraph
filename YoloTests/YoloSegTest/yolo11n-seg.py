from pathlib import Path
import cv2
from ultralytics import YOLO


def segment_image(
    image_path: str,
    model_path: str = "yolo11n-seg.pt",
    output_path: str = "segmented_result.jpg",
    conf: float = 0.25
) -> None:
    """
    Run instance segmentation on a single image and save the visualized result.

    Args:
        image_path: Path to input image.
        model_path: YOLO segmentation model, e.g. yolo11n-seg.pt
        output_path: Path to save annotated image.
        conf: Confidence threshold.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load segmentation model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=False,
        verbose=True
    )

    # There will be one result for one input image
    result = results[0]

    # Render masks + boxes + labels onto the image
    annotated = result.plot()

    # Save annotated output
    cv2.imwrite(output_path, annotated)
    print(f"Saved result to: {output_path}")

    # Optional: print some structured info
    if result.boxes is not None:
        print(f"Detected {len(result.boxes)} objects")

    if result.masks is not None:
        print("Segmentation masks were generated")
    else:
        print("No segmentation masks were generated")


if __name__ == "__main__":
    segment_image(
        image_path="pictures/test2.jpg",         # change this
        model_path="yolo11n-seg.pt",    # or yolo11s-seg.pt
        output_path="YoloTests/YoloSegTest/PicResult/segmented_result.jpg",
        conf=0.25
    )