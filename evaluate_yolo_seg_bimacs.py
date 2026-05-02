"""
YOLO Segmentation Evaluation on BIMACS Object Detection Dataset

This script evaluates YOLO segmentation model performance on the BIMACS dataset
by comparing predictions against ground truth annotations.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import tqdm


# BIMACS object class names (excluding unused classes)
BIMACS_CLASSES = {
    0: "bowl",
    1: "knife", 
    2: "screwdriver",
    3: "cuttingboard",
    4: "whisk",
    5: "hammer",
    6: "bottle",
    7: "cup",
    8: "banana",
    9: "cereals",
    10: "sponge",
    11: "wood",
    12: "saw",
    13: "harddrive"
}

# COCO classes that can map to BIMACS classes
# YOLO segmentation models are trained on COCO dataset
COCO_TO_BIMACS_MAPPING = {
    # bowl -> bowl
    "bowl": "bowl",
    "cup": "cup",
    "bottle": "bottle",
    "banana": "banana",
    # knife - COCO has knife
    "knife": "knife",
    # scissors - closest to whisk
    "scissors": "whisk",
    # cutting board - not in COCO, will need to skip
    # hammer - not in COCO standard, but some models have it
    "hammer": "hammer",
    # saw - not in COCO
    # harddrive - not in COCO
    # sponge - not in COCO
    # wood - not in COCO
    # screwdriver - not in COCO
}

# More specific COCO->BIMACS mapping based on available classes
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush', 80: 'keyboard'
}

# Map COCO class names to BIMACS class IDs
# Only include classes that can map to BIMACS objects
COCO_TO_BIMACS_ID = {
    45: 0,   # bowl -> bowl (ID 0)
    43: 1,   # knife -> knife (ID 1)
    41: 7,   # cup -> cup (ID 7)
    39: 6,   # bottle -> bottle (ID 6)
    46: 8,   # banana -> banana (ID 8)
    76: 4,   # scissors -> whisk (ID 4) - closest match
}

# Filter: Only allow these COCO classes (all others ignored)
RELEVANT_COCO_CLASSES = set(COCO_TO_BIMACS_ID.keys())


def load_ground_truth(label_path: Path, img_width: int, img_height: int):
    """Load ground truth boxes from label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx = float(parts[1]) * img_width
                cy = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height
                boxes.append({
                    'class_id': class_id,
                    'x': cx - w/2,
                    'y': cy - h/2,
                    'w': w,
                    'h': h
                })
    return boxes


def box_iou(box1, box2):
    """Calculate IoU between two boxes in xyxy format."""
    x1_min = box1['x']
    y1_min = box1['y']
    x1_max = box1['x'] + box1['w']
    y1_max = box1['y'] + box1['h']
    
    x2_min = box2['x']
    y2_min = box2['y']
    x2_max = box2['x'] + box2['w']
    y2_max = box2['y'] + box2['h']
    
    # Calculate intersection
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Calculate union
    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def evaluate_model(
    model_path: str,
    images_dir: Path,
    labels_dir: Path,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    max_images: int = None  # None for all, or set to sample size
):
    """Evaluate YOLO model on BIMACS dataset."""
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get all label files
    label_files = sorted(labels_dir.glob("*.txt"))
    
    # Limit to sample if specified
    if max_images is not None and max_images > 0:
        label_files = label_files[:max_images]
    
    print(f"Found {len(label_files)} label files")
    
    # Statistics
    total_gt_boxes = 0
    total_pred_boxes = 0
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    
    # Per-class metrics
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'predictions': 0})
    
    # Process each image
    for label_file in tqdm(label_files, desc="Evaluating"):
        # Get corresponding image
        img_name = label_file.stem + ".jpg"
        img_path = images_dir / img_name
        
        if not img_path.exists():
            continue
        
        # Get image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        # Load ground truth
        gt_boxes = load_ground_truth(label_file, img_width, img_height)
        total_gt_boxes += len(gt_boxes)
        
        # Run YOLO prediction
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False
        )
        
        if len(results) == 0 or results[0].boxes is None:
            # No predictions - all GT boxes are false negatives
            for gt in gt_boxes:
                if gt['class_id'] in BIMACS_CLASSES:
                    false_negatives[gt['class_id']] += 1
                    class_metrics[gt['class_id']]['fn'] += 1
            continue
        
        result = results[0]
        pred_boxes = result.boxes
        
        # Convert predictions to dict format
        predictions = []
        for i in range(len(pred_boxes)):
            box = pred_boxes.xyxy[i].cpu().numpy()
            cls_id = int(pred_boxes.cls[i].cpu().numpy())
            conf = float(pred_boxes.conf[i].cpu().numpy())
            predictions.append({
                'class_id': cls_id,
                'x': box[0],
                'y': box[1],
                'w': box[2] - box[0],
                'h': box[3] - box[1],
                'conf': conf
            })
        
        total_pred_boxes += len(predictions)
        
        # Match predictions to ground truth
        matched_gt = set()
        
        for pred in predictions:
            # Map COCO class to BIMACS class
            bimacs_class_id = COCO_TO_BIMACS_ID.get(pred['class_id'])
            
            if bimacs_class_id is None:
                # Not a mappable class - IGNORE (filter out irrelevant classes)
                continue
            
            # Find best matching GT box
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if gt['class_id'] != bimacs_class_id:
                    continue
                
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True positive
                matched_gt.add(best_gt_idx)
                true_positives[bimacs_class_id] += 1
                class_metrics[bimacs_class_id]['tp'] += 1
            else:
                # False positive
                false_positives[bimacs_class_id] += 1
                class_metrics[bimacs_class_id]['fp'] += 1
        
        # Count unmatched GT as false negatives
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt and gt['class_id'] in BIMACS_CLASSES:
                false_negatives[gt['class_id']] += 1
                class_metrics[gt['class_id']]['fn'] += 1
    
    # Calculate overall metrics
    total_tp = sum(true_positives.values())
    total_fp = sum(false_positives.values())
    total_fn = sum(false_negatives.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Total images evaluated: {len(label_files)}")
    print(f"Total ground truth boxes: {total_gt_boxes}")
    print(f"Total predictions: {total_pred_boxes}")
    print(f"True positives: {total_tp}")
    print(f"False positives: {total_fp}")
    print(f"False negatives: {total_fn}")
    print(f"\nOverall Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print("-"*60)
    print(f"{'Class':<15} {'TP':<8} {'FP':<8} {'FN':<8} {'Prec':<8} {'Rec':<8}")
    print("-"*60)
    
    per_class_results = []
    for class_id in sorted(BIMACS_CLASSES.keys()):
        tp = class_metrics[class_id]['tp']
        fp = class_metrics[class_id]['fp']
        fn = class_metrics[class_id]['fn']
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        class_name = BIMACS_CLASSES[class_id]
        print(f"{class_name:<15} {tp:<8} {fp:<8} {fn:<8} {prec:.4f}    {rec:.4f}")
        
        per_class_results.append({
            'class_id': class_id,
            'class_name': class_name,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': prec,
            'recall': rec
        })
    
    # Summary for unmapped classes
    if -1 in class_metrics:
        print("-"*60)
        print(f"{'Unmapped':<15} {0:<8} {class_metrics[-1]['fp']:<8} {0:<8} {'N/A':<8} {'N/A':<8}")
    
    print("="*60)
    
    # Return results for saving
    return {
        'model': model_path,
        'images_evaluated': len(label_files),
        'total_gt_boxes': total_gt_boxes,
        'total_predictions': total_pred_boxes,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class': per_class_results
    }


if __name__ == "__main__":
    # Paths
    BASE_DIR = Path(r"c:\Users\ikawan\OneDrive\Desktop\Uni\Year4\StudyPeriod4\CSProject\Yolo")
    IMAGES_DIR = BASE_DIR / "bimacs_rgbd" / "bimacs_object_detection_data" / "images"
    LABELS_DIR = BASE_DIR / "bimacs_rgbd" / "bimacs_object_detection_data" / "labels"
    MODEL_PATH = BASE_DIR / "yolo26m-seg.pt"
    OUTPUT_DIR = BASE_DIR / "evaluation_output"
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run evaluation on sample of 500 images for quick testing
    results = evaluate_model(
        model_path=str(MODEL_PATH),
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        iou_threshold=0.5,
        conf_threshold=0.25,
        max_images=500  # Sample for quick testing
    )
    
    # Save results to JSON
    output_json = OUTPUT_DIR / "yolo26m_seg_evaluation.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_json}")