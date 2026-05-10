import cv2
from ultralytics import YOLO
import os
import glob
import time
import torch

# Paths
image_dir = r"bimacs_rgbd\bimacs_object_detection_data\images"
label_dir = r"bimacs_rgbd\bimacs_object_detection_data\labels"
model_path = r"yolo26m-seg.pt"

# COCO class names
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Ground truth class names
gt_classes = ['unused1', 'bowl', 'knife', 'screwdriver', 'cuttingboard', 'whisk', 'hammer', 'bottle', 'unused2', 'cup', 'banana', 'cereals', 'sponge', 'wood', 'saw', 'harddrive']

# Intersecting classes
intersecting_classes = set(['bowl', 'knife', 'bottle', 'cup', 'banana'])

# Map COCO indices to names
coco_class_dict = {i: name for i, name in enumerate(coco_classes)}

# Map GT indices to names
gt_class_dict = {i: name for i, name in enumerate(gt_classes)}

# Function to compute IoU
def compute_iou(box1, box2):
    # box: [x_center, y_center, width, height]
    x1_min = box1[0] - box1[2]/2
    y1_min = box1[1] - box1[3]/2
    x1_max = box1[0] + box1[2]/2
    y1_max = box1[1] + box1[3]/2

    x2_min = box2[0] - box2[2]/2
    y2_min = box2[1] - box2[3]/2
    x2_max = box2[0] + box2[2]/2
    y2_max = box2[1] + box2[3]/2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    return inter_area / (box1_area + box2_area - inter_area)

# Function to load ground truth
def load_ground_truth(label_path):
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    class_name = gt_class_dict.get(class_id, 'unknown')
                    if class_name in intersecting_classes:
                        gt_boxes.append({
                            'class': class_name,
                            'box': [x_center, y_center, width, height]
                        })
    return gt_boxes

# Load model
model = YOLO(model_path)

# Check if running on GPU or CPU
print(f"Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Get all label files
label_files = glob.glob(os.path.join(label_dir, '*.txt'))

# Initialize metrics
class_metrics = {cls: {'TP': 0, 'FP': 0, 'FN': 0} for cls in intersecting_classes}
inference_times = []

# Process each frame
for label_file in label_files:  # Limit to first 10 for testing
    frame_name = os.path.splitext(os.path.basename(label_file))[0]
    image_path = os.path.join(image_dir, f'{frame_name}.jpg')
    
    if not os.path.exists(image_path):
        continue
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    # Run inference
    start_time = time.time()
    results = model(image, verbose=False)
    end_time = time.time()
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    
    # Get predictions
    pred_boxes = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xywh = box.xywh.cpu().numpy()[0]
            class_name = coco_class_dict.get(cls, 'unknown')
            if class_name in intersecting_classes:
                if class_name not in pred_boxes:
                    pred_boxes[class_name] = []
                pred_boxes[class_name].append({
                    'box': [xywh[0] / image.shape[1], xywh[1] / image.shape[0], xywh[2] / image.shape[1], xywh[3] / image.shape[0]],
                    'conf': conf
                })
    
    # Load ground truth
    gt_boxes = load_ground_truth(label_file)
    gt_by_class = {}
    for box in gt_boxes:
        cls = box['class']
        if cls not in gt_by_class:
            gt_by_class[cls] = []
        gt_by_class[cls].append(box['box'])
    
    # Evaluate per class
    for cls in intersecting_classes:
        gt_list = gt_by_class.get(cls, [])
        pred_list = pred_boxes.get(cls, [])
        
        # Sort pred by confidence
        pred_list.sort(key=lambda x: x['conf'], reverse=True)
        
        matched_gt = set()
        for pred in pred_list:
            best_iou = 0
            best_gt_idx = -1
            for i, gt in enumerate(gt_list):
                if i in matched_gt:
                    continue
                iou = compute_iou(pred['box'], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            if best_iou > 0.5:  # IoU threshold
                matched_gt.add(best_gt_idx)
                class_metrics[cls]['TP'] += 1
            else:
                class_metrics[cls]['FP'] += 1
        
        class_metrics[cls]['FN'] += len(gt_list) - len(matched_gt)

# Compute and print metrics
print("Evaluation Metrics (IoU > 0.5):")
for cls in intersecting_classes:
    tp = class_metrics[cls]['TP']
    fp = class_metrics[cls]['FP']
    fn = class_metrics[cls]['FN']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"{cls}: TP={tp}, FP={fp}, FN={fn}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Compute and print average inference time
if inference_times:
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"\nAverage inference time per frame: {average_inference_time:.3f} seconds")
else:
    print("\nNo frames processed for inference time.")

# Save detailed results for one frame
frame_name = 'recording_11-08-2018_14-06-08.849_rgb_frame_55'
image_path = os.path.join(image_dir, f'{frame_name}.jpg')
label_path = os.path.join(label_dir, f'{frame_name}.txt')

image = cv2.imread(image_path)
results = model(image, verbose=False)

pred_boxes = []
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xywh = box.xywh.cpu().numpy()[0]
        class_name = coco_class_dict.get(cls, 'unknown')
        if class_name in intersecting_classes:
            pred_boxes.append({
                'class': class_name,
                'x_center': xywh[0] / image.shape[1],
                'y_center': xywh[1] / image.shape[0],
                'width': xywh[2] / image.shape[1],
                'height': xywh[3] / image.shape[0],
                'confidence': conf
            })

gt_boxes = load_ground_truth(label_path)

output_file = 'YoloBenchmark\YoloSeg\evaluation_output.txt'
with open(output_file, 'w') as f:
    f.write("Ground Truth Boxes:\n")
    for box in gt_boxes:
        f.write(f"  Class: {box['class']}, Center: ({box['box'][0]:.3f}, {box['box'][1]:.3f}), Size: ({box['box'][2]:.3f}, {box['box'][3]:.3f})\n")
    f.write("\nPredicted Boxes:\n")
    for box in pred_boxes:
        f.write(f"  Class: {box['class']}, Center: ({box['x_center']:.3f}, {box['y_center']:.3f}), Size: ({box['width']:.3f}, {box['height']:.3f}), Conf: {box['confidence']:.3f}\n")

print(f"\nDetailed results for one frame saved to {output_file}")