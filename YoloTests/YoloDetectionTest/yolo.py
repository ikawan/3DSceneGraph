from ultralytics import YOLO
# Load a pretrained YOLOv8n model
model = YOLO( 'yolov8n.pt')
# Run inference on the source
results = model(source = 'YoloTests/YoloDetectionTest/cutting.png', show = True, conf = 0.1, save = True, project= r'YoloTests/YoloDetectionTest/outputs', name = 'run 1')