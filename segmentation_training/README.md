# BIMACS YOLO26m Segmentation Workflow

This workflow trains `yolo26m-seg.pt` for instance segmentation on the BIMACS
object dataset while preserving the existing 16 class ids, including `unused1`
and `unused2`.

Use the project `.venv` for this workflow. The global Anaconda environment on
this machine currently has a NumPy/Matplotlib binary mismatch during training.

## 1. Generate Segmentation Labels

The current BIMACS labels are YOLO detection boxes. YOLO segmentation needs
polygon labels, so first generate SAM pseudo-masks from the existing boxes:

```powershell
.\.venv\Scripts\python.exe .\segmentation_training\generate_sam_segments.py --sam-model sam2_b.pt --device 0 --overwrite
```

This writes:

```text
darknet/bimacs_object_detection_data/labels-segment/
```

Review a sample of these masks before full training. SAM-generated masks are a
strong starting point, but they are still pseudo-labels.

## 2. Package the Ultralytics Dataset

```powershell
.\.venv\Scripts\python.exe .\segmentation_training\prepare_bimacs_seg_dataset.py --overwrite
```

This creates:

```text
datasets/bimacs-seg/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  data.yaml
  split_summary.json
```

The splitter groups frames by recording name so adjacent video frames do not
leak across train, validation, and test.

## 3. Train YOLO26m-Seg

```powershell
.\.venv\Scripts\python.exe .\segmentation_training\train_yolo26m_seg.py --epochs 150 --imgsz 640 --batch -1 --device 0
```

The default run writes to:

```text
runs/bimacs-seg/yolo26m-seg_640/
```

## 4. Validate the Test Split

```powershell
.\.venv\Scripts\yolo.exe segment val model=.\runs\bimacs-seg\yolo26m-seg_640\weights\best.pt data=.\datasets\bimacs-seg\data.yaml split=test device=0
```

Watch the mask mAP, per-class metrics, confusion matrix, and prediction plots.

## 5. Export for NVIDIA GPU Deployment

```powershell
.\.venv\Scripts\python.exe .\segmentation_training\export_tensorrt.py --model .\runs\bimacs-seg\yolo26m-seg_640\weights\best.pt --imgsz 640 --device 0
```

TensorRT export may require a matching local TensorRT installation. If TensorRT
is not ready, export ONNX first with Ultralytics and convert on the deployment
machine.
