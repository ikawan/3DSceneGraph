# Benchmarking and Testing Utilities

This directory contains standalone benchmark and testing scripts for the 3D Scene Graph pipeline.

## Scripts

### `all_benchmark.py`
Processes video files with YOLO segmentation and MediaPipe hand detection to generate annotated output videos. Useful for visual validation of the detection pipeline on custom video data.

**Dependencies:**
- cv2 (OpenCV)
- mediapipe
- ultralytics (YOLO)

**Configuration:**
Edit the top-level constants in the script:
- `VIDEO_PATH`: Input video file
- `SEG_MODEL_PATH` / `POSE_MODEL_PATH`: YOLO weight files
- `OUTPUT_PATH`: Output video destination
- `DEVICE`: "cuda:0" for GPU, "cpu" for CPU

### `all_ground_truth.py`
Batch processes BIMACS RGBD dataset frames with YOLO segmentation and MediaPipe hands. Generates annotated visualizations for dataset validation and model evaluation.

**Dependencies:**
- cv2 (OpenCV)
- mediapipe
- torch
- ultralytics (YOLO)

**Configuration:**
The script expects the BIMACS dataset at `bimacs_rgbd/bimacs_rgbd_data/`. Edit constants:
- `BASE_PATH`: Path to BIMACS data
- `SEG_MODEL_PATH`: YOLO weights
- `DEVICE`: GPU or CPU
- `MAX_TAKES_PER_TASK`: How many takes per task to process

## Running the Benchmarks

```powershell
# Activate the virtual environment
.\.venv\Scripts\activate

# Run the benchmark on a video
python benchmarks/all_benchmark.py

# Run ground truth processing on BIMACS dataset
python benchmarks/all_ground_truth.py
```

## Notes

- These scripts are GPU-optimized but can run on CPU (slower)
- Ensure YOLO weight files (`.pt`) are present in the workspace root
- For the ground truth script, you need the BIMACS dataset downloaded separately
- Output videos are saved to the configured `OUTPUT_PATH` / `OutputVids/` directory
