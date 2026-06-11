# Real-time 3D Scene Graphs for Robot Manipulation

This project builds RGB-D scene graphs from BIMACS cooking sequences. It combines:

- **YOLO segmentation** for object/person detection
- **MediaPipe hand detection** for hand pose estimation
- **XMem segmentation tracking** for temporal consistency
- **Depth-aware ROI filtering** for 3D understanding
- **Rule-based relation extraction** for semantic scene graphs

The main 3D scene graph pipeline lives in `3DGraph/scene_graph/` with command-line scripts in `3DGraph/scripts/`.

## Repository Structure

```
├── 3DGraph/
│   ├── scene_graph/         # Core pipeline (importable module)
│   ├── scripts/             # Command-line entry points
│   └── outputs/             # Generated files (Git-ignored)
├── benchmarks/              # Benchmark utilities (see benchmarks/README.md)
├── YoloBenchmark/           # YOLO detection/segmentation tests
├── YoloTests/               # Various YOLO model tests
```

## What Is NOT Stored In Git

Large data and model artifacts are intentionally ignored to keep the repository lean:

**Ignored Directories/Files:**
- `bimacs_rgbd/` - BIMACS RGBD dataset
- `bimacs_rgbd.zip` - Compressed dataset
- `*.pt` and `*.pth` - Model weights (YOLO, XMem)
- `videos/`, `OutputVids/` - Video input/output
- `3DGraph/outputs/` - Generated scene graphs and visualizations
- `runs/`, `datasets/`, `pictures/` - Experiment outputs
- `XMem/` - Tracking backend (cloned separately)

To run this project on another machine, you must download/copy these assets separately (see **Setup** below).

## Dependencies & Requirements

**System Requirements:**
- Python 3.9+
- NVIDIA GPU (optional, but recommended for real-time performance)
- CUDA 11.8+ (if using GPU)

**Package Dependencies:**
All listed in `requirements.txt`:
- PyTorch 2.0+ with TorchVision
- YOLO (Ultralytics)
- OpenCV, NumPy, SciPy, Pillow
- MediaPipe for hand detection
- Open3D for 3D visualization
- tqdm, PyYAML for utilities

The tracking backend (XMem) is installed separately from the official repository.

## Quick Start

### 1. Clone the Repository

```powershell
git clone <https://github.com/ikawan/3DSceneGraph.git>
cd Yolo
```

### 2. Set Up Python Environment

**Create a virtual environment:**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Install dependencies:**

```powershell
pip install -r requirements.txt
```

### 3. Install XMem (for Tracking)

The project uses the official XMem repository for segmentation tracking:

```powershell
git clone https://github.com/hkchengrex/XMem.git
pip install -r XMem/requirements.txt
```

### 4. Download Data

**BIMACS Dataset:**

If processing the BIMACS cooking dataset, download it and extract to:

```
bimacs_rgbd/bimacs_rgbd_data/
```

Expected structure:

```
bimacs_rgbd/bimacs_rgbd_data/
├── subject_1/
│   ├── task_1_k_cooking/
│   │   ├── take_0/
│   │   │   ├── rgb/
│   │   │   ├── depth/
│   │   │   └── ...
│   │   └── take_1/
│   └── task_2_...
└── subject_2/
    └── ...
```

## Pipeline Usage

### Run Single Frame

Process a single RGB-D frame pair to generate a 3D scene graph:

```powershell
.\.venv\Scripts\python.exe 3DGraph/scripts/run_single_frame.py
```

**Default configuration** loads from `3DGraph/scene_graph/config.py`:
- RGB: `bimacs_rgbd/bimacs_rgbd_data/subject_2/task_1_k_cooking/take_0/rgb/chunk_0/frame_72.png`
- Depth: corresponding depth frame
- Output: `3DGraph/outputs/scene_graph_single_frame/`

### Run Batch Processing (Stream)

Process multiple frames/videos from the BIMACS dataset:

```powershell
.\.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py --num-takes 1 --max-frames-per-take 20
```

### Visualize Results

**Visualize single frame graph:**

```powershell
.\.venv\Scripts\python.exe 3DGraph/scripts/visualize_single.py --graph 3DGraph/outputs/scene_graph_single_frame/scene_graph_frame_72.json
```

**Visualize stream results:**

```powershell
.\.venv\Scripts\python.exe 3DGraph/scripts/visualize_stream.py --take take_0
```

## Benchmark Utilities

The `benchmarks/` directory contains standalone tools for testing and validation:

- **`all_benchmark.py`**: Process custom videos with YOLO segmentation + MediaPipe hands
- **`all_ground_truth.py`**: Batch process BIMACS dataset for evaluation

See [benchmarks/README.md](benchmarks/README.md) for details.

## Pipeline Behavior

The default pipeline includes:

- **Detection**: YOLO v8 segmentation to detect objects and people
- **Hand Detection**: MediaPipe for hand pose landmarks
- **Depth Processing**: Converts 2D detections to 3D bounding boxes
- **Filtering**: Excludes table/TV detections by default
- **Tracking**: XMem tracks masks across frames (optional)
- **Hand Limits**: Caps active hands to 2 per frame
- **Relations**: Generates semantic edges (near, touching, above, hand_near_object)

## Configuration

Edit `3DGraph/scene_graph/config.py` to customize:

- **PathConfig**: Input/output paths and model weights
- **RuntimeConfig**: Device (cuda:0 / cpu), verbosity
- **DepthConfig**: Depth processing parameters
- **PerceptionConfig**: Detection filtering (table exclusion, etc.)
- **HandConfig**: Hand detection sensitivity and limits
- **TrackingConfig**: XMem parameters and tracking behavior

Example to disable tracking:

```python
from 3DGraph.scene_graph.config import SceneGraphConfig

config = SceneGraphConfig()
config.tracking.enabled = False
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes with clear messages
4. Push to the branch and open a Pull Request

## Citation

If you use this project in research, please cite the BIMACS dataset and original papers:

```bibtex
@misc{yolo_scene_graphs,
  title={3D Scene Graph},
  author={Ahmad Hamid & Ibrahim Aljuboori},
  year={2026},
  howpublished={\url{https://github.com/ikawan/3DSceneGraph.git}}
}
```

## Related Work

- **YOLO v8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **MediaPipe**: [Google MediaPipe](https://mediapipe.dev/)
- **XMem**: [Long-Term Class-Incremental Learning](https://github.com/hkchengrex/XMem)
- **Open3D**: [Open3D Library](http://www.open3d.org/)

## License

[e.g., MIT, Apache 2.0, etc.]

