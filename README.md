# Real-time 3D Scene Graphs for Robot Manipulation

This project builds RGB-D scene graphs from BIMACS cooking sequences. It combines
YOLO segmentation, MediaPipe hand detection, XMem segmentation tracking,
depth-aware ROI filtering, and rule-based relation extraction.

The main 3D scene graph pipeline lives in `3DGraph/`.

## Repository Layout

- `3DGraph/scene_graph/` - importable pipeline modules
- `3DGraph/scripts/` - command-line entry points
- `3DGraph/outputs/` - generated files, ignored by Git
- `YoloBenchmark/`, `YoloTests/`, `MediaPipeHands/` - supporting experiments
- `allBenchmark.py`, `allGroundTruth.py` - project-level utilities

## What Is Not Stored In Git

Large data and model artifacts are intentionally ignored:

- `bimacs_rgbd/`
- `bimacs_rgbd.zip`
- `*.pt` YOLO weights
- videos and generated outputs
- `3DGraph/outputs/`

To run the project on another PC, copy/download these assets separately.

Expected local assets include:

- BIMACS data under `bimacs_rgbd/bimacs_rgbd_data/`
- YOLO segmentation weights such as `yolo26m-seg.pt`
- the official XMem repository under `XMem/`
- XMem weights at `XMem/saves/XMem.pth`
- optional pose weights if using the older benchmark utilities

## Setup On A New PC

```powershell
git clone <your-repo-url>
cd Yolo

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Then place the ignored assets in the same relative locations used by the config:

```text
bimacs_rgbd/bimacs_rgbd_data/
yolo26m-seg.pt
```

If you want GPU inference, install a CUDA-compatible PyTorch build for the new
machine before running the pipeline.

Tracking uses the official XMem implementation rather than the unrelated
`xmem` PyPI package:

```powershell
git clone https://github.com/hkchengrex/XMem.git XMem
pip install -r XMem/requirements.txt
```

Download `XMem.pth` from the official XMem v1.0 release and place it at
`XMem/saves/XMem.pth`, or pass `--xmem-repo` and `--xmem-checkpoint` to the
pipeline scripts. Use `--no-tracking` for detector-only runs.

## Run One Frame

```powershell
.\.venv\Scripts\python.exe 3DGraph/scripts/run_single_frame.py
```

## Run A Stream Batch

```powershell
.\.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py --num-takes 1 --max-frames-per-take 20
```

## Visualize Results

```powershell
.\.venv\Scripts\python.exe 3DGraph/scripts/visualize_single.py --graph 3DGraph/outputs/scene_graph_single_frame/scene_graph_frame_72.json
.\.venv\Scripts\python.exe 3DGraph/scripts/visualize_stream.py --take take_0
```

## Notes

The default pipeline ignores table and TV detections, excludes person nodes from
relations, tracks object/hand masks across frames with XMem, caps active hands
to two, and generates relation edges such as `near`, `touching`, `above`, and
`hand_near_object`.
