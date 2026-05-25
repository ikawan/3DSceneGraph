# 3D Scene Graph Pipeline

`3DGraph` is organized around one importable package and a small set of command
entry points:

- `scene_graph/` contains the pipeline modules.
- `scripts/` contains runnable CLIs.
- `outputs/` contains generated graphs, debug artifacts, and calibration files.

## Architecture

- `scene_graph/config.py` stores paths, thresholds, runtime options, tracking
  settings, relation settings, and visualization settings.
- `scene_graph/io.py` loads RGB/depth frames, decodes depth, loads camera
  intrinsics, lists BIMACS frames, and saves graph JSON.
- `scene_graph/camera.py` exports RealSense D435 color intrinsics.
- `scene_graph/geometry.py` contains pure geometry, depth, mask, and bbox
  helpers.
- `scene_graph/perception.py` runs YOLO segmentation and person-depth ROI
  filtering for scene nodes.
- `scene_graph/hands.py` runs MediaPipe Hands, creates landmark-derived hand
  masks, and leaves handedness identity to tracking by default.
- `scene_graph/tracking.py` uses XMem to propagate stable object and hand masks
  across frames.
- `scene_graph/relations.py` creates rule-based graph edges.
- `scene_graph/graph_builder.py` builds one frame graph.
- `scene_graph/pipeline.py` processes one frame, one take, or a batch of takes.
- `scene_graph/roi_debug.py` exports one-frame ROI/mask debug artifacts.
- `scene_graph/visualization.py` visualizes graph dictionaries and JSON files.

## Commands

Build one frame:

```powershell
.venv\Scripts\python.exe 3DGraph/scripts/run_single_frame.py
```

Build stream graphs for BIMACS takes:

```powershell
.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py
```

Visualize one graph:

```powershell
.venv\Scripts\python.exe 3DGraph/scripts/visualize_single.py --graph 3DGraph/outputs/scene_graph_single_frame/scene_graph_frame_72.json
```

Visualize stream output:

```powershell
.venv\Scripts\python.exe 3DGraph/scripts/visualize_stream.py --take take_0
```

Export ROI debug artifacts for one frame:

```powershell
.venv\Scripts\python.exe 3DGraph/scripts/export_roi_debug.py
```

Export RealSense D435 color intrinsics:

```powershell
.venv\Scripts\python.exe 3DGraph/scripts/export_d435_intrinsics.py
```

Useful options:

```powershell
.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py --num-takes 1 --max-frames-per-take 20
.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py --tracking-certainty-threshold 0.55 --tracking-max-missing-frames 15 --tracking-max-hand-tracks 2
.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py --xmem-repo XMem --xmem-checkpoint XMem/saves/XMem.pth --xmem-size 480
.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py --image-left-hand-label right_hand --image-right-hand-label left_hand
.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py --keep-tables
.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py --no-tracking
.venv\Scripts\python.exe 3DGraph/scripts/visualize_stream.py --take take_0 --hide-boxes --hide-edge-labels
.venv\Scripts\python.exe 3DGraph/scripts/visualize_stream.py --take take_0 --fps 60
.venv\Scripts\python.exe 3DGraph/scripts/visualize_stream.py --take take_0 --show-rgb-overlays
```

The visualizers open a synchronized `RGB Frame` window by default when graph
metadata contains `rgb_path`. The RGB preview shows the raw frame by default.
Use `--hide-rgb-frame` to return to the Open3D-only view, or
`--show-rgb-overlays` to draw graph centers and labels over the frame.

## XMem Setup

Temporal tracking expects the official XMem repository and checkpoint:

```powershell
git clone https://github.com/hkchengrex/XMem.git XMem
pip install -r XMem/requirements.txt
```

Download `XMem.pth` from the XMem v1.0 release and place it at
`XMem/saves/XMem.pth`, or point the scripts at custom paths with
`--xmem-repo` and `--xmem-checkpoint`.

## Graph Format

Each graph JSON contains:

- `frame_id`
- `timestamp`
- `camera_intrinsics`
- `roi_filter`
- `nodes`
- `edges`
- `metadata`

Nodes use `id` as their canonical identifier. With temporal tracking enabled,
node IDs become stable track IDs and each node includes a `tracking` block with
`track_id`, `certainty`, `detected_this_frame`, `carried_forward`,
`xmem_label_id`, `seen_frames`, and `missing_frames`.

Tracked object and hand nodes are rebuilt from current XMem masks while
certainty remains above the configured threshold and the missing-frame limit has
not been reached. YOLO and MediaPipe detections are correction signals; when
they disappear for a frame, XMem still propagates the previous labels.

Hand tracking is capped to two active tracks. MediaPipe detections are treated
as generic `hand` observations by default because the stable left/right identity
comes from XMem. When a hand track is born, the image-left hand becomes
`right_hand` and the image-right hand becomes `left_hand` by default; XMem then
keeps that identity through missed detections. Use
`--use-mediapipe-handedness` only if you want to keep MediaPipe's handedness
labels as an extra cue.

Edges are rule-based: `near`, `hand_near_object`, `touching`, `above`, and
`on_top_of`. Person nodes are excluded from relation creation, while hand nodes
are kept. Table and TV detections are ignored by default; use `--keep-tables`
to include table nodes.
