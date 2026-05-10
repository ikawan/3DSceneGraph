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
- `scene_graph/hands.py` runs MediaPipe Hands and creates hand nodes.
- `scene_graph/tracking.py` stabilizes object and hand nodes across frames.
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
.venv\Scripts\python.exe 3DGraph/scripts/run_stream_batch.py --keep-tables
.venv\Scripts\python.exe 3DGraph/scripts/visualize_stream.py --take take_0 --hide-boxes --hide-edge-labels
```

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
`seen_frames`, and `missing_frames`.

Tracked object and hand nodes are carried forward at their last known position
while certainty remains above the configured threshold and the missing-frame
limit has not been reached. Hand tracking is capped to two active hand tracks
and can match through temporary left/right handedness label flips. If MediaPipe
labels both detected hands as the same side, the hand detector uses wrist
horizontal position to assign one `left_hand` and one `right_hand` before
tracking.

Edges are rule-based: `near`, `hand_near_object`, `touching`, `above`, and
`on_top_of`. Person nodes are excluded from relation creation, while hand nodes
are kept. Table detections are ignored by default; use `--keep-tables` to
include table nodes.
