from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise RuntimeError("Open3D is required for visualization. Install it with: pip install open3d") from exc

from .config import DEFAULT_CONFIG
from .io import load_graph_json, resolve_path


DEFAULT_COLOR = [0.8, 0.8, 0.8]
CLASS_COLORS = {
    "person": [0.1, 0.35, 1.0],
    "dining table": [1.0, 0.65, 0.05],
    "table": [1.0, 0.65, 0.05],
    "bowl": [0.1, 0.8, 0.35],
    "bottle": [0.95, 0.15, 0.15],
    "chair": [0.65, 0.3, 1.0],
    "tv": [0.6, 0.6, 0.6],
    "mouse": [0.1, 0.9, 0.9],
    "keyboard": [0.9, 0.9, 0.2],
    "left_hand": [1.0, 0.25, 0.75],
    "right_hand": [0.25, 1.0, 0.75],
}

RELATION_COLORS = {
    "near": [0.75, 0.75, 0.75],
    "hand_near_object": [0.2, 0.9, 0.9],
    "touching": [1.0, 0.3, 0.15],
    "above": [1.0, 0.85, 0.2],
    "on_top_of": [0.25, 1.0, 0.35],
}


def graph_sort_key(path):
    take_name = path.parent.name
    frame_stem = path.stem.replace("frame_", "")
    try:
        frame_id = int(frame_stem)
    except ValueError:
        frame_id = -1
    return take_name, frame_id


def find_graph_files(root, task_name=None, take_name=None, max_frames=None):
    root = resolve_path(root)
    if not root.exists():
        raise FileNotFoundError(f"Stream graph folder not found: {root}")
    files = sorted(root.rglob("frame_*.json"), key=graph_sort_key)
    if task_name is not None:
        files = [path for path in files if task_name in path.parts]
    if take_name is not None:
        files = [path for path in files if path.parent.name == take_name]
    if max_frames is not None:
        files = files[:max_frames]
    if not files:
        raise FileNotFoundError(f"No frame_*.json graph files found under {root}")
    return files


def node_color(class_name):
    return CLASS_COLORS.get(class_name.lower(), DEFAULT_COLOR)


def relation_color(relation):
    return RELATION_COLORS.get(str(relation).lower(), [0.95, 0.95, 0.95])


def display_relation_name(relation):
    return str(relation).replace("_", " ")


def read_xyz(values, field_name, node_id):
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"Node {node_id} has invalid {field_name}: expected [x, y, z].")
    xyz = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"Node {node_id} has NaN or infinity in {field_name}.")
    return xyz


def read_valid_nodes(graph):
    valid_nodes = []
    skipped = []
    for node in graph.get("nodes", []):
        node_id = node.get("id") or "unknown_node"
        try:
            class_name = str(node["class_name"])
            confidence = float(node["confidence"])
            center = read_xyz(node["center_3d_m"], "center_3d_m", node_id)
            min_xyz = None
            max_xyz = None
            bbox = node.get("bbox_3d_m")
            if bbox is not None:
                min_xyz = read_xyz(bbox["min_xyz"], "bbox_3d_m.min_xyz", node_id)
                max_xyz = read_xyz(bbox["max_xyz"], "bbox_3d_m.max_xyz", node_id)
                if np.any(max_xyz <= min_xyz):
                    raise ValueError("bbox_3d_m max_xyz must be greater than min_xyz")
            valid_nodes.append({
                "id": node_id,
                "class_name": class_name,
                "confidence": confidence,
                "center": center,
                "min_xyz": min_xyz,
                "max_xyz": max_xyz,
            })
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append((node_id, str(exc)))
    return valid_nodes, skipped


def read_valid_edges(graph, nodes):
    nodes_by_id = {node["id"]: node for node in nodes}
    valid_edges = []
    skipped = []
    for edge in graph.get("edges", []):
        edge_id = edge.get("id", "unknown_edge")
        try:
            source_id = str(edge["source"])
            target_id = str(edge["target"])
            relation = str(edge["relation"])
            source = nodes_by_id[source_id]
            target = nodes_by_id[target_id]
            valid_edges.append({
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "directed": bool(edge.get("directed", True)),
                "confidence": float(edge.get("confidence", 1.0)),
                "source_center": source["center"],
                "target_center": target["center"],
            })
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append((edge_id, str(exc)))
    return valid_edges, skipped


def display_label_name(class_name):
    return class_name.replace("_", " ")


def assign_display_labels(nodes):
    class_counts = {}
    for node in nodes:
        class_counts[node["class_name"]] = class_counts.get(node["class_name"], 0) + 1
    class_seen = {}
    for node in nodes:
        class_name = node["class_name"]
        base_label = display_label_name(class_name)
        if class_counts[class_name] == 1:
            node["display_label"] = base_label
            continue
        class_seen[class_name] = class_seen.get(class_name, 0) + 1
        node["display_label"] = f"{base_label}{class_seen[class_name]}"


def create_center_sphere(center, color, config=DEFAULT_CONFIG):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=config.visualization.sphere_radius_m)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere


def create_bbox(min_xyz, max_xyz, color):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_xyz, max_bound=max_xyz)
    bbox.color = color
    return bbox


def create_label_point_cloud(text, anchor, config=DEFAULT_CONFIG, color=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    width = text_size[0] + 8
    height = text_size[1] + baseline + 8
    image = np.zeros((height, width), dtype=np.uint8)
    cv2.putText(image, text, (4, text_size[1] + 4), font, font_scale, 255, thickness, cv2.LINE_AA)
    ys, xs = np.where(image > 80)
    if xs.size == 0:
        return o3d.geometry.PointCloud()

    anchor = np.asarray(anchor, dtype=np.float64)
    scale = config.visualization.label_text_scale_m
    points = np.column_stack((
        anchor[0] + xs.astype(np.float64) * scale,
        anchor[1] + ys.astype(np.float64) * scale,
        np.full(xs.shape, anchor[2], dtype=np.float64),
    ))
    label = o3d.geometry.PointCloud()
    label.points = o3d.utility.Vector3dVector(points)
    label_color = color or [1.0, 1.0, 1.0]
    label.colors = o3d.utility.Vector3dVector(np.tile(label_color, (points.shape[0], 1)))
    return label


def create_edge_lines(edges):
    if not edges:
        return None
    points = []
    lines = []
    colors = []
    for edge in edges:
        line_index = len(lines)
        points.extend([edge["source_center"], edge["target_center"]])
        lines.append([line_index * 2, line_index * 2 + 1])
        colors.append(relation_color(edge["relation"]))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def edge_label_anchor(edge, edge_index, config=DEFAULT_CONFIG):
    midpoint = (edge["source_center"] + edge["target_center"]) / 2.0
    offset = np.asarray(config.visualization.edge_label_offset_m, dtype=np.float64)
    stagger = np.asarray([0.0, 0.0, 0.015 * (edge_index % 3)], dtype=np.float64)
    return midpoint + offset + stagger


def build_frame_geometries(nodes, edges=None, config=DEFAULT_CONFIG):
    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=config.visualization.axes_size_m)]
    edges = edges or []
    if config.visualization.show_edges:
        edge_lines = create_edge_lines(edges)
        if edge_lines is not None:
            geometries.append(edge_lines)
        if config.visualization.show_labels and config.visualization.show_edge_labels:
            for edge_index, edge in enumerate(edges):
                geometries.append(create_label_point_cloud(
                    display_relation_name(edge["relation"]),
                    edge_label_anchor(edge, edge_index, config),
                    config,
                    color=relation_color(edge["relation"]),
                ))

    label_offset = np.asarray(config.visualization.label_offset_m, dtype=np.float64)
    for node in nodes:
        color = node_color(node["class_name"])
        if config.visualization.show_3d_boxes and node["min_xyz"] is not None:
            geometries.append(create_bbox(node["min_xyz"], node["max_xyz"], color))
        if config.visualization.show_centroids:
            geometries.append(create_center_sphere(node["center"], color, config))
        if config.visualization.show_labels:
            geometries.append(create_label_point_cloud(node["display_label"], node["center"] + label_offset, config))
    return geometries


def setup_view(vis):
    render_options = vis.get_render_option()
    render_options.background_color = np.array([0.02, 0.02, 0.025])
    render_options.line_width = 2.0
    render_options.point_size = 2.0
    view = vis.get_view_control()
    view.set_front([0.0, -0.25, -1.0])
    view.set_up([0.0, -1.0, 0.0])
    view.set_zoom(0.65)


def print_frame_summary(graph_path, graph, nodes, edges, skipped_nodes, skipped_edges):
    metadata = graph.get("metadata", {})
    take_name = metadata.get("take", graph_path.parent.name)
    frame_id = graph.get("frame_id", graph_path.stem)
    classes = ", ".join(node["class_name"] for node in nodes) or "none"
    print(f"{take_name} frame {frame_id}: {len(nodes)} nodes [{classes}], edges={len(edges)}")
    if skipped_nodes:
        print(f"  skipped {len(skipped_nodes)} malformed nodes")
    if skipped_edges:
        print(f"  skipped {len(skipped_edges)} malformed edges")


def show_stream(graph_files, config=DEFAULT_CONFIG):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Scene Graph Stream", width=1280, height=800)
    frame_delay = 1.0 / max(config.visualization.playback_fps, 1)
    frame_index = 0
    initialized_view = False

    print(f"Loaded {len(graph_files)} graph frames.")
    print("Close the Open3D window to stop playback.")

    try:
        while True:
            graph_path = Path(graph_files[frame_index])
            graph = load_graph_json(graph_path)
            nodes, skipped = read_valid_nodes(graph)
            assign_display_labels(nodes)
            edges, skipped_edges = read_valid_edges(graph, nodes)
            geometries = build_frame_geometries(nodes, edges, config)

            vis.clear_geometries()
            for geometry_item in geometries:
                vis.add_geometry(geometry_item, reset_bounding_box=not initialized_view)

            if not initialized_view:
                setup_view(vis)
                initialized_view = True
            if config.visualization.print_frame_summary:
                print_frame_summary(graph_path, graph, nodes, edges, skipped, skipped_edges)
            if not vis.poll_events():
                break
            vis.update_renderer()
            time.sleep(frame_delay)

            frame_index += 1
            if frame_index >= len(graph_files):
                if config.visualization.loop_playback:
                    frame_index = 0
                else:
                    break
    finally:
        vis.destroy_window()


def summarize_graph(graph_path, graph, nodes, edges, skipped_nodes=None, skipped_edges=None):
    skipped_nodes = skipped_nodes or []
    skipped_edges = skipped_edges or []
    metadata = graph.get("metadata", {})
    frame_id = graph.get("frame_id", Path(graph_path).stem)
    print(f"\n3D scene graph for frame {frame_id}")
    if metadata.get("take"):
        print(f"Source: {metadata.get('subject', 'unknown')} / {metadata.get('task', 'unknown')} / {metadata['take']}")
    print("Coordinate frame: camera_xyz_m; X right, Y down, Z forward/depth")
    print(f"Nodes: {len(nodes)}, edges: {len(edges)}")

    for node in nodes:
        x, y, z = node["center"]
        if node["min_xyz"] is None:
            bbox_text = "bbox_size=n/a"
        else:
            size = node["max_xyz"] - node["min_xyz"]
            bbox_text = f"bbox_size=({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}) m"
        print(
            f"  {node['id']}: {node['class_name']} "
            f"conf={node['confidence']:.2f}, center=({x:.3f}, {y:.3f}, {z:.3f}) m, {bbox_text}"
        )

    if skipped_nodes:
        print(f"Skipped malformed nodes: {len(skipped_nodes)}")
    if skipped_edges:
        print(f"Skipped malformed edges: {len(skipped_edges)}")


def show_graph(graph, graph_path="<memory>", screenshot_path=None, show_window=True, config=DEFAULT_CONFIG):
    nodes, skipped_nodes = read_valid_nodes(graph)
    assign_display_labels(nodes)
    edges, skipped_edges = read_valid_edges(graph, nodes)
    summarize_graph(graph_path, graph, nodes, edges, skipped_nodes, skipped_edges)
    geometries = build_frame_geometries(nodes, edges, config)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Scene Graph", width=1280, height=800, visible=show_window)
    try:
        for geometry_item in geometries:
            vis.add_geometry(geometry_item)
        setup_view(vis)
        vis.poll_events()
        vis.update_renderer()

        if screenshot_path is not None:
            screenshot_path = resolve_path(screenshot_path)
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            vis.capture_screen_image(str(screenshot_path), do_render=True)
            print(f"Saved screenshot to: {screenshot_path}")

        if show_window:
            print("Close the Open3D window to finish.")
            vis.run()
    finally:
        vis.destroy_window()


def show_graph_file(graph_path, screenshot_path=None, show_window=True, config=DEFAULT_CONFIG):
    graph_path = resolve_path(graph_path)
    graph = load_graph_json(graph_path)
    show_graph(
        graph,
        graph_path=graph_path,
        screenshot_path=screenshot_path,
        show_window=show_window,
        config=config,
    )


def show_stream_from_root(root=None, task_name=None, take_name=None, config=DEFAULT_CONFIG):
    root = root or config.paths.stream_graph_root
    graph_files = find_graph_files(
        root,
        task_name=task_name,
        take_name=take_name,
        max_frames=config.visualization.max_frames,
    )
    show_stream(graph_files, config)
