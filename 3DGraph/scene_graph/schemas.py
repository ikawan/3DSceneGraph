from __future__ import annotations

from typing import Any


def make_node(
    node_id: str,
    class_name: str,
    node_type: str,
    source_model: str,
    bbox_2d=None,
    center_2d=None,
    center_3d_m=None,
    median_depth_m=None,
    bbox_3d_m=None,
    confidence=None,
    attributes: dict[str, Any] | None = None,
) -> dict:
    attributes = attributes or {}
    node = {
        "id": node_id,
        "class_name": class_name,
        "node_type": node_type,
        "source_model": source_model,
        "bbox_2d": bbox_2d,
        "center_2d": center_2d,
        "center_3d_m": center_3d_m,
        "median_depth_m": median_depth_m,
        "bbox_3d_m": bbox_3d_m,
        "confidence": confidence,
        "attributes": attributes,
    }

    for key in (
        "landmarks_2d",
        "raw_handedness_label",
        "normalized_handedness_label",
        "corrected_handedness_label",
        "handedness_was_swapped",
        "handedness_disambiguation",
    ):
        if key in attributes:
            node[key] = attributes[key]

    return node


def make_edge(
    edge_id: str,
    source: str,
    target: str,
    relation: str,
    confidence: float = 1.0,
    directed: bool = True,
    attributes: dict[str, Any] | None = None,
) -> dict:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": float(confidence),
        "directed": bool(directed),
        "attributes": attributes or {},
    }


def make_scene_graph(
    frame_id,
    timestamp=None,
    camera_intrinsics=None,
    roi_filter=None,
    nodes=None,
    edges=None,
    metadata=None,
) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera_intrinsics": camera_intrinsics or {},
        "roi_filter": roi_filter or {},
        "nodes": nodes or [],
        "edges": edges or [],
        "metadata": metadata or {},
    }


def node_id(node: dict) -> str:
    return node.get("id")


def validate_scene_graph(graph: dict) -> list[str]:
    errors = []
    ids = [node_id(node) for node in graph.get("nodes", [])]
    missing_ids = [idx for idx, value in enumerate(ids) if not value]
    if missing_ids:
        errors.append(f"Nodes missing IDs at indices: {missing_ids}")

    duplicate_ids = sorted({value for value in ids if value and ids.count(value) > 1})
    if duplicate_ids:
        errors.append(f"Duplicate node IDs: {duplicate_ids}")

    node_ids = set(ids)
    for edge in graph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_ids:
            errors.append(f"Edge {edge.get('id')} has unknown source: {source}")
        if target not in node_ids:
            errors.append(f"Edge {edge.get('id')} has unknown target: {target}")

    return errors
