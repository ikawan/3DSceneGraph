from __future__ import annotations

from . import geometry
from .config import DEFAULT_CONFIG
from .schemas import make_edge, node_id


def edge_id(source, relation, target):
    return f"edge_{source}_{relation}_{target}".replace(" ", "_")


def add_edge_once(edges, seen, source, target, relation, confidence=1.0, directed=True, attributes=None):
    eid = edge_id(source, relation, target)
    if eid in seen:
        return
    seen.add(eid)
    edges.append(make_edge(eid, source, target, relation, confidence, directed, attributes))


def center(node):
    return node.get("center_3d_m")


def node_kind(node):
    return node.get("node_type") or ("hand" if "hand" in node.get("class_name", "") else "object")


def is_table(node):
    return node_kind(node) == "table" or node.get("class_name", "").lower() in {"table", "dining table"}


def is_hand(node):
    return node_kind(node) == "hand" or node.get("class_name", "").endswith("_hand")


def is_person(node):
    return node_kind(node) == "person" or node.get("class_name", "").lower() == "person"


def extract_relations(nodes, previous_graph=None, config=DEFAULT_CONFIG):
    config = config or DEFAULT_CONFIG
    if not config.relations.enabled:
        return []

    edges = []
    seen = set()
    above_pairs = {}
    touching_pairs = {}
    valid_nodes = [
        node
        for node in nodes
        if node_id(node) and center(node) is not None and not is_person(node)
    ]

    for i, a in enumerate(valid_nodes):
        for b in valid_nodes[i + 1:]:
            a_id = node_id(a)
            b_id = node_id(b)
            dist = geometry.distance_3d(center(a), center(b))
            if dist is None:
                continue

            near_threshold = (
                config.relations.hand_near_threshold_m
                if is_hand(a) or is_hand(b)
                else config.relations.near_threshold_m
            )
            if dist <= near_threshold:
                relation = "hand_near_object" if is_hand(a) or is_hand(b) else "near"
                add_edge_once(
                    edges,
                    seen,
                    a_id,
                    b_id,
                    relation,
                    confidence=max(0.0, 1.0 - dist / near_threshold),
                    directed=False,
                    attributes={"distance_m": dist, "threshold_m": near_threshold},
                )

            bbox_dist = geometry.bbox_3d_distance(a.get("bbox_3d_m"), b.get("bbox_3d_m"))
            iou_2d = geometry.bbox_2d_iou(a.get("bbox_2d"), b.get("bbox_2d"))
            touching_by_3d = bbox_dist is not None and bbox_dist <= config.relations.touching_threshold_m
            touching_by_2d = iou_2d is not None and iou_2d >= config.relations.bbox_2d_iou_touching_threshold
            if touching_by_3d or touching_by_2d:
                touching_attributes = {
                    "bbox_3d_distance_m": bbox_dist,
                    "bbox_2d_iou": iou_2d,
                    "touching_threshold_m": config.relations.touching_threshold_m,
                }
                touching_pairs[frozenset((a_id, b_id))] = touching_attributes
                add_edge_once(
                    edges,
                    seen,
                    a_id,
                    b_id,
                    "touching",
                    confidence=0.8,
                    directed=False,
                    attributes=touching_attributes,
                )

            horizontal = geometry.horizontal_distance_xz(center(a), center(b))
            if horizontal is not None and horizontal <= config.relations.above_below_horizontal_threshold_m:
                dy = center(a)[1] - center(b)[1]  # Camera Y points down.
                if abs(dy) >= config.relations.above_below_threshold_m:
                    if dy < 0:
                        above_attributes = {
                            "vertical_delta_y_m": float(dy),
                            "horizontal_distance_xz_m": horizontal,
                            "threshold_m": config.relations.above_below_threshold_m,
                        }
                        above_pairs[(a_id, b_id)] = above_attributes
                        add_edge_once(
                            edges,
                            seen,
                            a_id,
                            b_id,
                            "above",
                            confidence=0.7,
                            directed=True,
                            attributes=above_attributes,
                        )
                    else:
                        above_attributes = {
                            "vertical_delta_y_m": float(-dy),
                            "horizontal_distance_xz_m": horizontal,
                            "threshold_m": config.relations.above_below_threshold_m,
                        }
                        above_pairs[(b_id, a_id)] = above_attributes
                        add_edge_once(
                            edges,
                            seen,
                            b_id,
                            a_id,
                            "above",
                            confidence=0.7,
                            directed=True,
                            attributes=above_attributes,
                        )

    for source in valid_nodes:
        if is_table(source):
            continue
        for target in valid_nodes:
            if source is target or not is_table(target):
                continue
            source_id = node_id(source)
            target_id = node_id(target)
            above_attributes = above_pairs.get((source_id, target_id))
            touching_attributes = touching_pairs.get(frozenset((source_id, target_id)))
            if above_attributes is None or touching_attributes is None:
                continue
            add_edge_once(
                edges,
                seen,
                source_id,
                target_id,
                "on_top_of",
                confidence=0.75,
                directed=True,
                attributes={
                    "requires": ["above", "touching"],
                    "above": above_attributes,
                    "touching": touching_attributes,
                },
            )

    return edges
