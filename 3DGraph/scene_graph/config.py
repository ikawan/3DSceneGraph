from dataclasses import asdict, dataclass, field


@dataclass
class PathConfig:
    rgb_image_path: str = "bimacs_rgbd/bimacs_rgbd_data/subject_2/task_1_k_cooking/take_0/rgb/chunk_0/frame_72.png"
    depth_image_path: str = "bimacs_rgbd/bimacs_rgbd_data/subject_2/task_1_k_cooking/take_0/depth/chunk_0/frame_72.png"
    yolo_model_path: str = "yolo26m-seg.pt"
    camera_intrinsics_path: str = "3DGraph/outputs/calibration/d435_color_intrinsics.json"
    single_frame_output_dir: str = "3DGraph/outputs/scene_graph_single_frame"
    stream_output_dir: str = "3DGraph/outputs/scene_graph_stream"
    stream_graph_root: str = "3DGraph/outputs/scene_graph_stream"


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    pipeline_version: str = "0.1"
    verbose_yolo: bool = False


@dataclass
class DepthConfig:
    encoding: str = "auto"
    scale: float = 0.001
    min_m: float = 0.15
    max_m: float = 10.0
    person_class_name: str = "person"
    person_front_percentile: float = 20.0
    person_behind_tolerance_m: float = 0.15
    object_depth_tolerance_m: float = 0.10
    object_in_front_tolerance_m: float = 0.02


@dataclass
class CameraConfig:
    fallback_model: str = "approx_realsense_d435_depth_640x480"
    fallback_depth_fov_deg: tuple[float, float] = (85.2, 58.0)


@dataclass
class PerceptionConfig:
    min_mask_area_pixels: int = 25
    bbox_3d_percentile_low: float = 5.0
    bbox_3d_percentile_high: float = 95.0
    table_classes: tuple[str, ...] = ("dining table", "table")
    ignore_table_detections: bool = True
    ignored_detection_classes: tuple[str, ...] = ("tv",)


@dataclass
class HandConfig:
    enabled: bool = True
    max_num_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    depth_sample_radius_px: int = 6
    use_mediapipe_handedness: bool = False
    swap_mediapipe_handedness: bool = True
    disambiguate_duplicate_handedness: bool = True
    image_left_hand_label: str = "right_hand"
    image_right_hand_label: str = "left_hand"
    hand_mask_landmark_radius_px: int = 14
    hand_mask_connection_thickness_px: int = 10
    hand_mask_dilation_px: int = 8


@dataclass
class TrackingConfig:
    enabled: bool = True
    backend: str = "xmem"
    certainty_threshold: float = 0.55
    max_missing_frames: int = 15
    max_hand_tracks: int = 2
    detected_update_rate: float = 0.35
    missing_decay: float = 0.96
    min_match_score: float = 0.45
    hand_min_match_score: float = 0.30
    max_3d_match_distance_m: float = 0.75
    max_2d_center_distance_px: float = 140.0
    hand_max_3d_match_distance_m: float = 0.90
    hand_max_2d_center_distance_px: float = 180.0
    depth_similarity_m: float = 0.50
    min_observation_confidence: float = 0.20
    xmem_repo_path: str = "XMem"
    xmem_checkpoint_path: str = "XMem/saves/XMem.pth"
    xmem_size: int = 480
    xmem_top_k: int = 30
    xmem_mem_every: int = 5
    xmem_deep_update_every: int = -1
    xmem_enable_long_term: bool = True
    xmem_enable_long_term_count_usage: bool = True
    xmem_num_prototypes: int = 128
    xmem_min_mid_term_frames: int = 5
    xmem_max_mid_term_frames: int = 10
    xmem_max_long_term_elements: int = 10000
    xmem_min_mask_area_pixels: int = 25
    xmem_amp: bool = True


@dataclass
class RelationConfig:
    enabled: bool = True
    near_threshold_m: float = 0.35
    hand_near_threshold_m: float = 0.20
    touching_threshold_m: float = 0.06
    above_below_threshold_m: float = 0.12
    above_below_horizontal_threshold_m: float = 0.22
    table_support_threshold_m: float = 0.12
    table_support_overlap_threshold: float = 0.10
    bbox_2d_iou_touching_threshold: float = 0.02


@dataclass
class StreamConfig:
    dataset_root: str = "bimacs_rgbd/bimacs_rgbd_data"
    subject_name: str = "subject_2"
    num_takes: int = 5
    frame_stride: int = 1
    max_frames_per_take: int | None = None
    overwrite_existing: bool = True


@dataclass
class VisualizationConfig:
    playback_fps: int = 30
    loop_playback: bool = True
    max_frames: int | None = None
    show_rgb_frame: bool = True
    show_rgb_overlays: bool = False
    rgb_preview_max_width_px: int = 960
    show_edges: bool = True
    show_edge_labels: bool = True
    show_3d_boxes: bool = True
    show_centroids: bool = True
    show_labels: bool = True
    print_frame_summary: bool = True
    sphere_radius_m: float = 0.04
    axes_size_m: float = 0.35
    label_text_scale_m: float = 0.0022
    label_offset_m: tuple[float, float, float] = (0.0, -0.08, 0.0)
    edge_label_offset_m: tuple[float, float, float] = (0.0, -0.04, 0.0)


@dataclass
class SceneGraphConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    hands: HandConfig = field(default_factory=HandConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    relations: RelationConfig = field(default_factory=RelationConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = SceneGraphConfig()
