from dataclasses import dataclass
from pathlib import Path


DEFAULT_LAS = Path(r"C:\task\xlk_work\MindCloudXAI_output\test1-pointcloud-0704.las")
DEFAULT_OUTPUT_DIR = Path(r"C:\task\xlk_work\tools\trajectory_work\outputs")


@dataclass
class MapConfig:
    las_path: Path = DEFAULT_LAS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    axis_transform: str = "zup-to-yup"
    floor_y_override: float | None = None
    roi_center_xz: tuple[float, float] | None = (0.0, 0.0)
    roi_size_xz: tuple[float, float] | None = (12.0, 12.0)
    floor_search_y_min: float = 0.0
    floor_search_y_max: float = 3.0

    # Y-up internal convention used by test1_yup.ply:
    # physical downward is +Y, physical upward is -Y, ground plane is X-Z.
    resolution_m: float = 0.08
    robot_radius_m: float = 0.25
    robot_height_m: float = 0.8
    ground_margin_m: float = 0.06
    safety_margin_m: float = 0.10
    ground_band_m: float = 0.08

    # Projection cleanup.
    min_points_per_cell: int = 2
    min_ground_points_per_cell: int = 2
    open_kernel_cells: int = 1
    close_kernel_cells: int = 2
    min_obstacle_component_cells: int = 8
    ground_close_kernel_cells: int = 2

    # LAS streaming.
    chunk_size: int = 1_000_000
    max_stream_points: int = 0
    floor_sample_limit: int = 1_200_000
    floor_hist_bins: int = 180
    floor_xy_resolution_m: float = 0.25

    # Planning.
    start_xz: tuple[float, float] | None = None
    goal_xz: tuple[float, float] | None = None
    min_start_goal_distance_m: float = 3.0
    obstacle_cost_weight: float = 0.8
    obstacle_cost_power: float = 1.5
    random_seed: int = 7

    # Path post-processing.
    shortcut_passes: int = 120
    smooth_samples_per_meter: float = 8.0
