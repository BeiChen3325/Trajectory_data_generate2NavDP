"""Project-wide filesystem locations."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RENDER_OUTPUT_DIR = OUTPUT_DIR / "render"
TRAJECTORY_OUTPUT_DIR = OUTPUT_DIR / "trajectory"

DEFAULT_LAS = DATA_DIR / "try1-pointcloud-0706.las"
DEFAULT_PLY = DATA_DIR / "try1_yup.ply"


def ensure_output_dirs() -> None:
    """Create standard output directories when a command starts."""

    RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
