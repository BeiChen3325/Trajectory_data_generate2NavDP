"""Validate the environment required by the repository's core scripts.

The gsplat backend import triggers JIT compilation for source installations
and reuses the compiled cache on later runs.
"""

from __future__ import annotations

import importlib
import shutil
import sys

from robotnav.config import ensure_output_dirs
from robotnav.navigation.scene.config import load_scene_build_config

PATHS = load_scene_build_config().paths
DATA_DIR = PATHS.data_dir


def check_base_dependencies() -> None:
    import cv2
    import numpy
    import plyfile

    print(f"[OK] numpy {numpy.__version__}")
    print(f"[OK] opencv {cv2.__version__}")
    print(f"[OK] plyfile {plyfile.__file__}")


def check_cuda_backend() -> None:
    import gsplat
    import torch

    print(f"[OK] torch {torch.__version__}")
    print(f"[OK] gsplat {gsplat.__version__}")
    print(f"[INFO] torch CUDA runtime: {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    print(f"[OK] CUDA device: {torch.cuda.get_device_name(0)}")

    nvcc = shutil.which("nvcc")
    if nvcc:
        print(f"[INFO] nvcc: {nvcc}")
    else:
        print("[WARN] nvcc was not found; a cached gsplat extension is required")

    # This is the decisive check. For a source gsplat install, this import
    # compiles the CUDA extension when no precompiled extension is present.
    from gsplat.cuda._backend import _C

    if _C is None:
        raise RuntimeError("gsplat CUDA backend loaded as None")
    print(f"[OK] gsplat CUDA backend: {_C}")


def check_project_modules() -> None:
    modules = (
        "robotnav.navigation.scene.builder",
        "robotnav.navigation.scene.ground_estimation",
        "robotnav.navigation.scene.las_io",
        "robotnav.navigation.scene.occupancy_map",
        "robotnav.navigation.trajectory.astar",
        "robotnav.navigation.trajectory.smoothing",
        "robotnav.navigation.trajectory.visualization",
        "robotnav.navigation.semantic_pointcloud.exporter",
        "robotnav.commands.prepare_navigation_data",
        "robotnav.rendering.render_one_view",
        "robotnav.rendering.compare_render",
        "robotnav.rendering.inspect_point_samples",
        "robotnav.rendering.gs_ply_to_usd",
    )
    for name in modules:
        importlib.import_module(name)
        print(f"[OK] project module: {name}")


def main() -> int:
    print(f"RobotNav environment check: {DATA_DIR.parent.parent}")
    ensure_output_dirs()
    if not DATA_DIR.exists():
        print(f"[WARN] input directory does not exist yet: {DATA_DIR}")
    try:
        check_base_dependencies()
        check_cuda_backend()
        check_project_modules()
    except Exception as exc:
        print(f"[FAIL] environment check: {exc!r}", file=sys.stderr)
        print("[FAIL] The environment is not ready for the core scripts.", file=sys.stderr)
        return 1

    print("[PASS] Environment is ready for the core trajectory and rendering scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
