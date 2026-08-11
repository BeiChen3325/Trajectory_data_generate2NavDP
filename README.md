# RobotNav trajectory data generation → NavDP

This repository contains a reproducible GPU data-generation pipeline for
Unitree Go2 navigation training data:

```text
LAS + 3DGS PLY → global navigation scene → long trajectories
→ camera trajectories → 3DGS RGB-D → RobotNav → NavDP
```

Generated scenes, images, depth, parquet files, point clouds and model inputs
are intentionally not versioned.  The repository contains code, configuration,
validation tools and small-batch smoke configurations only.

## Inputs

Place the current production inputs locally; do not commit them:

```text
data/input/4globalsecond.las
data/input/4globalsecond3DGS_yup.ply
```

The global-scene configuration reads LAS header bounds with ROI cropping
disabled.  The production architecture is a single global navigation scene and
long trajectories, rather than per-trajectory local scenes.

## GPU prerequisites

Use a CUDA-capable PyTorch installation and `gsplat`.  The renderer deliberately
requires CUDA; it does not fall back to a CPU rasterizer.

```bash
uv sync --extra gpu
uv run check-gpu-environment

# Optional: write a local, ignored preflight snapshot.
./scripts/export_gpu_environment.sh
```

`check-gpu-environment` verifies the NVIDIA runtime, PyTorch CUDA availability,
CUDA toolkit discovery and gsplat readiness.  `export_gpu_environment.sh` writes
`gpu_environment_report.txt`, which is intentionally ignored by Git.

## Reproducible 4globalsecond smoke run

The following uses two long trajectories (`5–10 m`) and isolated output paths.
It is the recommended preflight before launching a larger production batch.

```bash
uv run build-scene --config navigation_scene_forth_global_4globalsecond_camera_extrinsic_smoke.toml
uv run export-pointcloud --config pointcloud_export_forth_global_4globalsecond_camera_extrinsic_smoke.toml
uv run generate-trajectories --config trajectories_forth_global_4globalsecond_camera_extrinsic_smoke.toml
uv run build-dataset \
  --config dataset_build_forth_global_4globalsecond_camera_extrinsic_smoke.toml \
  --render-config render_forth_global_4globalsecond_camera_extrinsic_smoke.toml
uv run convert-navdp-dataset \
  --input data/target/robotnav/forth_global_4globalsecond_camera_extrinsic_smoke \
  --output traj_data/robotnav/forth_global_4globalsecond_camera_extrinsic_smoke
uv run navdp-loader-test \
  --dataset-root traj_data \
  --scene robotnav/forth_global_4globalsecond_camera_extrinsic_smoke \
  --loader-path "${INTERNNAV_ROOT}/internnav/dataset/navdp_lerobot_dataset.py"
```

Alternatively, export `INTERNNAV_LOADER_PATH` to the same file before invoking
`navdp-loader-test`.  InternNav is intentionally not bundled in this repository.

Outputs are ignored by design:

```text
outputs/forth_global_4globalsecond_camera_extrinsic_smoke/
data/target/robotnav/forth_global_4globalsecond_camera_extrinsic_smoke/
traj_data/robotnav/forth_global_4globalsecond_camera_extrinsic_smoke/
```

The loader test selects a single scene and filters RGB/Depth by that episode's
image-index range before invoking InternNav's `__getitem__`; it does not compare
the full scene image count with one episode parquet.

## NavDP camera-extrinsic contract

`observation.camera_extrinsic` is fixed camera calibration, not a trajectory
pose:

```text
observation.camera_extrinsic = T_base_from_camera
p_base = T_base_from_camera @ p_camera
```

It is stored as a row-major nested `float32[4][4]`.  The converter retains the
source fields `observation.T_base_from_camera`,
`observation.T_camera_from_base`, and `observation.T_world_camera`, and validates
that the first two are inverses.  It does not compose the camera extrinsic with
the NavDP world transform.

See [training compatibility](docs/training_compatibility.md) for the distinct
pixel-goal/action coordinate-chain requirement.

## Repository layout

- `src/robotnav/` — navigation scene, Go2 camera trajectory, 3DGS rendering,
  RobotNav packaging and NavDP conversion.
- `src/camera_resource/` — RealSense D435i intrinsics and camera pose preset.
- `src/go2_resource/` — Go2 URDF/xacro and controller source used by the
  project (large optional visual meshes are excluded).
- `configs/` — production, global-scene, long-trajectory and smoke TOML files.
- `scripts/` — GPU preflight and environment export helpers.
- `tests/` — pipeline, coordinate, depth and scene-bound validation.
