# Training compatibility

## NavDP camera extrinsic

The generator writes `observation.camera_extrinsic` as the fixed calibration
`T_base_from_camera`, with column-vector semantics:

```text
p_base = T_base_from_camera @ p_camera
```

It is a row-major nested `float32[4][4]` parquet field.  It is not a
world-to-camera pose and must not be used as one.

## Pixel-goal projection

No generator change is required for the pixel-goal issue.  NavDP `action` is
the trajectory-dependent camera pose in NavDP world coordinates:

```text
action = T_NavDP-world_from_camera
```

Training code must use the `action` NavDP-world coordinate chain when projecting
world goals into camera pixels.  It must not substitute
`observation.camera_extrinsic` for a world-to-camera transform.

The required training-side change belongs in the separate InternNav repository,
at `internnav/dataset/navdp_lerobot_dataset.py`.  This repository intentionally
does not vendor or modify InternNav; submit that training-side fix in its own
repository.
