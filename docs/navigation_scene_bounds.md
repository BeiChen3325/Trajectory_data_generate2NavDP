# Global navigation-scene bounds

`build-scene` now creates a global navigation map from the LAS header X-Z bounds by default.
The resulting `scene_manifest.json` records `world_bounds`, their `source` (`las_header`), and
the map resolution in `grid.resolution_m`.

`[scene].enable_roi_crop = true` is an opt-in diagnostic mode. It uses
`roi_center_xz` and `roi_size_xz` with the existing 0.5 m padding, and records
`world_bounds.source = "roi_crop"`.

Existing local scene artifacts, including `outputs/forth_local/navigation_scene`, remain readable
and can still be used by trajectory generation. To migrate to a global artifact, choose a new
`[paths].output_dir`, keep `enable_roi_crop = false`, run `build-scene`, annotate one
`valid_region.yaml` on that global map, then point the trajectory configuration at the new scene
directory and YAML. `valid_region.yaml` remains a trajectory-stage input and is not used by the
scene builder.
