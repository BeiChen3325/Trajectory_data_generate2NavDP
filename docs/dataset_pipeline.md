# 多轨迹目标数据集流水线

## 数据映射

数据集流水线消费完整的 `trajectory_manifest.json`。一个 manifest 对应一个 target scene，
manifest 中每条 route 对应一个 episode：

```text
trajectory_manifest.json
  ├── routes/auto_000.json -> episode 000 -> 000.parquet
  ├── routes/auto_001.json -> episode 001 -> 001.parquet
  └── ...
```

所有 route 必须绑定同一个 `source_scene_model_sha256`，并共享最终 scene 中的
`meta/pointcloud.ply`。

episode 顺序严格使用 manifest 的 `trajectories` 数组顺序。第 `i` 条 route 使用数字化的
episode 名称 `000`、`001`……，从而保证排序后的 parquet 顺序与
`episodes_stats.jsonl` 行序一致。原始 `trajectory_id` 保留在中间目录和
`run_manifest.json` 中。

## 配置

`configs/dataset_build.toml` 只负责连接已经生成的 Navigation 产物、数据集工作目录和打包
参数：

```toml
[paths]
trajectory_manifest = "outputs/trajectories/trajectory_manifest.json"
semantic_pointcloud_dir = "outputs/semantic_pointcloud"
semantic_pointcloud_filename = "pointcloud.ply"
semantic_pointcloud_report_filename = "pointcloud_report.json"
work_dir = "outputs/dataset_build"
dataset_root = "data/target"
```

轨迹数量、顺序和 route 路径只来自 trajectory manifest。3DGS PLY、图像宽高、FOV 和背景
只来自 `render.toml`。LAS、场景构建和 A* 参数不属于数据集配置。

## 输入校验

`load_trajectory_batch()` 在相机转换和 GPU 渲染之前完成以下校验：

- manifest 版本、非空条目和数量字段；
- `trajectory_id` 唯一且可安全作为目录名；
- route 使用 manifest 目录内的安全相对路径；
- route 文件 SHA-256 等于 manifest 条目；
- route 内的 ID 和 scene model SHA 与 manifest 一致；
- `floor_y`、`smooth_path_xz` 和坐标约定完整有效。

打包前还会校验：

- camera manifest 与 route、batch manifest、scene model 的绑定；
- render manifest 与 camera 文件、route、batch manifest、scene model 的绑定；
- RGB/Depth 数量、格式、尺寸和帧数；
- pointcloud report 的 scene model SHA；
- report 中记录的 pointcloud SHA 与当前 PLY；
- PLY 中存在目标深蓝色障碍点。

任何 episode 校验失败都会使整个命令失败，不会静默减量。

## 中间目录

每条轨迹拥有独立工作目录：

```text
outputs/dataset_build/
├── batch_manifest.json
└── episodes/
    ├── auto_000/
    │   ├── camera_trajectory.npz
    │   ├── camera_trajectory.json
    │   └── rendered_episode/
    │       ├── rgb/
    │       ├── depth/
    │       └── render_manifest.json
    └── auto_001/
        └── ...
```

`EpisodePaths` 是这些路径的唯一构造位置，防止 episode 之间覆盖文件。
`batch_manifest.json` 汇总 route、camera 和 render 产物的路径、SHA 与当前阶段状态，但下游
仍会复算被引用文件的哈希。

## 三个阶段

### 1. route → camera

```bash
uv run robotnav-trajectory-to-camera --config dataset_build.toml
```

程序按 manifest 顺序遍历 route，复用同一个 `build_camera_trajectory()` 姿态算法，为每条
轨迹写独立的 NPZ 和 JSON。

NPZ 包含：

- `camera_to_world: (T,4,4)`，写入 parquet 的 `action`；
- `world_to_camera: (T,4,4)`，交给 gsplat；
- `frame_index: (T,)`，连续的 `0..T-1`。

JSON 记录 route SHA、batch manifest SHA、scene model SHA、轨迹 ID、episode 序号、相机高度
和坐标约定。

### 2. camera → RGB-D

```bash
uv run robotnav-render-trajectory \
  --config dataset_build.toml \
  --render-config render.toml
```

程序只把共享 3DGS PLY 加载到 GPU 一次，然后依次渲染所有 episode。不同 episode 不并行；
每个 episode 内按 `camera_batch_size` 分批。

每个 episode 使用 staging 渲染目录。所有帧和 manifest 成功写入后才替换该 episode 的旧
`rendered_episode`，因此轨迹变短时不会残留旧 PNG，也不会影响其他 episode。

RGB 固定为 uint8，Depth 固定为 uint16、`10000 units/m`、无效值 0。

### 3. 多 episode → target scene

```bash
uv run robotnav-package-dataset --config dataset_build.toml
```

打包器先校验全部 episode，再在 scene 同级 staging 目录中生成完整数据。最终图片在 scene
范围内使用至少六位的连续全局编号：

```text
episode 000: 本地 000..099 -> 全局 000000..000099
episode 001: 本地 000..079 -> 全局 000100..000179
```

对应的 `episodes_stats.jsonl` 为：

```json
{"image_index":{"min":0,"max":99}}
{"image_index":{"min":100,"max":179}}
```

完整输出：

```text
<dataset_root>/<group>/<scene>/
├── data/chunk-000/
│   ├── 000.parquet
│   ├── 001.parquet
│   └── ...
├── videos/chunk-000/
│   ├── observation.images.rgb/
│   │   ├── 000000.png
│   │   └── ...
│   └── observation.images.depth/
│       ├── 000000.png
│       └── ...
└── meta/
    ├── episodes_stats.jsonl
    ├── pointcloud.ply
    └── run_manifest.json
```

只有 staging 目录通过独立目标格式验证后才会发布。如果目标 scene 已存在，则遵循
`[dataset].overwrite`。

## 最终验证

```bash
uv run robotnav-validate-dataset data/target/robotnav/scene-000
```

验证器不依赖构建时的上游文件，直接检查最终目录：

- parquet 数量非零且等于 stats 行数；
- stats 图片区间从 0 开始，连续、不重叠并完整覆盖图片；
- 每个 parquet 的行数等于对应 episode 图片数；
- intrinsic、extrinsic 和 action 的 shape、有限性正确；
- RGB/Depth 总数相同，格式、尺寸和编号对应；
- pointcloud 含目标障碍颜色。

报告返回 episode 总数、总帧数和逐 episode 的 parquet、图片区间与帧数。

## 总编排

以下命令按文件边界顺序执行三个阶段：

```bash
uv run robotnav-build-dataset \
  --config dataset_build.toml \
  --render-config render.toml
```

阶段之间只通过版本化文件交付。批量层不复制相机姿态算法或 gsplat 光栅化算法。
