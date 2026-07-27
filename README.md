# ZJUgive DataEngine for RobotNav

实验性点云、3D Gaussian 渲染和机器人导航轨迹工具集。

## 环境

项目使用 Python 3.10+，GPU 功能需要与本机 CUDA 匹配的 PyTorch 和 gsplat。

```bash
uv sync
uv sync --extra gpu
uv run env-check
```

环境检查是运行渲染和轨迹生成前的第一步，会检查 Python 依赖、CUDA、gsplat 后端和项目模块。

## 数据和输出

输入文件默认设置放在`data/input/`目录，例如：

```text
data/input/
├── try1-pointcloud-0706.las
└── try1_yup.ply
```

运行中间结果默认写入 `outputs/`下子目录中。

最终生成的数据集默认生成在`data/target/robotnav/`中。

大体积点云和生成结果不提交到 Git。

## 常用命令

### 准备工作：

准备导航需要的数据与轨迹：

```bash
uv run prepare-navigation-data
```

也可以分为三个阶段运行：

```bash
uv run build-scene            # 场景构建
uv run generate-trajectories  # 轨迹生成
uv run export-pointcloud      # 语义点云导出
```

场景构建、轨迹生成和 PLY 导出拥有各自的配置与输出目录。修改 A* 或批量参数只需重跑轨迹；
修改颜色或体素参数只需重跑 PLY。轨迹批次由
`outputs/trajectories/trajectory_manifest.json` 索引。

### 目标数据集构建

数据集构建分为将轨迹转化成相机位姿、用相机位姿渲染rgb图和深度图、打包数据三个阶段。三个阶段通过版本化文件交付，可以独立执行：

```bash
uv run trajectory-to-camera
uv run render-trajectory
uv run package-dataset
```

中间产物按轨迹 ID 隔离：

```text
outputs/dataset_build/episodes/<trajectory_id>/
├── camera_trajectory.npz
├── camera_trajectory.json
└── rendered_episode/
    ├── rgb/
    ├── depth/
    └── render_manifest.json
```

也可以顺序执行全部阶段：

```bash
uv run build-dataset --config dataset_build.toml --render-config render.toml
```

最终 scene 包含多个按顺序命名的 parquet，RGB/Depth 使用跨 episode 的连续全局编号，
`episodes_stats.jsonl` 记录每个 episode 的图片闭区间。语义点云默认位于
`outputs/semantic_pointcloud/pointcloud.ply`。

以下命令可独立验证最终 scene：

```bash
uv run validate-dataset data/target/robotnav/scene-000
```

### 静态检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

源码位于 `src/robotnav/`。导航职责进一步分为 `navigation/scene/`、
`navigation/trajectory/` 和 `navigation/semantic_pointcloud/`，CLI 位于 `commands/`。
目标数据集契约与多 episode 映射详见
[`docs/target_data.md`](docs/target_data.md) 和
[`docs/dataset_pipeline.md`](docs/dataset_pipeline.md)。
