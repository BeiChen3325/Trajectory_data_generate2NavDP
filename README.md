# ZJUgive DataEngine for RobotNav

实验性点云、3D Gaussian 渲染和机器人导航轨迹工具集。

## 环境

项目使用 Python 3.10+，GPU 功能需要与本机 CUDA 匹配的 PyTorch 和 gsplat。

```bash
uv sync
uv sync --extra gpu
uv run robotnav-env-check
```

环境检查是运行渲染和轨迹生成前的第一步，会检查 Python 依赖、CUDA、gsplat 后端和项目模块。

## 数据和输出

输入文件放在：

```text
data/input/
├── try1-pointcloud-0706.las
└── try1_yup.ply
```

运行结果统一写入：

```text
outputs/render/
outputs/navigation_scene/
outputs/trajectories/
outputs/semantic_pointcloud/
```

大体积点云和生成结果不提交到 Git。

## 常用命令

```bash
uv run robotnav-render
uv run robotnav-compare
uv run robotnav-build-scene --config navigation_scene.toml
uv run robotnav-generate-trajectories --config trajectories.toml
uv run robotnav-export-pointcloud --config pointcloud_export.toml
```

也可以用一个薄编排入口顺序执行三个阶段：

```bash
uv run robotnav-prepare-navigation-data \
  --scene-config navigation_scene.toml \
  --trajectory-config trajectories.toml \
  --pointcloud-config pointcloud_export.toml
```

场景构建、轨迹生成和 PLY 导出拥有各自的配置与输出目录。修改 A* 或批量参数只需重跑轨迹；
修改颜色或体素参数只需重跑 PLY。轨迹批次由
`outputs/trajectories/trajectory_manifest.json` 索引。

## 目标数据集构建

`configs/dataset_build.toml` 当前明确选择批次中的一条轨迹（默认 `auto_000.json`），并引用
独立生成的语义点云、工作目录和目标 scene。
三个阶段通过版本化文件交付，可独立执行：

```bash
uv run robotnav-trajectory-to-camera --config dataset_build.toml
uv run robotnav-render-trajectory --config dataset_build.toml --render-config render.toml
uv run robotnav-package-dataset --config dataset_build.toml
```

也可以顺序执行全部阶段：

```bash
uv run robotnav-build-dataset --config dataset_build.toml --render-config render.toml
```

语义点云默认位于 `outputs/semantic_pointcloud/pointcloud.ply`。以下命令可单独验证最终 scene：

```bash
uv run robotnav-validate-dataset data/target/robotnav/scene-000
```

## 静态检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

源码位于 `src/robotnav/`。导航职责进一步分为 `navigation/scene/`、
`navigation/trajectory/` 和 `navigation/semantic_pointcloud/`，CLI 位于 `commands/`。
