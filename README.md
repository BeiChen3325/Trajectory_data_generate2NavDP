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
outputs/trajectory/
```

大体积点云和生成结果不提交到 Git。

## 常用命令

```bash
uv run robotnav-render
uv run robotnav-compare
uv run robotnav-trajectory
```

也可以直接传入命令行参数覆盖默认参数。命令行参数优先于 TOML 配置和代码默认值。

## 目标数据集构建

`configs/dataset_build.toml` 配置既有轨迹、语义点云、工作目录和目标 scene。
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

打包阶段要求 `dataset_build.toml` 指定的标准彩色 `pointcloud.ply` 已存在，并可用
以下命令单独验证最终 scene：

```bash
uv run robotnav-validate-dataset data/target/robotnav/scene-000
```

## 静态检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

源码位于 `src/robotnav/`，渲染代码位于 `rendering/`，导航和轨迹代码位于 `navigation/`。
