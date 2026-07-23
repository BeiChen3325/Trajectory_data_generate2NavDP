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

## 静态检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

源码位于 `src/robotnav/`，渲染代码位于 `rendering/`，导航和轨迹代码位于 `navigation/`。
