# RobotNav 项目协作约定

这是一个实验性点云、3D Gaussian 渲染和机器人导航数据工程项目。优先保持代码直接、可运行和容易修改，不为旧目录、旧命令或旧参数保留兼容层。

## 目录职责

- `src/robotnav/`：项目源码。
- `src/robotnav/rendering/`：渲染和点云检查。
- `src/robotnav/navigation/`：地图、规划和轨迹后处理。
- `configs/`：默认 TOML 参数。
- `data/input/`：本地输入数据，不提交大型点云文件。
- `outputs/`：运行生成的结果，不提交实验输出。

## 配置约定

- `configs/*.toml` 是运行参数和路径配置的唯一事实源。代码不得硬编码数据目录、输出目录、LAS/PLY 文件名或完整默认路径。
- 路径配置目前拆分为 `data_dir`、`output_dir`、`las_filename` 和可选的 `ply_filename`，禁止把目录和文件名混在单个路径字段中。
- `src/robotnav/config.py` 负责定位项目根目录、读取 TOML 和组合 `Path`；业务模块必须通过 `load_path_config()` 或其返回对象读取路径。
- 不得在渲染模块、导航模块或命令入口中重新定义第二套路径常量；路径工具也必须集中在 `config.py`。
- 新增命令时，应为其增加对应的 `configs/<command>.toml`，并让命令入口从该 TOML 获取默认路径。

## 开发约定

- 使用 `pathlib.Path`，不要写死机器相关绝对路径。
- 命令行入口负责参数处理，算法函数接收明确的配置或参数。
- 随机过程必须使用显式 seed。
- 新代码使用 Python 3.10+ 类型标注。
- 先运行环境检查，再运行 GPU 相关流程。

## 验证

```bash
uv run robotnav-env-check
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## 数据集构建进展（2026-07-24）

面向 `docs/target_data.md` 的首版模块化流水线已经实现，配置入口为
`configs/dataset_build.toml`。新增代码位于 `src/robotnav/dataset/`，三个阶段只通过固定
中间文件交换数据，禁止绕过文件契约直接传递阶段内部对象：

1. `robotnav-trajectory-to-camera`：读取现有 `trajectory.json`，生成 contract version 1
   的 `camera_trajectory.npz` 和 `camera_trajectory.json`；
2. `robotnav-render-trajectory`：读取相机轨迹文件和 `render.toml`，生成 RGB、uint16
   metric Depth 与 `render_manifest.json`；
3. `robotnav-package-dataset`：读取相机轨迹、渲染产物和已有语义点云，写 Parquet、
   `episodes_stats.jsonl`、`pointcloud.ply` 和 `run_manifest.json`。

另外提供：

- `robotnav-build-dataset`：按独立子进程依次运行三个阶段，不省略中间文件；
- `robotnav-validate-dataset`：单独验证已打包的目标 scene；
- `tests/test_dataset_build.py`：覆盖配置、重复路径点、位姿互逆和目标目录打包；
- `pandas`、`pyarrow` 已加入正式依赖并更新 `uv.lock`。

当前已使用仓库中的真实 `trajectory.json` 和 3DGS PLY 完成 GPU smoke test：生成 124 组
`640×480` RGB/Depth，文件位于 `outputs/dataset_build/`。新增范围 Ruff/format、全部测试
和 `ty check` 均通过。

## 数据集构建当前缺失与下一步

- 缺少默认配置指定的 `data/input/pointcloud.ply`。它必须是 Open3D 可读的标准彩色点云，
  且障碍点颜色与 `(0,0,0.5)` 的距离小于 `0.05`。该文件属于可替换的上游语义点云产物，
  当前流水线只校验和复制，不从 LAS 临时生成。
- 因上述语义点云尚未提供，真实最终目录
  `data/target/robotnav/scene-000/` 尚未执行打包；合成输入下的完整打包测试已经通过。
- 提供语义点云后，先运行
  `uv run robotnav-package-dataset --config dataset_build.toml`；如需从第一阶段重跑全部流程，
  使用 `uv run robotnav-build-dataset --config dataset_build.toml --render-config render.toml`。
- 全仓 `uv run ruff check .` 仍会命中既有的
  `src/robotnav/rendering/render_one_view.py:35` `B904` 告警；该问题不在本次新增模块中。
