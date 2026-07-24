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
uv run pytest -q
```
## 数据集构建与语义点云状态（2026-07-24）

面向 `docs/target_data.md` 的首版模块化流水线已经实现。数据集配置入口为
`configs/dataset_build.toml`，三个数据集阶段只通过固定中间文件交换数据：

1. `robotnav-trajectory-to-camera`：将 `trajectory.json` 转换为 contract version 1 的
   `camera_trajectory.npz/json`；
2. `robotnav-render-trajectory`：生成 RGB、uint16 metric Depth 和 `render_manifest.json`；
3. `robotnav-package-dataset`：校验并打包相机轨迹、渲染产物和语义点云。

`robotnav-trajectory` 现在同时生成 `trajectory.json`、`occupancy_map.npz`、
`pointcloud.ply` 和 `pointcloud_report.json`。`pointcloud.ply` 是轨迹阶段的内生产物，默认位于
`outputs/trajectory/`；不得重新视为 `data/input/` 中的外生文件。打包配置只引用并复制它。

## 障碍模型约定

轨迹规划与语义点云共享坐标变换、地面估计和未膨胀的 `cleaned_obstacles`，但派生语义必须
保持隔离：

- 规划分支只统计 `ground_margin_m` 到 `robot_height_m` 的碰撞高度带，并使用机器人半径、
  安全边距和未知区生成 `planning_blocked`；
- 语义点云分支保留 `cleaned_obstacles` 足迹内、地面间隙以上的完整高度 LAS 表面；不得使用
  `inflated_obstacles` 或把未知区伪装成物理刚体；
- 体素降采样必须保留距体素中心最近的真实 LAS 代表点，禁止把人工体素中心写入 PLY；
- 默认障碍色为 `(0, 0, 128)`，上下文色为 `(192, 192, 192)`；障碍体素 2 cm，上下文体素
  5 cm，并默认包含上下文；
- 当前内部坐标为 Y-up 表达，但物理向下是 `+Y`，离地高度统一计算为
  `floor_y - y`。不得对语义点云单独居中、归一化或缩放。

详细设计和验收规则见 `docs/pointcloud_plan.md`。

## 当前验证结果与下一步

- 真实 15,439,148 点 LAS 已完成端到端轨迹与点云生成：轨迹 124 点且碰撞复检为 false；
- 完整高度障碍候选为 3,326,094 点，输出 219,649 个障碍代表点和 90,244 个上下文代表点；
- 最终 PLY 共 309,893 点、约 4.5 MB，Y 坐标有 2,853 个不同值，并已通过语义点云校验；
- 当前单元测试 8 个全部通过，`ty check` 与本次相关 Ruff 检查通过；
- 既有 RGB-D manifest 绑定的是旧 `trajectory.json` 哈希。正式打包前必须重新运行轨迹到相机、
  RGB-D 渲染和打包阶段，不能复用旧 manifest 绕过哈希完整性检查；
- 全仓 `uv run ruff check .` 仍会命中既有的
  `src/robotnav/rendering/render_one_view.py:35` `B904` 告警。
