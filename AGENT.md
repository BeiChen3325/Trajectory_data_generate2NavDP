# RobotNav 项目协作约定

这是一个实验性点云、3D Gaussian 渲染和机器人导航数据工程项目。优先保持代码直接、可运行和容易修改，不为旧目录、旧命令或旧参数保留兼容层。

## 目录职责

- `src/robotnav/`：项目源码。
- `src/robotnav/rendering/`：渲染和点云检查。
- `src/robotnav/navigation/scene/`：LAS 读取、地面估计、占据地图和版本化场景产物。
- `src/robotnav/navigation/trajectory/`：端点采样、单轨迹规划、批量编排和轨迹可视化。
- `src/robotnav/navigation/semantic_pointcloud/`：基于场景物理障碍语义的独立 PLY 导出。
- `src/robotnav/commands/`：薄 CLI 入口，不放算法。
- `configs/`：默认 TOML 参数。
- `data/input/`：本地输入数据，不提交大型点云文件。
- `outputs/`：运行生成的结果，不提交实验输出。

## 配置约定

- `configs/*.toml` 是运行参数和路径配置的唯一事实源。代码不得硬编码数据目录、输出目录、LAS/PLY 文件名或完整默认路径。
- Navigation 使用三个互不交叉的配置：`navigation_scene.toml`、`trajectories.toml` 和
  `pointcloud_export.toml`。轨迹配置不得声明 LAS，PLY 配置不得声明规划参数。
- 路径字段区分目录与文件名；相对路径统一相对项目根目录解析，禁止硬编码机器绝对路径。
- `src/robotnav/config.py` 负责项目根目录和基础 TOML 工具，各阶段的强类型配置加载器位于其
  职责子包的 `config.py`。
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
uv run env-check
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
```
## Navigation 三阶段状态（2026-07-27）

Navigation 已从单体轨迹入口拆为三个可独立执行的阶段：

1. `build-scene --config navigation_scene.toml`：LAS →
   `outputs/navigation_scene/scene_model.npz` 与 `scene_manifest.json`；
2. `generate-trajectories --config trajectories.toml`：场景产物 →
   `outputs/trajectories/routes/*.json` 与 `trajectory_manifest.json`；
3. `export-pointcloud --config pointcloud_export.toml`：LAS + 场景产物 →
   `outputs/semantic_pointcloud/pointcloud.ply` 与报告。

`prepare-navigation-data` 只是顺序调用以上公开阶段的便利编排器，禁止在其中维护
另一套算法。旧 `trajectory` 和 `configs/trajectory.toml` 已移除。

所有下游必须通过 `load_scene_artifact()` 同时加载 NPZ 和 manifest，不能绕过哈希校验直接读取
`scene_model.npz`。PLY 导出还必须校验当前 LAS 与 scene manifest 中的大小和 SHA-256 一致。
轨迹批次要求准确生成 `count` 条；无法满足端点距离、分离或连通性约束时必须失败，不能静默减量。

## 数据集与 trajectory→img 衔接

数据集流水线从 `trajectory_manifest.json` 枚举全部轨迹，每个 `trajectory_id` 使用独立
工作目录。加载时必须校验 route SHA-256 与 scene SHA；相机、渲染 manifest 继续把上游哈希
向下传递。批量层不得复制相机姿态或 gsplat 算法。

一个 trajectory manifest 对应一个 target scene，每条 route 对应一个 parquet episode。
RGB/Depth 在最终 scene 内使用连续全局编号，`episodes_stats.jsonl` 的行序必须与排序后的
parquet 一致。具体契约见 `docs/dataset_pipeline.md` 和 `docs/target_data.md`。

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

语义点云 report 必须记录 scene model SHA 和生成的 PLY SHA。打包前要同时校验
trajectory manifest、pointcloud report 与当前 PLY，不能复用旧产物绕过完整性检查。
