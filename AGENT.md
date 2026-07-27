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
uv run robotnav-env-check
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
```
## Navigation 三阶段状态（2026-07-27）

Navigation 已从单体轨迹入口拆为三个可独立执行的阶段：

1. `robotnav-build-scene --config navigation_scene.toml`：LAS →
   `outputs/navigation_scene/scene_model.npz` 与 `scene_manifest.json`；
2. `robotnav-generate-trajectories --config trajectories.toml`：场景产物 →
   `outputs/trajectories/routes/*.json` 与 `trajectory_manifest.json`；
3. `robotnav-export-pointcloud --config pointcloud_export.toml`：LAS + 场景产物 →
   `outputs/semantic_pointcloud/pointcloud.ply` 与报告。

`robotnav-prepare-navigation-data` 只是顺序调用以上公开阶段的便利编排器，禁止在其中维护
另一套算法。旧 `robotnav-trajectory` 和 `configs/trajectory.toml` 已移除。

所有下游必须通过 `load_scene_artifact()` 同时加载 NPZ 和 manifest，不能绕过哈希校验直接读取
`scene_model.npz`。PLY 导出还必须校验当前 LAS 与 scene manifest 中的大小和 SHA-256 一致。
轨迹批次要求准确生成 `count` 条；无法满足端点距离、分离或连通性约束时必须失败，不能静默减量。

## 数据集与 trajectory→img 衔接

现有数据集流水线仍是单 episode：`configs/dataset_build.toml` 默认选择
`outputs/trajectories/routes/auto_000.json`，语义点云来自独立目录
`outputs/semantic_pointcloud/`。route JSON 保持 `floor_y`、`smooth_path_xz` 和
`coordinate_convention` 字段，因此现有相机转换可以复用。

后续多 episode 改造必须从 `trajectory_manifest.json` 枚举轨迹，为每个 `trajectory_id` 使用独立
工作目录，并校验每条 `trajectory_sha256`。不得为批处理复制相机姿态或渲染算法。实施边界、配置
建议和验收项见 `docs/trajectory_to_img_migration_guide.md`。

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

- Navigation 的合成测试覆盖同一 seed 稳定生成 8 条不同端点轨迹、显式重复任务失败和配置校验；
- 2026-07-27 已通过真实 LAS 三阶段烟雾测试、场景/轨迹/PLY 哈希链校验、12 个单元测试、
  全仓 Ruff 格式与 lint、`ty check`；现有 trajectory-to-camera 成功读取 `auto_000.json`
  并生成 116 个相机位姿；
- 既有 RGB-D manifest 绑定的是旧 `trajectory.json` 哈希。正式打包前必须重新运行轨迹到相机、
  RGB-D 渲染和打包阶段，不能复用旧 manifest 绕过哈希完整性检查；
- 多 episode trajectory→img 尚未实现；改造时遵循上述指引。
