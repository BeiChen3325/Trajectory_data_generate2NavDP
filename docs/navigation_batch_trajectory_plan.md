# Navigation 多轨迹生成与代码重组计划

> 实施状态（2026-07-27）：阶段 A～D 已完成。场景构建、批量轨迹和语义 PLY 已拆为独立
> 包、配置、命令和输出契约；合成批量测试已覆盖一次生成 8 条确定性轨迹。阶段 E 的
> `SceneGeometry`/`TraversabilityMap` 细分与多 episode 数据集仍作为后续工作。

## 1. 背景与目标

当前 `robotnav-trajectory` 已经具备一条完整的轨迹生成链路：

1. 读取 LAS、裁剪 ROI、估计地面；
2. 构建 2.5D 占据地图、障碍膨胀和距离场；
3. 选择一个起点和目标点；
4. 执行 A*、shortcut、稠密采样和平滑；
5. 输出一条 `trajectory.json`、调试图、占据地图和语义点云。

现阶段的主要问题不是算法缺失，而是“场景级工作”和“单轨迹工作”耦合在
`src/robotnav/generate_trajectory.py` 中，导致：

- 一次命令只能自然地表达一组起终点；
- 地图、距离场和点云这些可复用产物与单条轨迹绑定；
- 后续增加批量采样、失败重试、轨迹去重和清单输出时，入口会继续膨胀；
- 配置中的地图参数、规划参数和单条任务参数混在同一个 `[navigation]` 表中；
- 下游只能通过固定的 `trajectory.json` 找到一条轨迹，缺少批次索引。

本次重组的目标是：

- 一次场景预处理后自动生成指定数量的不同轨迹；
- 同时支持全自动、固定起终点和“固定一端、自动选择另一端”；
- 最大限度复用已有地图构建、A*、shortcut、平滑、碰撞检查和点云导出代码；
- 将场景处理、轨迹生成和语义 PLY 导出做成可独立运行的阶段；
- 每个入口只负责参数解析和调用本阶段，不在 CLI 中放算法；
- 建立批次 manifest，为后续多 episode 渲染和数据集打包提供稳定入口；
- 保持随机过程可复现，并对无法生成足够多样轨迹的情况明确报错。

## 2. 可行性与必要性评估

### 2.1 结论

应该按目录和运行阶段明确拆成三部分：

1. **全局场景处理**：LAS → 版本化场景模型；
2. **轨迹生成**：场景模型 → 多条轨迹与轨迹 manifest；
3. **语义 PLY 导出**：LAS + 场景模型 → `pointcloud.ply`。

目录拆分是必要的，但只有目录拆分还不够。要实现“独立修改和独立重跑”，还必须同时具备：

- 独立配置文件；
- 独立命令入口；
- 稳定的中间文件契约；
- 输入来源与配置指纹校验。

建议保留一个可选的总编排命令，但它只能依次调用三个公开阶段，不能再维护另一套实现。

### 2.2 现有代码已经具备的基础

| 边界 | 当前真实依赖 | 可行性 | 必要性 |
|---|---|---:|---:|
| 场景处理独立 | LAS、ROI、地面/障碍和机器人几何参数 | 高 | 高 |
| 轨迹生成独立 | `planning_blocked`、`planning_distance_m`、栅格坐标和 `floor_y` | 高 | 高 |
| PLY 导出独立 | LAS、`cleaned_obstacles`、地面高度、体素和颜色配置 | 高 | 中高 |

具体依据：

- `scene_obstacles.py` 已提供带版本号的 `SceneObstacleModel`、
  `save_scene_obstacle_model()` 和 `load_scene_obstacle_model()`；
- 轨迹规划不需要重新读取 LAS，只需要场景模型中的规划栅格、距离场和坐标范围；
- `pointcloud_export.py` 不读取或依赖 `trajectory.json`，它只是重新流式读取 LAS，
  使用 `cleaned_obstacles` 分类真实点，再做体素降采样和着色；
- 因此当前主要耦合发生在 `generate_trajectory.py` 的编排层，而不是算法本身。

### 2.3 为什么 PLY 导出也值得独立

PLY 的颜色、是否包含上下文、障碍/上下文体素尺寸都可能频繁调整。这些变化：

- 不应触发场景地图重建；
- 不应重新生成轨迹；
- 不应改变 `planning_blocked`；
- 只需读取同一个场景模型并再次扫描 LAS。

独立后可以单独执行：

```bash
uv run robotnav-export-pointcloud --config pointcloud_export.toml
```

需要注意：PLY 导出不是完全无输入的后处理。为了保留真实 LAS 代表点，它仍需重新读取原始
LAS；但它不需要再次估计地面、构建栅格或执行轨迹规划。

### 2.4 合理依赖不等于错误耦合

三个阶段可以独立修改和重跑，但它们仍有必要的数据依赖：

```text
场景处理 ──scene_model.npz──> 轨迹生成
    │
    └──scene_model.npz + LAS──> 语义 PLY 导出
```

如果全局场景处理算法或机器人半径改变，旧轨迹在语义上已经失效，必须重新生成；
这属于正确的依赖，不能也不应该通过目录拆分隐藏。

相反，以下修改应当完全局部：

- A* 代价、端点采样和平滑参数改变：只重跑轨迹生成；
- PLY 颜色、体素大小和上下文开关改变：只重跑 PLY 导出；
- RGB-D 渲染参数改变：不重跑以上三个阶段。

### 2.5 暂不立即拆成两种场景契约

当前 `SceneObstacleModel` 同时包含：

- 来源于场景几何的 `cleaned_obstacles`；
- 来源于机器人尺寸的 `inflated_obstacles` 和 `planning_blocked`。

理论上可进一步拆为 `SceneGeometry` 和 `TraversabilityMap`。这样更换机器人半径时，
可以从已保存的几何栅格快速重新计算通行图，而无需重新扫描 LAS。

这项拆分可行，但第一轮不是必须。建议先把三个运行阶段拆开并继续复用当前版本化模型；
如果后续确实需要频繁切换机器人尺寸，再升级契约，避免第一轮同时改动过多数据结构。

## 3. 本次不做的事项

为了控制范围，第一阶段不包含：

- 更换 A* 或引入新的全局规划算法；
- 重新实现占据地图、地面估计、平滑或点云导出；
- 为每条轨迹重复构建地图或重复导出语义点云；
- 立即把 RGB-D 渲染和数据集打包改成并行批处理；
- 以路径聚类、覆盖率优化或学习方法追求全局最优的轨迹多样性；
- 为旧输出目录复制额外的兼容文件。
- 第一轮就把 `SceneObstacleModel` 拆成几何模型和机器人通行模型。

下游现有阶段仍可选择 manifest 中的一条轨迹运行；多 episode 数据集构建作为独立后续阶段。

## 4. 设计原则

### 4.1 场景只构建一次

LAS 读取、地面估计、栅格化、障碍清理、膨胀、距离场和语义点云都属于场景级产物。
无论生成 1 条还是 100 条轨迹，这些步骤在一次命令中只执行一次。

### 4.2 单轨迹规划保持纯粹

单轨迹规划器只接收：

- 已构建的规划场景；
- 已解析到自由栅格的起点和目标点；
- A* 与平滑参数；
- 本条轨迹的 seed。

它返回规划结果，不负责读取 LAS、解析 CLI、挑选下一组端点或决定批量输出目录。

### 4.3 批量层只负责任务管理

批量层负责：

- 生成或解析多组起终点；
- 去重和多样性约束；
- 为每条轨迹派生稳定 seed；
- 调用同一个单轨迹规划器；
- 对自动任务进行候选重试；
- 汇总并写入 manifest。

### 4.4 PLY 是场景派生产物，不是轨迹副作用

PLY 导出通过版本化场景模型读取物理障碍语义，但不从轨迹生成器中调用。总编排命令可以
顺序执行两个阶段，单独的轨迹命令不能隐式改写 PLY。

### 4.5 输出使用显式清单

不依赖“目录中第一个 JSON”或固定根目录 `trajectory.json`。所有轨迹由
`trajectory_manifest.json` 显式索引，并使用相对路径关联。

## 5. 建议的目录与代码职责

这里建议进行真实的包边界调整，而不只是从入口中再抽几个平级文件：

```text
src/robotnav/
├── commands/
│   ├── build_navigation_scene.py       # 场景处理 CLI
│   ├── generate_trajectories.py        # 多轨迹 CLI
│   ├── export_semantic_pointcloud.py   # PLY 导出 CLI
│   └── prepare_navigation_data.py      # 可选总编排，不含算法
└── navigation/
    ├── scene/
    │   ├── config.py
    │   ├── contracts.py                # SceneObstacleModel、SceneArtifact 及保存/加载
    │   ├── builder.py                  # LAS -> 场景模型
    │   ├── las_io.py
    │   ├── ground_estimation.py
    │   ├── occupancy_map.py
    │   └── visualization.py
    ├── trajectory/
    │   ├── config.py
    │   ├── contracts.py                # Task、Trajectory、Manifest
    │   ├── endpoint_sampler.py
    │   ├── astar.py
    │   ├── smoothing.py
    │   ├── planner.py                  # 单轨迹
    │   ├── batch.py                    # 多任务编排
    │   └── visualization.py
    └── semantic_pointcloud/
        ├── config.py
        ├── exporter.py
        └── voxel.py
```

这不是要求重写现有文件。优先采用移动、提取和改 import 的方式：

| 当前模块 | 目标位置 | 处理方式 |
|---|---|---|
| `las_io.py` | `scene/las_io.py` | 原样移动 |
| `ground_estimation.py` | `scene/ground_estimation.py` | 原样移动 |
| `occupancy_map.py` | `scene/occupancy_map.py` | 原样移动，坐标工具继续共享 |
| `scene_obstacles.py` | `scene/contracts.py` | 原样移动并补来源 metadata |
| `astar_planner.py` | `trajectory/astar.py` | 原样移动 |
| `path_smoothing.py` | `trajectory/smoothing.py` | 原样移动 |
| `pointcloud_export.py` | `semantic_pointcloud/exporter.py` | 拆出 voxel 类，算法不改 |
| `generate_trajectory.py` | 三个阶段 + 一个编排入口 | 提取现有代码，不保留第二套逻辑 |

如果希望降低第一批提交的 import 变化，也可以先创建目标目录和新模块，对旧算法模块暂时直接
import；测试稳定后再移动旧文件。最终状态应只有一份实现。

### 5.1 `navigation.scene`

公开接口：

```python
def build_scene(config: SceneBuildConfig) -> SceneObstacleModel:
    ...


def load_scene_artifact(scene_dir: Path) -> SceneArtifact:
    ...
```

职责：

- 读取 LAS、解析 ROI 和估计 `floor_y`；
- 构建物理障碍、可通行地面、膨胀障碍和规划距离场；
- 保存场景级调试图；
- 写入版本化 `scene_model.npz` 和 `scene_manifest.json`；
- 不选择起终点、不执行 A*、不导出 PLY。

`SceneArtifact` 将 NPZ 模型、manifest metadata 和模型 SHA-256 作为整体返回。所有下游只使用
这个公共加载器，不能绕过 manifest 单独加载 NPZ。这样 PLY 能读取场景构建时的
`ground_margin_m`，轨迹 manifest 也能记录准确的场景来源。

`SceneObstacleModel.spec` 已经提供栅格坐标信息，第一轮继续使用它，不必同步引入新的
`GridSpec` 数据类。

### 5.2 `navigation.trajectory`

单轨迹公开接口：

```python
def plan_trajectory(
    scene: SceneArtifact,
    task: TrajectoryTask,
    config: PlannerConfig,
) -> PlannedTrajectory:
    ...
```

批量公开接口：

```python
def plan_trajectory_batch(
    scene: SceneArtifact,
    config: TrajectoryBatchConfig,
) -> TrajectoryManifest:
    ...
```

内部直接复用现有 A*、shortcut、稠密采样、Chaikin 平滑和碰撞复检。该包：

- 只读取 `scene_model.npz`；
- 不读取 LAS；
- 不构建或改写场景模型；
- 不导出 PLY；
- 只写轨迹 JSON、轨迹调试图和轨迹 manifest。

### 5.3 `navigation.semantic_pointcloud`

公开接口：

```python
def export_semantic_pointcloud(
    las_path: Path,
    scene: SceneArtifact,
    config: PointCloudExportConfig,
) -> PointCloudReport:
    ...
```

该包复用当前 `classify_scene_points()`、`VoxelAccumulator` 和 PLY 写出逻辑。它：

- 读取原始 LAS 和 `scene_model.npz`；
- 只使用物理 `cleaned_obstacles`，不能使用膨胀障碍或未知区；
- 不读取轨迹 manifest；
- 不改写场景模型；
- 只写 PLY 和导出报告。

### 5.4 `commands`

四个入口都保持很薄：

```text
robotnav-build-scene
robotnav-generate-trajectories
robotnav-export-pointcloud
robotnav-prepare-navigation-data
```

`robotnav-prepare-navigation-data` 只是便利入口，依次调用前三个阶段的公开函数。某个阶段的
实现不能只存在于总入口中，否则独立命令和总命令会再次产生行为差异。

## 6. 配置模型

不要继续用一个 `trajectory.toml` 同时配置三个可独立阶段。建议拆成：

```text
configs/
├── navigation_scene.toml
├── trajectories.toml
└── pointcloud_export.toml
```

`navigation_scene.toml` 只包含 LAS 到场景模型所需参数：

```toml
[paths]
data_dir = "data/input"
las_filename = "try1-pointcloud-0706.las"
output_dir = "outputs/navigation_scene"

[scene]
# 当前地面估计、ROI、分辨率、形态学和流式读取参数

[robot]
radius_m = 0.25
height_m = 0.8
ground_margin_m = 0.06
safety_margin_m = 0.10
```

`trajectories.toml` 不再声明 LAS，只读取场景文件：

```toml
[paths]
scene_dir = "outputs/navigation_scene"
output_dir = "outputs/trajectories"

[planner]
obstacle_cost_weight = 0.8
obstacle_cost_power = 1.5
shortcut_passes = 120
smooth_samples_per_meter = 8.0

[trajectory_batch]
count = 8
seed = 7
min_start_goal_distance_m = 3.0
min_endpoint_separation_m = 0.5
max_sampling_attempts = 200
manifest_filename = "trajectory_manifest.json"
requests = []
```

`pointcloud_export.toml` 只声明 PLY 输入输出和导出策略：

```toml
[paths]
data_dir = "data/input"
las_filename = "try1-pointcloud-0706.las"
scene_dir = "outputs/navigation_scene"
output_dir = "outputs/semantic_pointcloud"

[pointcloud]
# 复用当前 enabled 之外的文件名、颜色、体素尺寸和 include_context 字段
```

显式任务使用 TOML 内联表或数组表。建议最终采用数组表，较易阅读：

```toml
[[trajectory_batch.requests]]
id = "door_to_desk"
start_xz = [-2.0, 1.0]
goal_xz = [3.0, 4.0]

[[trajectory_batch.requests]]
id = "desk_to_auto"
start_xz = [3.0, 4.0]
goal_xz = []
```

规则：

- `count` 表示最终需要生成的轨迹总数；
- `requests` 可以为空，也可以少于 `count`；
- 显式任务不足时用自动任务补齐；
- `start_xz = []` 或 `goal_xz = []` 表示自动选择该端；
- `requests` 数量大于 `count` 时配置校验失败；
- ID 必须唯一且只能是普通文件名安全字符；
- 所有自动随机过程从 `seed` 确定性派生。

配置类不再组合成一个跨阶段的 `NavigationConfig`，而是三个顶层类型：

```python
@dataclass(frozen=True)
class SceneBuildConfig:
    scene: SceneConfig
    robot: RobotConfig


@dataclass(frozen=True)
class TrajectoryGenerationConfig:
    planner: PlannerConfig
    batch: BatchConfig


@dataclass(frozen=True)
class PointCloudExportConfig:
    pointcloud: PointCloudConfig
```

每个类型再组合自己所需的 path 类型。这样 PLY 配置不可能误传给轨迹规划器，
轨迹参数也不会参与场景模型的配置哈希。

## 7. 多起终点自动生成策略

### 7.1 第一阶段策略：复用现有选择器

第一阶段不重新设计采样算法。对每个自动任务：

1. 从 `planning_blocked` 得到自由空间；
2. 使用现有 `largest_free_component()` 限制起终点位于同一连通区域；
3. 使用现有 `choose_auto_start_goal()` 生成候选；
4. 每次尝试使用不同但确定性的派生 seed；
5. 将世界坐标输入用 `nearest_free_cell()` 吸附到自由栅格；
6. 检查端点距离、多样性和已使用端点；
7. 不满足约束则继续采样，达到 `max_sampling_attempts` 后报错。

派生 seed 不使用全局随机状态，例如：

```text
route_seed = batch_seed + route_index * 1009 + attempt_index
```

### 7.2 “不同轨迹”的最低定义

第一阶段要求：

- `(start_cell, goal_cell)` 组合不能重复；
- 新轨迹起点与已有自动轨迹起点至少相距 `min_endpoint_separation_m`；
- 新轨迹目标点与已有自动轨迹目标点至少相距 `min_endpoint_separation_m`；
- 起终点之间至少相距 `min_start_goal_distance_m`；
- 起终点均属于同一个自由空间连通分量；
- 最终平滑路径通过碰撞复检。

对于用户显式指定的端点，吸附后的重复或距离不足应直接给出清晰错误，不静默修改任务。

### 7.3 失败处理

- 自动候选不合法：重试候选，不立即终止整个批次；
- 显式起终点非法：立即失败，并指出任务 ID 和吸附后的坐标；
- A* 对自动候选失败：记录原因并换一组候选；
- 达到最大尝试次数仍不足 `count`：命令以非零状态结束，报告已完成数量、主要拒绝原因和建议；
- 不以悄悄减少输出数量的方式“成功”。

### 7.4 后续可选增强

如果仅靠端点分离仍产生大量相似路线，再增加第二层多样性判断：

- 路径长度区间采样；
- 路径栅格 Jaccard overlap 上限；
- 按自由空间区域或方向象限分桶；
- 对候选端点使用最短路距离而不是欧氏距离；
- 根据地图覆盖率选择下一条轨迹。

这些增强应建立在批量层之上，不修改 A* 核心。

## 8. 输出结构与文件契约

建议输出：

```text
outputs/
├── navigation_scene/
│   ├── scene_model.npz
│   ├── scene_manifest.json
│   ├── floor_estimation_report.json
│   └── debug/
├── trajectories/
│   ├── trajectory_manifest.json
│   └── routes/
│       ├── auto_000.json
│       ├── auto_000_debug.png
│       ├── auto_001.json
│       ├── auto_001_debug.png
│       ├── door_to_desk.json
│       └── door_to_desk_debug.png
└── semantic_pointcloud/
    ├── pointcloud.ply
    └── pointcloud_report.json
```

三个阶段不能写入彼此的输出目录。这能防止只重跑 PLY 时覆盖轨迹文件，或重新生成轨迹时
误删场景调试产物。

### 8.1 场景契约

`scene_model.npz` 第一轮直接沿用当前 `SceneObstacleModel` 的版本化数组契约。
新增 `scene_manifest.json` 保存：

- contract version；
- 源 LAS 路径、文件大小和内容指纹；
- ROI、坐标变换和 `floor_y`；
- 场景与机器人配置；
- `scene_model.npz` 的 SHA-256；
- 创建本模型的命令版本。

PLY 导出和轨迹生成都校验 scene contract。PLY 阶段还要确认当前 LAS 指纹与
`scene_manifest.json` 相符，防止使用另一份同名点云导出错误语义。

### 8.2 轨迹契约

每条轨迹 JSON 尽量保持当前字段不变，只新增：

```json
{
  "trajectory_id": "auto_000",
  "seed": 7,
  "coordinate_convention": "...",
  "floor_y": 0.247,
  "start_xz": [-2.0, 1.0],
  "goal_xz": [3.0, 4.0],
  "astar_path_xz": [],
  "shortcut_path_xz": [],
  "smooth_path_xz": [],
  "smooth_path_collides": false
}
```

这使现有 `trajectory_to_camera.py` 仍能读取任意一条被选中的 JSON，因为它只依赖
`floor_y`、`smooth_path_xz` 和 `coordinate_convention`。

manifest 建议包含：

```json
{
  "contract_version": 1,
  "requested_count": 8,
  "trajectory_count": 8,
  "batch_seed": 7,
  "source_scene_model": "../navigation_scene/scene_model.npz",
  "source_scene_model_sha256": "...",
  "trajectories": [
    {
      "trajectory_id": "auto_000",
      "path": "routes/auto_000.json",
      "debug_image": "routes/auto_000_debug.png",
      "start_xz": [-2.0, 1.0],
      "goal_xz": [3.0, 4.0],
      "path_length_m": 6.8,
      "point_count": 55,
      "smooth_path_collides": false
    }
  ]
}
```

manifest 中只写相对路径，并保存源场景文件哈希。目录整体移动后相对路径仍有效，
场景文件内容改变时则明确拒绝继续复用旧轨迹。

### 8.3 PLY 报告契约

`pointcloud_report.json` 在当前报告字段基础上新增：

- `source_scene_model` 和 SHA-256；
- `source_las` 和来源指纹；
- 本次点云配置；
- 障碍、上下文候选点和代表点数量。

## 9. CLI 规划

三个阶段可分别运行：

```bash
uv run robotnav-build-scene --config navigation_scene.toml
uv run robotnav-generate-trajectories --config trajectories.toml
uv run robotnav-export-pointcloud --config pointcloud_export.toml
```

便利入口显式接收三个配置：

```bash
uv run robotnav-prepare-navigation-data \
  --scene-config navigation_scene.toml \
  --trajectory-config trajectories.toml \
  --pointcloud-config pointcloud_export.toml
```

约定：

- 场景命令不接受轨迹或 PLY 专属参数；
- 轨迹命令不接受 `--las`；
- PLY 命令不接受 A* 或批次数量参数；
- 同时传 `--start-xz` 和 `--goal-xz` 时，表示一次性的单任务运行，并将本次 `count` 设为 1；
- 只传一端时，另一端自动选择；
- 复杂的多组固定任务只写 TOML，不在 CLI 中发明难维护的重复参数语法；
- 其他低频算法参数继续通过 TOML 调整。

## 10. 与下游数据集流水线的衔接

当前 `dataset_build.toml` 仍指向一个具体的轨迹文件。Navigation 批量化完成后，先采用最小改动：

```toml
[paths]
trajectory_dir = "outputs/trajectories/routes"
trajectory_filename = "auto_000.json"
```

因此第一阶段不会阻塞现有单 episode 的相机转换、渲染和打包。

后续另开计划，让数据集流水线读取 `trajectory_manifest.json`，为每条轨迹建立独立
episode 工作目录，并避免多个 episode 互相覆盖 `camera_trajectory.npz` 和
`rendered_episode/`。

## 11. 分阶段实施顺序

### 阶段 A：建立独立场景契约和命令

1. 将当前 Stage 1～3 提取到 `navigation.scene`；
2. 复用并加强 `SceneObstacleModel` 保存/加载校验；
3. 增加来源与配置 `scene_manifest.json`；
4. 建立 `robotnav-build-scene`；
5. 验证保存后重新加载的数组与当前内存结果一致。

完成标准：场景构建可以单独运行，之后的步骤不需要重新读取和构建地图。

### 阶段 B：建立独立单轨迹与批量规划

1. 提取单轨迹 `plan_trajectory()` 并复用现有算法；
2. 让它只接收加载后的 `SceneObstacleModel`；
3. 增加 `BatchConfig`、端点采样、去重和重试；
4. 建立 `robotnav-generate-trajectories`；
5. 输出每条轨迹和带场景哈希的 manifest。

完成标准：配置 `count = N` 时稳定产生恰好 N 条合法且端点不同的轨迹。

### 阶段 C：独立 PLY 导出

1. 将现有 `pointcloud_export.py` 移入独立职责目录；
2. 建立 `pointcloud_export.toml` 和 `robotnav-export-pointcloud`；
3. 从磁盘加载 scene contract 并校验 LAS 来源；
4. 验证只调整体素或颜色不会改写场景与轨迹输出。

完成标准：PLY 可以在完全不执行轨迹代码的情况下独立重新导出。

### 阶段 D：目录迁移、总编排和文档清理

1. 将已稳定的现有算法模块移动到目标子包；
2. 增加只调用三个阶段的便利编排入口；
3. 更新 README、默认配置和数据集单轨迹选择；
4. 删除旧入口和旧配置，不保留重复实现；
5. 运行全套检查。

完成标准：三个阶段既可独立执行，也可由同一个编排命令顺序执行，结果一致。

### 阶段 E：可选的契约细分与多 episode

根据实际需要再拆 `SceneGeometry`/`TraversabilityMap`，以及让数据集流水线消费全部轨迹。

## 12. 测试与验收标准

### 12.1 单元测试

- 配置可加载默认自动批次；
- 固定、半固定、全自动任务均能解析；
- 重复 ID、非法坐标、请求数超过 count 会失败；
- 同一 seed 和同一地图生成完全相同的任务端点；
- 不同 route seed 不共享随机状态；
- 所有自动任务起终点位于自由空间同一连通分量；
- 端点距离和端点分离约束生效；
- 无足够自由空间时在有限尝试后失败；
- 单轨迹 A*、shortcut、平滑和碰撞回退行为保持不变；
- manifest 中路径均为相对路径且文件存在。
- 轨迹配置无法访问 LAS 字段，PLY 配置无法访问规划字段；
- scene manifest、trajectory manifest 和 PLY report 的哈希链一致；
- PLY 使用了不同 LAS 时校验失败。

### 12.2 集成测试

使用小型合成占据栅格，不依赖真实 LAS：

- `count = 8` 得到 8 条轨迹；
- 8 组吸附后端点均满足唯一性和间距约束；
- 每条路径首尾与任务端点一致；
- 每条最终路径不穿过 `planning_blocked`；
- 连续运行两次产生一致的轨迹 JSON 和 manifest 内容。

真实 LAS 做一次低成本烟雾测试：

- 场景只生成一次，轨迹和 PLY 分别读取同一个 `scene_model.npz`；
- 只重跑轨迹不会改变场景模型和 PLY；
- 只重跑 PLY 不会改变场景模型和轨迹；
- `trajectory-to-camera` 能读取其中任意一条轨迹；
- 输出目录中不再依赖根目录固定 `trajectory.json`。

### 12.3 工程检查

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
```

## 13. 主要风险与控制方式

### 自由区域太小，无法满足轨迹数量

通过 `max_sampling_attempts` 限制运行时间，并明确报告可用连通区域、已生成数量和被拒原因。
不自动降低距离阈值。

### 多条轨迹端点不同但路径高度相似

第一阶段先保证端点多样性；用 manifest 中的路径长度等统计观察实际结果，再决定是否增加
路径 overlap 约束，避免提前实现复杂采样器。

### 配置拆分导致一次性改动过大

按阶段 A～D 迁移。先通过新接口复用旧算法文件，测试稳定后再移动目录；
不要在同一提交中同时移动模块、改变算法和升级中间契约。

### 下游固定读取 `trajectory.json`

轨迹文件本身保持原字段契约；下游第一阶段通过配置明确选择一条文件。批量 episode 改造单独进行。

### 场景模型与 LAS 来源不一致

使用 scene manifest 保存来源指纹，PLY 导出前强制校验。轨迹生成同时校验 scene model 哈希，
防止 manifest 与场景内容错配。

## 14. 预期结果

完成后，Navigation 的核心调用关系为：

```text
LAS ──build_scene()──> scene_model.npz + scene_manifest.json
                          │
                          ├──generate_trajectories()
                          │    ├──route 0
                          │    ├──route 1
                          │    └──trajectory_manifest.json
                          │
LAS ─────────────────────└──export_semantic_pointcloud()
                               ├──pointcloud.ply
                               └──pointcloud_report.json
```

这套结构保留现有算法资产，同时在代码目录、配置、命令和文件契约四个层面建立一致边界。
全局处理改变时，下游能明确知道需要失效；只改变轨迹或 PLY 参数时，则可以安全地局部重跑。
