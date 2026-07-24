# 面向 `target_data.md` 的模块化数据流水线计划

## 1. 设计目标

本计划将轨迹规划产生的共享障碍模型继续转换为目标数据集需要的
`pointcloud.ply`，不另建第二套障碍识别；3DGS 场景构建仍作为已有能力。

核心任务有四个：

1. 将现有轨迹转换为逐帧相机位姿；
2. 根据相机位姿渲染 RGB 和 Depth；
3. 由轨迹阶段的共享障碍模型生成二值刚体障碍 `pointcloud.ply`；
4. 将位姿、图像和生成的点云整理成 `target_data.md` 要求的目录。

各阶段是可独立运行、可单独替换的命令。阶段之间只交付约定文件，不直接调用上一阶段
的 Python 接口，也不依赖上一阶段的内部数据结构。现有代码只在各阶段内部复用。

```text
上游已有产物
├── trajectory.json
├── 3DGS PLY
└── 共享障碍模型生成的 pointcloud.ply
          │
          ▼
[阶段 1] trajectory -> camera poses
          │  camera_trajectory.npz
          │  camera_trajectory.json
          ▼
[阶段 2] camera poses -> RGB-D
          │  rgb/*.png
          │  depth/*.png
          │  render_manifest.json
          ▼
[阶段 3] package target dataset
          │
          ▼
<root>/<group>/<scene>/{data,videos,meta}
```

## 2. 对文件解耦方案的评估

### 2.1 采用该方案的原因

固定中间文件比阶段间直接调用接口更适合后续替换算法：

- 可以独立替换轨迹适配、相机朝向算法或渲染器；
- 可以只重跑发生变化的阶段；
- 中间结果可直接检查和保存，问题定位更清楚；
- 新实现可以使用不同语言或运行环境，只需满足文件契约；
- GPU 渲染阶段与无 GPU 的轨迹适配、打包阶段自然隔离；
- 下游阶段不必跟随上游 Python dataclass 或函数签名变化。

### 2.2 需要接受的代价

- 中间文件会带来少量磁盘和序列化开销；
- 文件契约一旦发布就需要版本管理；
- 仅靠文件存在不能证明内容正确，因此每个阶段都要做输入校验并写 manifest；
- 编排命令不能通过内存传递隐藏信息，所有下游需要的信息必须进入正式契约。

这些代价对当前数据生成任务是可控的。位姿文件很小，RGB-D 本来就必须落盘；换来的
可替换性和可调试性更重要。因此本计划采用文件解耦方案。

### 2.3 独立性的具体约束

- 阶段 2 不能导入阶段 1 的业务模块，只读取 `camera_trajectory.npz/json`；
- 阶段 3 不能导入阶段 1/2 的业务模块，只读取约定文件；
- 各阶段可以复用仓库已有的通用函数，但不得依赖其他新阶段的内部实现；
- manifest 只记录事实和校验信息，不能成为必须回读的隐式配置源；
- 可选总命令只负责按顺序启动三个阶段，不能绕过中间文件直接传内存对象。

## 3. 输入、输出与范围

### 3.1 上游必须提供的输入

| 输入 | 最低要求 | 生产是否属于本计划 |
| --- | --- | --- |
| `trajectory.json` | 含 `floor_y`、`smooth_path_xz` 和坐标约定 | 否；继续由现有 `robotnav-trajectory` 生成 |
| 3DGS PLY | 可由现有 `load_ply_to_torch` 读取，坐标与轨迹一致 | 否 |
| 二值障碍 `pointcloud.ply` | Open3D 可读，障碍颜色满足 `(0,0,0.5)` 距离小于 `0.05` | 由轨迹阶段共享障碍模型生成 |
| `configs/render.toml` | 3DGS 路径、图像尺寸、内参来源和渲染设置 | 已有配置，按需扩展原 schema |

`pointcloud.ply` 不是放在 `data/input` 的外生文件。它与 `trajectory.json` 共用一次
LAS 场景预处理和障碍识别，默认写到 `outputs/trajectory/pointcloud.ply`。打包阶段只
校验并复制该正式产物，不从调试 PNG 反推点云。详细方案见 `pointcloud_plan.md`。

### 3.2 最终输出

```text
<dataset_root>/<group_dir>/<scene_dir>/
├── data/
│   └── chunk-000/
│       └── 000.parquet
├── videos/
│   └── chunk-000/
│       ├── observation.images.rgb/
│       │   ├── 000.png
│       │   └── ...
│       └── observation.images.depth/
│           ├── 000.png
│           └── ...
└── meta/
    ├── episodes_stats.jsonl
    ├── pointcloud.ply
    └── run_manifest.json
```

第一版仍固定一个 scene、一个 `chunk-000` 和一个 `000` episode。

### 3.3 明确不属于核心范围的工作

- LAS 到 2.5D 地图、A* 和轨迹平滑；
- 3DGS PLY 的训练或坐标对齐；
- 对现有轨迹规划主流程进行大规模库化重构；
- 多 episode 调度、随机批量轨迹和质量优化。

这些能力可以有独立工具，但不能成为完成数据集转换三个阶段的前置开发任务。

## 4. 新增配置文件及参数归属

新增 `configs/dataset_build.toml`，它只配置本计划新增的输入连接、相机适配、
工作目录和打包需求，不复制 `trajectory.toml` 或 `render.toml` 的算法参数。

目标结构（实现点云生成后）：

```toml
[paths]
trajectory_dir = "outputs/trajectory"
trajectory_filename = "trajectory.json"
semantic_pointcloud_dir = "outputs/trajectory"
semantic_pointcloud_filename = "pointcloud.ply"
work_dir = "outputs/dataset_build"
dataset_root = "data/target"

[trajectory_to_camera]
height_above_floor_m = 0.5
base_extrinsic = [
  1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 1.0,
]

[rendering]
camera_batch_size = 8

[dataset]
group_dir = "robotnav"
scene_dir = "scene-000"
overwrite = false
```

参数所有权如下：

| 参数 | 唯一配置来源 |
| --- | --- |
| LAS、地面估计、共享障碍模型、点云导出、A*、平滑、轨迹随机性 | `trajectory.toml` |
| 3DGS PLY、宽高、FOV、up axis、背景 | `render.toml`，仅供渲染阶段使用 |
| 轨迹和已生成点云的连接路径、工作目录、数据集根目录 | `dataset_build.toml` |
| 相机离地高度、base extrinsic、camera batch size | `dataset_build.toml` |
| group、scene 和覆盖策略 | `dataset_build.toml` |
| `chunk-000`、`000`、ED depth、`10000 units/m`、无效深度 `0` | 第一版实现/文件契约常量，不配置 |

路径继续遵循项目约定，将目录和文件名分开配置。相机宽高和 K 只来自
`render.toml`，不能再次写入 `dataset_build.toml`。

所有阶段都允许选择 `configs/` 下另一份同 schema 配置：

```text
robotnav-trajectory-to-camera --config dataset_build.toml
robotnav-render-trajectory --config dataset_build.toml --render-config render.toml
robotnav-package-dataset --config dataset_build.toml
```

配置加载复用 `robotnav.config` 的 TOML 读取和项目根目录约定，并对 section、字段、
路径组成和数值范围做严格校验。

## 5. 固定中间文件契约

### 5.1 `camera_trajectory.npz`

这是阶段 1 交付给阶段 2/3 的机器可读文件：

| 数组 | dtype | shape | 含义 |
| --- | --- | --- | --- |
| `camera_to_world` | float32/float64 | `(T,4,4)` | 写入 Parquet `action` 的世界系相机位姿 |
| `world_to_camera` | float32/float64 | `(T,4,4)` | 直接交给渲染器的 view matrix |
| `frame_index` | int64 | `(T,)` | 固定为 `0..T-1` |

### 5.2 `camera_trajectory.json`

记录 NPZ 无法自描述的语义：

```json
{
  "contract_version": 1,
  "frame_count": 100,
  "coordinate_convention": "Y-up; ground X-Z; physical up -Y",
  "pose_convention": {
    "camera_to_world": "action",
    "world_to_camera": "gsplat viewmat"
  },
  "source_trajectory": ".../trajectory.json",
  "height_above_floor_m": 0.5
}
```

阶段 2/3 必须同时校验 JSON 版本、帧数和 NPZ shape。替代实现只要产出这两个文件，
不需要兼容原实现的函数或类。

### 5.3 RGB-D 渲染目录

```text
<work_dir>/rendered_episode/
├── rgb/000.png ...
├── depth/000.png ...
└── render_manifest.json
```

`render_manifest.json` 至少包含：

- `contract_version`；
- `frame_count`；
- 展平后的 3×3 K；
- width、height；
- RGB/Depth 的文件顺序；
- depth unit 为 `0.0001 m`，invalid value 为 `0`；
- 实际使用的渲染配置快照或摘要；
- 输入 `camera_trajectory` 的标识或内容 hash。

阶段 3 只能使用 manifest 中明确列出的帧顺序，不能依赖 glob 的偶然顺序。

## 6. 三个核心阶段

### 阶段 1：轨迹到相机位姿

输入：

- `dataset_build.toml`；
- 现有 `trajectory.json`。

处理：

1. 读取 `floor_y` 和 `smooth_path_xz`；
2. 每个 `(x,z)` 转为
   `(x, floor_y - height_above_floor_m, z)`；
3. 使用相邻有效路径点的 X-Z 切线确定朝向；
4. 沿用现有 `look_at_world_to_camera` 的坐标约定和计算公式构造 world-to-camera；
5. 求逆得到 camera-to-world；
6. 写 `camera_trajectory.npz/json`。

这里直接使用现有 `smooth_path_xz`，不再增加 `frame_spacing` 或第二次平滑参数。

完成条件：

- `T == len(smooth_path_xz)` 且 `T > 0`；
- 两组矩阵均为 `(T,4,4)`、数值有限且互为逆；
- `frame_index` 连续且唯一；
- 不调用轨迹规划模块重新生成或修改路径。

### 阶段 2：相机位姿到 RGB-D

输入：

- `camera_trajectory.npz/json`；
- `render.toml` 及其中指定的 3DGS PLY。

处理时复用现有渲染代码：

- `load_ply_to_torch`；
- `make_intrinsics`；
- `_rasterize_compat`；
- `render_batch` 中的 RGB 转换逻辑；
- `depth_render` 中的 ED、alpha mask 和 metric uint16 编码逻辑。

整个 episode 只加载一次 PLY。RGB 和 Depth 必须使用同一 K、同一组 view matrix 和同一
帧编号。目标 Depth 固定为 uint16、`10000 units/m`、无效值 0；不能使用当前调试命令的
LAS backend、adaptive 编码或 `1000 units/m` 默认值。

完成条件：

- RGB/Depth 数量都等于相机位姿数 T；
- RGB 为 `(H,W,3)` uint8；
- Depth 为 `(H,W)` uint16；
- RGB/Depth 文件名均对应 `frame_index`；
- 输出完整的 `render_manifest.json`；
- 不导入或调用阶段 1 的业务接口。

### 阶段 3：整理目标数据

输入：

- `camera_trajectory.npz/json`；
- RGB-D 渲染目录及 `render_manifest.json`；
- 轨迹阶段生成的二值障碍 `pointcloud.ply`；
- `dataset_build.toml`。

处理：

1. 校验三类输入的 contract version 和帧数；
2. 按 manifest 顺序写入目标 RGB/Depth 目录；
3. 写 `000.parquet`；
4. 写 `episodes_stats.jsonl`；
5. 校验并复制二值障碍点云；
6. 写 `run_manifest.json`；
7. 在 staging 目录完成后再提交最终 scene 目录，默认拒绝覆盖。

Parquet 每行对应一帧：

- `observation.camera_intrinsic`：来自 `render_manifest.json` 的同一展平 3×3 K；
- `observation.camera_extrinsic`：来自 `dataset_build.toml` 的同一展平 4×4 base extrinsic；
- `action`：该帧展平存储的 4×4 camera-to-world，读取后 reshape。

第一版 JSONL 固定为：

```json
{"image_index":{"min":0,"max":T-1}}
```

完成条件：

- Parquet 行数、action、RGB、Depth 均为 T；
- K/extrinsic 可按 `target_data.md` 的方式 reshape；
- JSONL 区间与排序后的帧严格对应；
- PLY 可由 Open3D 读取并包含目标障碍颜色；
- 最终 scene 目录不依赖工作目录即可读取；
- 不导入或调用阶段 1/2 的业务接口。

## 7. 编排与验证

可以新增便利命令：

```text
robotnav-build-dataset \
  --config dataset_build.toml \
  --render-config render.toml
```

它只按顺序运行三个文件型阶段并检查退出码。三个阶段仍必须能被单独执行，且编排器不能
省略正式中间文件。

验证分为两层：

- 每个阶段在读取时验证上一阶段的固定文件契约；
- 阶段 3 按 `target_data.md` 对最终目录做完整自检。

独立的 `robotnav-validate-dataset` 可以作为后续或测试辅助命令，但不属于三个核心转换
任务的必要新增模块。无 GPU 测试重点覆盖阶段 1 和阶段 3；阶段 2 使用少量帧做 GPU
smoke test。

## 8. 推荐实现顺序

1. 定义并测试 `camera_trajectory.npz/json` 文件契约；
2. 实现阶段 1，复用现有相机矩阵函数；
3. 定义并测试 RGB-D 目录和 `render_manifest.json` 契约；
4. 实现阶段 3，用合成位姿和图像先打通 Parquet/JSONL；
5. 实现阶段 2，接入现有 3DGS RGB/Depth 代码；
6. 最后增加可选编排命令和端到端 smoke test。

## 9. 总体验收定义

1. 三个阶段均可独立运行，只通过固定文件交换数据；
2. 任一阶段可被替代实现替换，只要输出满足相同 contract version；
3. `dataset_build.toml` 只含新增需求参数，不复制轨迹和渲染算法参数；
4. 轨迹到位姿、位姿到 RGB-D、点云生成和最终整理四个核心任务全部完成；
5. 最终目录完全符合 `target_data.md`；
6. 轨迹和点云共享一次障碍识别，规划膨胀与物理障碍几何保持语义隔离。

## 10. 当前实现状态

已实现：

- `configs/dataset_build.toml` 及严格配置校验；
- `camera_trajectory.npz/json` contract version 1；
- 三个独立阶段 CLI 和只负责顺序启动的编排 CLI；
- RGB/Depth 与相机轨迹 hash 绑定；
- Parquet、JSONL、语义点云结构校验、staging 写入和最终 scene 自检；
- 无 GPU 单元测试及真实 3DGS GPU smoke test。

共享障碍模型和保留原始代表点的点云导出已接入轨迹流水线；规划继续使用碰撞高度带，
语义点云则保留完整障碍高度并包含稀疏上下文。最终打包继续消费
`outputs/trajectory/pointcloud.ply`。
