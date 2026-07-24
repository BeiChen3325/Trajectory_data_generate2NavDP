# `pointcloud.ply` 障碍标注与生成计划

## 1. 结论

当前阶段不需要做桌子、椅子、墙、门等类别级语义分割。所有障碍统一抽象为
**不可穿透、不可通行的静态刚体**，生成一个二值几何障碍点云：

- 深蓝色 `(0, 0, 0.5)`：机器人实体不能穿过的物理障碍；
- 其他颜色：地面、非碰撞点或仅用于保留场景上下文的点。

这已经满足 `target_data.md` 和当前打包代码的真实契约：下游只识别接近
`(0, 0, 0.5)` 的点，并将其用于碰撞检测和 `rank_steps` 优先级采样，并不消费障碍物
的具体类别。

不过，“根据碰撞体积计算”不能简化为直接导出当前的
`04_planning_obstacles_inflated.png` 或 `07_planning_blocked_unknown_or_obstacle.png`：

1. 它们是 2.5D 栅格，不是带三维坐标的 PLY；
2. `inflated` 已包含机器人半径和安全边距，下游再次按机器人尺寸碰撞时会重复膨胀；
3. `planning_blocked` 还把未观测区域标成禁行，但未知区域不是物理障碍；
4. PNG 是调试可视化，可能被缩放，不能作为无损中间数据。

这里不使用“不可碰撞的刚体”这一说法，因为它容易被理解成不会发生碰撞；准确含义是
**碰撞后不可穿透的刚体障碍**。

因此第一版应输出**未按机器人半径膨胀的物理障碍几何**。类别级语义分析作为后续可选
增强，不作为数据集构建的前置条件。该点云是从 LAS 和共享障碍模型生成的内生产物，
不再视为 `data/input` 中的外部输入。

## 2. 依据与边界

仓库当前已经存在两套不同用途的障碍表示：

| 表示 | 用途 | 是否可直接作为 `pointcloud.ply` |
| --- | --- | --- |
| `cleaned` 障碍栅格 | 表示从 LAS 几何中检测到的物理障碍投影 | 可作为候选来源，但需要恢复/保留 3D 点 |
| `inflated` 障碍栅格 | A* 使用的机器人配置空间障碍 | 否，已包含机器人半径和安全边距 |
| `planning_blocked` | 障碍、膨胀区和未知区的并集 | 否，未知区不应伪装成物理障碍 |
| 语义 `pointcloud.ply` | 下游碰撞与采样 | 是，必须含 XYZ、RGB 和目标障碍色 |

第一版只回答“这个三维位置是否属于会阻挡当前机器人通行的实体”，不回答“它是什么
物体”。以下情况不需要类别识别：

- 墙和桌腿虽然类别不同，但都会产生碰撞；
- 地面可通过高度带和地面估计排除；
- 高于机器人包络的同一障碍表面仍属于刚体语义点云，不应被规划高度截断；
- 小型噪声可通过体素计数、连通域和最小体积阈值过滤。

仅在下游未来需要类别相关规则时再引入语义模型，例如可推门、可移动椅子、动态行人或
不同代价的软障碍。

## 3. 第一版文件契约

输出路径：

```text
outputs/trajectory/pointcloud.ply
```

它与 `trajectory.json` 是同一流水线的并列产物；打包阶段再复制到
`data/target/<group>/<scene>/meta/pointcloud.ply`。

PLY 的 `vertex` 至少包含：

```text
x: float
y: float
z: float
red: uchar
green: uchar
blue: uchar
```

颜色约定：

- 障碍点固定写为 `(0, 0, 128)`；归一化后约为 `(0, 0, 0.502)`，满足当前
  `color_distance < 0.05` 的校验；
- 非障碍点不得使用接近该颜色的值，建议统一写为浅灰色 `(192, 192, 192)`；
- 默认同时输出经过较稀疏体素采样的非障碍上下文点，便于人工检查坐标、地面和覆盖
  范围；下游会忽略这些点。

坐标必须与轨迹和渲染使用同一世界坐标系。按当前 `zup-to-yup` 变换后的约定，水平面
为 X-Z；代码中的物理离地高度应统一计算为：

```text
height_above_floor = floor_y - y
```

不得单独对输出点云做居中、归一化或额外缩放。

## 4. 与轨迹生成的关系

轨迹规划与 `pointcloud.ply` 生成不应拆成两套障碍识别。两者共用一次坐标变换、地面
估计、障碍点筛选和未膨胀物理障碍清理，再派生两种不同产物：

```text
LAS -> 共享场景障碍模型
              ├── 物理障碍 + 半径/安全边距 + 未知区 -> A* -> trajectory.json
              └── 完整高度的原始三维刚体表面 + 语义着色/代表点采样 -> pointcloud.ply
```

共享的是“哪些观测属于物理障碍”；派生后的语义不能混用。轨迹分支需要膨胀障碍并将
未知区设为禁行，点云分支则必须保留未膨胀的三维实体，不能把未知区伪装成刚体。

实现上将共享预处理、规划派生和 PLY 导出写成模块，但默认仍由
`robotnav-trajectory` 一次调用。模块解耦是为了测试和单独补产物，不代表要求用户执行
两个障碍识别任务。可选的点云导出子命令也只能读取无损共享障碍模型，不能另建识别逻辑。

## 5. 生成方案

### 阶段 A：固定输入与无损中间产物

1. 以 `configs/trajectory.toml` 指定的 LAS 为几何来源。
2. 复用轨迹生成阶段确认过的 `axis_transform`、`floor_y`、ROI 和栅格分辨率。
3. 优先从 `trajectory.json` 或 `floor_estimation_report.json` 读取本次轨迹使用的
   `floor_y`，避免两个命令分别估计后出现高度漂移。
4. 在轨迹生成过程中建立共享障碍模型，并增加无损输出 `occupancy_map.npz`，至少保存：
   - `cleaned_obstacles`；
   - `inflated_obstacles`，仅供对比，不用于标注；
   - `traversable_ground`；
   - `planning_blocked`，仅供对比，不用于标注；
   - `origin_xz`、`resolution`、`width`、`height`；
   - `floor_y` 和坐标变换名称。
5. 为避免在内存中保留完整 LAS，可在共享模型完成后进行第二次流式扫描，选出三维
   障碍点；这是同一流水线对同一识别结果的导出，不是第二套识别算法。
6. 调试 PNG 继续保留，但点云生成器不得从 PNG 反推栅格。

### 阶段 B：构建保留原始几何的三维语义点云

默认由轨迹命令一次生成全部产物：

```bash
uv run robotnav-trajectory --config trajectory.toml
```

```text
outputs/trajectory/
├── trajectory.json
├── occupancy_map.npz
├── pointcloud.ply
└── pointcloud_report.json
```

处理流程：

1. 分块读取 LAS，并执行与轨迹生成完全相同的坐标变换。
2. 丢弃非有限点和 ROI 外点；正式数据集应覆盖整条轨迹的扫掠包络，并额外保留感知或
   碰撞查询余量，不能只覆盖单个相机附近。
3. 计算每个点的 `height_above_floor`。
4. 将离地高度不小于 `ground_margin_m`、且投影落入 `cleaned_obstacles` 的原始点标为
   完整高度障碍；`robot_height_m` 只约束规划分支，不再截断语义点云。
5. 不使用 `inflated_obstacles`，不把 `~traversable_ground` 的未知区标为物理障碍。
6. 对障碍与上下文分别做 3D 体素代表点采样；每个体素保留距体素中心最近的真实 LAS
   点，绝不把人工体素中心写入 PLY。
7. 写出标准彩色 PLY，并同时写出生成报告
   `outputs/trajectory/pointcloud_report.json`。

这里的语义点云保留 LAS 的观测表面，而不是把规划禁行范围或实体内部填充成点。体素
只负责选择真实输入代表点，因此能够降低点数而不制造规则坐标层；若下游未来要求实体内部
采样，应另建明确的体积表示，不能覆盖本文件的表面几何契约。

### 阶段 C：配置化

在 `configs/trajectory.toml` 中配置点云输出：

```toml
[pointcloud]
enabled = true
filename = "pointcloud.ply"
report_filename = "pointcloud_report.json"
obstacle_color_rgb = [0, 0, 128]
context_color_rgb = [192, 192, 192]
obstacle_voxel_size_m = 0.02
context_voxel_size_m = 0.05
include_context = true
```

`configs/dataset_build.toml` 只引用生成后的 `outputs/trajectory/pointcloud.ply`。

## 6. 验证与验收

### 6.1 文件级校验

- PLY 可由 `plyfile` 和 Open3D 读取；
- `vertex` 非空，XYZ 全部有限；
- RGB 能归一化到 `[0, 1]`；
- 至少存在一个与 `(0, 0, 0.5)` 距离小于 `0.05` 的点；
- 点云包围盒与 LAS、轨迹包围盒在同一坐标范围；
- 输出报告记录输入文件哈希、参数、点数、障碍点数、包围盒和颜色计数。

### 6.2 几何校验

生成俯视图和侧视图，至少叠加以下内容：

- 原始 LAS 抽样点；
- 深蓝障碍点；
- `cleaned` 与 `inflated` 栅格边界；
- 平滑轨迹及机器人未膨胀碰撞包络。

重点人工检查：

- 地面没有大面积变成深蓝；
- 墙、桌腿、柜体等真实障碍没有明显断裂；
- 完整障碍表面没有在机器人高度处被截断，高处非障碍噪声处于可接受范围；
- 未知区没有被生成实体障碍点；
- 深蓝点与 RGB/Depth 场景方向一致，没有轴翻转或尺度偏差。


### 6.3 回放验收
1. 沿现有无碰撞轨迹逐帧放置机器人碰撞体，深蓝点不得进入碰撞体；
2. 从轨迹中构造若干向墙、桌腿方向偏移的负样本，应能检测出碰撞；
3. 比较几何点云碰撞结果与 `cleaned` 栅格，统计假阴性和假阳性；
4. 运行：

   ```bash
   uv run robotnav-package-dataset --config dataset_build.toml
   uv run robotnav-validate-dataset data/target/robotnav/scene-000
   ```

5. 最终 scene 中存在 `meta/pointcloud.ply`，且打包 manifest 记录其哈希。

建议第一版验收门槛：

- 现有规划轨迹碰撞数为 0；
- 人工构造的明显穿墙/穿桌腿样本全部检出；
- 抽查区域无系统性轴向或尺度误差；
- 点云文件大小和下游加载时间处于可接受范围。

## 7. 实施顺序

1. **抽出共享障碍模型**：将现有地面、障碍计数和 `cleaned` 结果整理为单一内部数据
   结构，A* 继续消费由它派生的膨胀图。
2. **补无损地图产物**：让轨迹生成命令保存 `occupancy_map.npz` 和完整元数据。
3. **同一流水线导出 PLY**：二次流式读取 LAS，完成几何标注、体素降采样和 PLY
   写出；默认由 `robotnav-trajectory` 调用。
4. **增加单元测试**：用合成地面、墙、桌面和高处点验证高度筛选、颜色与坐标。
5. **增加集成测试**：生成 PLY 后调用现有 `validate_semantic_pointcloud` 和打包流程。
6. **更新连接路径**：让 `dataset_build.toml` 从 `outputs/trajectory` 读取点云。
7. **做真实场景可视化与碰撞回放**：调节体素大小、最小体积和覆盖余量。
8. **冻结第一版契约**：记录配置、输入哈希和质量报告，再批量构建数据集。

## 8. 何时升级到语义分析

满足以下任一条件时，再启动类别级或学习式语义标注：

- 下游明确新增按类别规划或排序的输入字段；
- 需要区分可移动、可穿越、动态或软障碍；
- 单靠高度、连通域和体素几何无法稳定排除大量玻璃反射、漂浮噪声等伪障碍；
- LAS 几何缺失导致关键障碍长期漏检，并且 RGB/3DGS 语义能够可靠补足；
- 评测指标证明二值几何方案成为主要误差来源。

在这些条件出现前，引入类别分割会增加模型依赖、标注成本和类别映射误差，却不会改善
当前 `pointcloud.ply` 的下游契约。因此应先完成可复现、可回放验证的二值几何版本。
