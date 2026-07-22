# Trajectory Work

当前阶段已搭建 LAS 点云到 2.5D 轨迹生成的算法模块，内部统一采用 `Y-up` 坐标：

```text
现实竖直向下 = +Y
现实竖直向上 = -Y
地面平面 = X-Z
高度 h = floor_y - y
```

## 文件职责

- `simple_plan.md`：前期路线讨论和整体方案。
- `trajectory_config.py`：集中保存默认路径、机器人半径、高度、地图分辨率、安全边距、A* 参数等超参数。
- `las_io.py`：原始 LAS 头解析、分块读取 XYZ、`zup-to-yup` 坐标变换。
- `ground_estimation.py`：从采样点估计 `floor_y`，并输出 `floor_y_histogram.png` 与 `floor_estimation_report.json` 方便检查。
- `occupancy_map.py`：按机器人高度范围截取障碍候选点、检测地面可观测区域、投影到 X-Z 栅格、清理障碍、膨胀机器人半径、把未知区域设为不可通行、计算 2D distance transform，并输出中间地图。
- `astar_planner.py`：自由空间起终点选择、最近自由格搜索、8 邻域 A*。
- `path_smoothing.py`：路径 shortcut、稠密采样、Chaikin 平滑、平滑后碰撞复检。
- `visualization.py`：保存带路径的调试图。
- `generate_trajectory.py`：主入口，把以上模块串成完整 pipeline。

## 运行方式

在能读取本项目 Python 依赖的环境中运行：

```powershell
cd C:\task\xlk_work\tools\trajectory_work
python .\generate_trajectory.py
```

常用参数：

```powershell
python .\generate_trajectory.py --robot-height 0.8 --robot-radius 0.25 --resolution 0.08
python .\generate_trajectory.py --start-xz -3.0,1.0 --goal-xz 2.0,8.0
python .\generate_trajectory.py --floor-y 0.247
python .\generate_trajectory.py --floor-search-y-min 0 --floor-search-y-max 3
python .\generate_trajectory.py --roi-center-xz 0,0 --roi-size-xz 12,12
python .\generate_trajectory.py --roi-center-xz none --roi-size-xz none
python .\generate_trajectory.py --resolution 0.2 --floor-sample-limit 200000 --max-stream-points 2000000
```

默认会只处理原点附近 `12m x 12m` 的 ROI。地面估计默认只在：

```text
0 <= y <= 3
```

这个范围里查找高 `y` 的大面积水平面，避免把 `y` 更小的天花板误判为地面。

`render_one_view.py` 现在默认 `--unit-scale 1.0`，和 `compare_render.py`、`trajectory_work` 使用同一套 LAS/PLY 坐标。后续把轨迹接到 3DGS 渲染时不需要再额外缩放轨迹点。

```powershell
python C:\task\xlk_work\tools\render_one_view.py --ply C:\task\xlk_work\MindCloudXAI_output\test1_yup.ply --unit-scale 1.0
```

## 主要输出

默认输出目录：

```text
C:\task\xlk_work\tools\trajectory_work\outputs
```

关键输出：

- `floor_y_histogram.png`：地面高度估计可视化，红线为估计地面。
- `floor_estimation_report.json`：地面估计数值报告。
- `01_obstacle_point_counts.png`：机器人高度范围内点云投影计数图。
- `02_raw_obstacles.png`：按点数阈值得到的原始障碍图。
- `03_cleaned_obstacles.png`：形态学和连通域清理后的障碍图。
- `04_planning_obstacles_inflated.png`：按机器人半径和安全边距膨胀后的规划障碍图。
- `05_raw_ground.png`：地面高度带投影。
- `06_traversable_ground.png`：清理后的可通行地面区域。
- `07_planning_blocked_unknown_or_obstacle.png`：最终规划禁行图，包含障碍和未知区域。
- `08_distance_transform.png`：2D 距离场预览。
- `06_path_debug.png`：A*、shortcut 和平滑轨迹叠加图。
- `trajectory.json`：最终轨迹点，均为 Y-up 坐标系下的 X-Z 平面坐标。

## 当前状态

- 已完成模块拆分和第一版轨迹生成算法。
- 第一版使用点云做几何规划，后续沿 `trajectory.json` 中的轨迹点调用 3DGS 渲染 RGB-D。
- 已加入 ROI 机制，默认只处理原点附近 `12m x 12m` 区域，避免全局巨大点云中的无关地面影响地面估计和起终点采样。
- 已加入地面可观测区域约束，未知区域不会被当成自由空间。
- 已修正地面搜索策略：默认在 `0 <= y <= 3` 中寻找地面，避免把天花板选成 `floor_y`。
- 当前原点 ROI 粗分辨率验证得到 `floor_y ~= 0.247`，旧结果 `floor_y ~= -2.459` 属于天花板/上方平面，不应作为地面使用。
- 已验证输出：`outputs_roi_r015_floor_fixed_v2/trajectory.json`。
- 平滑路径会进行碰撞复检；如果平滑后碰撞，会退回到稠密 shortcut 路径。
