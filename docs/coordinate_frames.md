# 坐标系与变换约定

项目统一使用列向量齐次坐标。`T_A_B` 的含义是把 `B` 坐标系中的点变换到 `A`：
`p_A = T_A_B @ p_B`。矩阵左上角为 `R_A_B`，最后一列为 B 原点在 A 中的位置。

## World (`world`)

Navigation、LAS 变换、3DGS 渲染都使用同一个内部世界系：

- `+X`：场景 X；
- `+Y`：物理向下；
- `+Z`：场景水平轴。

当 LAS 是 Z-up 时，显式变换为 `[X, -Z, Y]`。因此地面是 X-Z 平面，物理向上为 `-Y`。

## 3DGS scene 与 navigation world 的绑定

`hall.las` 的原始点 `p_las=[X,Y,Z]` 进入导航 world 时显式变换为
`p_world=[X,-Z,Y]`。`hall3DGS_yup.ply` 的 Gaussian `x/y/z` 则以单位比例
直接作为这个 `p_world` 送入 gsplat；不再应用任何隐式轴变换。`render.toml`
必须同时引用这对 Hall 资产。

测试会把一个经 LAS 变换的已知障碍点与同位置 Gaussian 通过同一个
`T_camera_world` 和 color `K` 投影，并断言两者落在相同 RGB 像素。

## Go2 base (`base`)

Go2 与 ROS 的 base_link 轴为 `+X` 前、`+Y` 左、`+Z` 上。路线本身只描述地面
投影，数据阶段将其分为两个绝不混用的 pose：

- `robot_ground_pose` / `T_world_ground`：原点是 base_link 在 `floor_y` 地面上的垂直投影；
- `robot_base_pose` / `T_world_base_link`：原点是实际 Go2 base_link，离地高度为
  `base_height_above_floor_m`（来自 `camera_pose_resource`）。

路线切线指定 base_link 的 `+X`，base_link 的 `+Z` 显式映射到 world `-Y`。

## Camera link 与 optical camera

`camera_pose_presets.yaml` 的安装位姿是 `T_base_from_camera_link`，使用 Go2/ROS base 轴。标定和
renderer 使用 pinhole optical 轴：`+X` 右、`+Y` 下、`+Z` 前。二者的显式固定转换为：

```text
R_camera_link_camera_optical =
[[ 0,  0,  1],
 [-1,  0,  0],
 [ 0, -1,  0]]
```

因此 `T_base_from_camera = T_base_from_camera_link @ T_camera_link_camera_optical`，并且
`T_camera_from_base = inverse(T_base_from_camera)`。

## Renderer (`camera`)

gsplat 和 LAS 对照渲染器均接收 `T_camera_world`：

```text
p_camera = T_camera_world @ p_world
depth = p_camera.z
u = fx * p_camera.x / depth + cx
v = fy * p_camera.y / depth + cy
```

所以正深度沿 camera optical 的 `+Z`，图像右与下分别沿 `+X`、`+Y`。数据流水线显式计算：

```text
T_world_camera = T_world_base_link @ T_base_from_camera
T_camera_world = inverse(T_world_camera)
```

`T_camera_world` 是交给 renderer 的 view matrix；`T_world_camera` 写入 dataset 的
`observation.T_world_camera` 与 `action`。`robot_ground_pose` 仅用于记录导航参考，绝不参与
camera 外参组合。
