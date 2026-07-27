# 所需输入文件格式与约束

以下介绍下游任务需要的输入数据结构。本项目把同一场景中的每条轨迹打包成一个 episode，
因此一个 scene 可以包含多个 parquet 和多行 `episodes_stats.jsonl`。

---

## 目录结构

```
root_dirs/
└── {group_dir}/                     # e.g. "gibson_zed", "3dfront"
    └── {scene_dir}/                 # 每个场景一个子目录
        ├── data/
        │   └── {chunk_name}/        # 只有一个 chunk 子目录，取其第一个
        │       ├── 000.parquet      # 每个 episode 一个 parquet 文件
        │       ├── 001.parquet
        │       └── ...
        ├── videos/
        │   └── {chunk_name}/
        │       ├── observation.images.rgb/    # RGB 图像序列
        │       │   ├── 000.png
        │       │   ├── 001.png
        │       │   └── ...
        │       └── observation.images.depth/  # 深度图像序列
        │           ├── 000.png
        │           ├── 001.png
        │           └── ...
        └── meta/
            ├── episodes_stats.jsonl   # Episode 索引元数据
            └── pointcloud.ply         # 场景点云（含可通行性标注）
```

## 各文件格式与约束

### 1. `meta/episodes_stats.jsonl` — Episode 索引文件

**格式**：JSONL（每行一个 JSON 对象）

**约束**：
- 每个 episode 一行，**行序**与 `data/{chunk_name}/` 下的 parquet 文件**按文件名排序后的顺序一一对应**（`data_paths[episode_idx]`）
- 每行结构：

```json
{
  "image_index": {
    "min": 0,
    "max": 149
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_index.min` | `int` | 该 episode 在 RGB/Depth 文件列表中的起始索引（闭区间） |
| `image_index.max` | `int` | 该 episode 在 RGB/Depth 文件列表中的结束索引（闭区间） |

- RGB 和 Depth 的文件数量需一致，且 `max` 不能越界

---

### 2. `data/{chunk_name}/*.parquet` — 轨迹数据文件

**格式**：Apache Parquet（`pd.read_parquet` 读取）

**必需列**：

| 列名 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `observation.camera_intrinsic` | `list[list[float]]` | `3×3`（展平存储） | 相机内参矩阵，取 `tolist()[0]` 后 reshape |
| `observation.camera_extrinsic` | `list[list[float]]` | `4×4`（展平存储） | 相机外参矩阵（base extrinsic），取 `tolist()[0]` 后 reshape |
| `action` | `list[ndarray]` | `(T, 4, 4)` | 每帧一个 4×4 世界坐标系下的相机位姿矩阵（`np.stack` 后 reshape） |

**约束**：
- `observation.camera_intrinsic` 和 `observation.camera_extrinsic` 的 `tolist()[0]` 分别能 reshape 为 `(3, 3)` 和 `(4, 4)`
- `action` 的帧数 = 轨迹长度，需 >= RGB/Depth 图像数
- Parquet 文件名需与 `episodes_stats.jsonl` 行序一一对应

---

### 3. `meta/pointcloud.ply` — 场景点云

**格式**：PLY 文件（Open3D `read_point_cloud` 读取）

**必需属性**：
- `points`：3D 坐标 `(x, y, z)`
- `colors`：RGB 颜色 `(r, g, b)`，值域 `[0, 1]`

**语义约定**：
- 颜色接近 **`(0, 0, 0.5)`**（深蓝色，`color_distance < 0.05`）的点被识别为**障碍物点**（`scene_obstacle`），用于后续碰撞检测和 `rank_steps` 的优先级采样
- 其余颜色的点被忽略

---

### 4. `videos/{chunk_name}/observation.images.rgb/*` — RGB 图像

**格式**：PIL 可读取的任意图像格式（PNG / JPEG 等）

**约束**：
- `np.array(image, np.uint8)` 可读，shape 为 `(H, W, 3)`
- 文件按**文件名排序**后使用，排序后索引与 `episodes_stats.jsonl` 中的 `image_index` 对应
- 处理时会 resize + pad 到 `image_size × image_size`（默认 224×224），值归一化到 `[0, 1]`

---

### 5. `videos/{chunk_name}/observation.images.depth/*` — 深度图像

**格式**：PIL 可读取的 16-bit 图像

**约束**：
- `np.array(depth, np.uint16)` 可读，shape 为 `(H, W)`
- 深度值单位为 **0.1mm**（代码中 `/10000.0` 转换为米）
- 处理时会 resize + pad 到 `image_size × image_size`，并做裁剪：
  - `> 5.0m` → 置 0（超出有效范围）
  - `< 0.1m` → 置 0（太近无效）
- 文件数量与 RGB 图像一致，按文件名排序

---

## 数据对应关系总结

```
episodes_stats.jsonl 第 i 行
  ├── image_index: {min, max}
  │     ├── rgb_paths[min..max]      → episode_rgb_path
  │     └── depth_paths[min..max]    → episode_depth_path
  ├── data_paths[i]                  → trajectory_data_dir (parquet)
  └── afford_dir                     → trajectory_afford_path (pointcloud.ply, 同一场景共享)
```
