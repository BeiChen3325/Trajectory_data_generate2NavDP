# 多轨迹到图像流水线改造指引

## 目标

当前 `dataset_build.toml` 仍显式选择 `outputs/trajectories/routes/auto_000.json`，适合验证
单条轨迹。下一步应让 trajectory→camera→RGB-D 流水线消费
`trajectory_manifest.json`，为每条轨迹生成互不覆盖的 episode。

Navigation 已为这次改造准备好以下稳定输入：

- `trajectory_manifest.json`：批次入口，路径均相对 manifest 所在目录；
- 每条 route JSON：继续提供 `floor_y`、`smooth_path_xz` 和
  `coordinate_convention`，现有 `build_camera_trajectory()` 可直接复用；
- manifest 的 `source_scene_model_sha256`：绑定全局场景；
- 每条索引项的 `trajectory_sha256`：下游读取前必须复算校验；
- `pointcloud_report.json`：绑定同一场景模型和源 LAS，可在打包前交叉校验。

## 建议目录

```text
outputs/dataset_build/
└── episodes/
    ├── auto_000/
    │   ├── camera_trajectory.npz
    │   ├── camera_trajectory.json
    │   └── rendered_episode/
    │       ├── rgb/
    │       ├── depth/
    │       └── render_manifest.json
    └── auto_001/
        └── ...
```

`trajectory_id` 是 episode 目录名的唯一来源。禁止多个轨迹继续共用根目录下固定的
`camera_trajectory.npz` 或 `rendered_episode/`。

## 代码边界

建议新增 `dataset/trajectory_manifest.py` 和 `dataset/build_episodes.py`：

1. `load_trajectory_manifest()` 只负责版本、相对路径、ID 唯一性、route SHA-256 和场景哈希校验；
2. `EpisodePaths` 只描述一个 episode 的轨迹、相机产物和渲染目录；
3. 批量层遍历 manifest，构造 `EpisodePaths`；
4. 相机转换继续调用现有 `build_camera_trajectory()`，不要复制姿态计算；
5. 渲染继续调用现有单 episode 渲染函数，批量层不包含 gsplat 算法；
6. 打包层明确决定“一个 route 对应一个 target scene”还是“一个 scene 下多个 episode”，
   在目标数据格式确定前不要把该选择隐含在路径拼接中。

配置应从“选择一个轨迹文件”升级为：

```toml
[paths]
trajectory_manifest = "outputs/trajectories/trajectory_manifest.json"
semantic_pointcloud_dir = "outputs/semantic_pointcloud"
semantic_pointcloud_filename = "pointcloud.ply"
work_dir = "outputs/dataset_build"
dataset_root = "data/target"
```

不要把 LAS、场景构建或 A* 参数引入数据集配置。

## 推荐实施顺序

1. 提取当前 `run_trajectory_to_camera()` 中的文件路径部分，让核心函数接收
   `trajectory_path` 和 `EpisodePaths`；
2. 为单条 route 增加 SHA 校验后，保持现有单 episode 测试通过；
3. 增加 manifest 加载器和两个 episode 的合成测试；
4. 将渲染入口改为按 episode 调用现有渲染函数；
5. 最后改打包阶段，并为中途失败增加按 episode 续跑策略。

验收时至少确认：

- 相同 manifest 重跑得到相同相机位姿和 episode 顺序；
- 两条轨迹不会覆盖任何文件；
- 修改任一 route JSON 后会因 SHA 不匹配而失败；
- scene hash 与 PLY report 不一致时在渲染/打包前失败；
- 单条轨迹入口仍通过同一个 episode 实现，不形成第二套转换逻辑。
