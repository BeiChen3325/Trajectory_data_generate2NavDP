# NavDP 持续右偏：数据与标签审计

日期：2026-08-16（Asia/Shanghai）

## 结论

根因已确认在数据生成契约迁移与 InternNav Dataset 之间：long300 的
`observation.camera_extrinsic` 已改为固定标定 `T_base_from_camera`，但
`NavDP_Base_Datset.relative_pose()` 仍执行旧契约需要的二维适配
`[raw_y, -raw_x]`。因此真实的 `[前进, 左移]` 被写成训练监督
`[左移, -前进]`。部署明确把输出第 0 维解释为前进、第 1 维解释为左移，故直行监督被
部署解释为负左移，即向右。

根因分类：`CONFIRMED: coordinate/label contract bug`。这不是原始轨迹数量上的右转过采样，
而是每条直行轨迹都会触发的确定性 90° 轴旋转与符号错误。

## 数据身份

- 数据生成仓库：`/home/ely/ZJUgive_DataEngine_For_RobotNav`
- Git HEAD：`71fd9c99acc097f07605d51cbeae6f5977f4ab3b`
- 训练索引：`/home/ely/Desktop/InternNav/data/datasets/navdp_long300_union_index.json`
- 三批场景：seed 17、seed 18、去重 batch3，各 100 条轨迹。
- 唯一 episode：300；总 parquet/RGB/depth 帧：253,221。
- Dataset 固定重复 50 次，训练长度 15,000。
- 三批轨迹均为规划并平滑的成功无碰撞轨迹；源报告记录 collisions=0。
- 路径长度均值：seed17 27.224 m、seed18 27.179 m、batch3 26.637 m。
- batch3 与 seed17/seed18 在原值和 8 位量化下均无重复轨迹。

当前数据盘 `/media/ely/新加卷` 在本次审计时未挂载；因此无法重新扫描 300 个 parquet。
上面的身份、帧数和采样覆盖来自训练前已保存且状态为 PASS 的验收报告。固定契约由三批各自的
验收文件独立证明；本地仍可读的同版本 smoke 数据用于逐帧数值复核。

## 坐标链证据

数据生成端已验证并保存：

```text
action = T_navdp_world_from_camera
observation.camera_extrinsic = T_base_from_camera
```

三批验收报告的
`camera_extrinsic_equals_T_base_from_camera_max_abs_error` 均为 0；seed18 和 batch3 的
`action_equals_T_NavDP_world_from_camera_max_abs_error` 也为 0。

Dataset 当前计算为：

```text
R_world_base = R_action @ inverse(R_base_from_camera)
raw = R_world_base.T @ (p_future_camera - p_current_camera)
raw = [forward, left, up]

returned = [raw_y, -raw_x, raw_z]
         = [left, -forward, up]
```

关键源码位置：

- 生成端固定外参和 action：`src/robotnav/tools/convert_to_navdp_dataset.py`
- Dataset 多余适配：`/home/ely/Desktop/InternNav/internnav/dataset/navdp_lerobot_dataset.py:310`
- 部署端把轨迹 `[0]` 当作前向分母、把 `[1]` 当作横向：
  `/home/ely/Desktop/NavDP_workspace/NavDP/baselines/navdp/policy_agent.py:36`

所以直行的确定性映射是：

```text
canonical [forward=d>0, left=0]
       -> Dataset [dim0=0, dim1=-d]
       -> deployment [forward=0, left=-d]
       -> right
```

## 数值复核

同一生成版本、固定 `T_base_from_camera` 的本地 camera-extrinsic smoke 数据共有 2 条 episode、
436 个 stride-4 位移：

| 量 | 均值 | 判定 |
|---|---:|---|
| canonical forward | +0.123362 m | 正常前进 |
| canonical left | +3.14e-9 m | 无左右偏 |
| Dataset dim0 | +3.14e-9 m | 前向被清空 |
| Dataset dim1 | -0.123362 m | 全部变成右向 |

- 436/436 canonical 位移均为中心方向（`abs(left) <= 0.05 m`）。
- 436/436 经 Dataset 后均被部署解释为右向。
- 256 个按 Dataset 采样规则生成的 point-goal 中，canonical forward 全为正；经适配后
  Dataset dim1 全为负，按 5 cm 阈值 246/256 被部署解释为右目标。
- 逐元素恒等式误差：`dataset_dim0-canonical_left = 0`，
  `dataset_dim1+canonical_forward = 0`。

对照组是旧版 verify 数据。它把 `camera_extrinsic` 写成 ground-frame camera pose，旧的
`[y,-x]` 适配恰好能恢复前向；这解释了为何 Dataset 旧代码曾经可用，也直接证明问题来自
生成契约改变后 Dataset 没有同步更新。

long300 训练前保存的真实 loader 验收又提供了独立证据：15 个跨三批抽样中，11 个
`batch_labels` 的最大值不超过 `1.52e-4`，而最小值约为 `-0.494..-3.715`；对应
point-goal 也主要为负值。其余 4 个弯曲样本正负混合，但都仍经过同一个错误适配。

详细数值：

- `outputs/navdp_right_bias_audit/data_statistics_fixed_extrinsic_smoke.json`
- `outputs/navdp_right_bias_audit/data_statistics_verify10.json`（旧契约对照）

## 左右分布与 sampler

原始 300 条路线的完整 signed-curvature/左右幅值重扫因数据盘未挂载而 `BLOCKED`。但是这不
影响坐标根因判定：错误是逐样本的代数恒等式，而不是依赖总体左右比例的统计相关。

已保存的实际 sampler 证据：

- epoch 0 全量：三批各 5,000 样本。
- 前 3,000 样本：seed17=991、batch3=1,033、seed18=976。
- 无 scene 级 oversampling 差异；50 倍复制对三批相同。

## augmentation 审计

- 未发现 horizontal flip、mirror 或图像单独翻转。
- 唯一方向增强是在 `process_actions()` 中采样
  `yaw ~ Uniform(-pi/3, pi/3)`，分布本身左右对称。
- 该旋转只生成 `augment_actions` 并用于 critic 分支；RGB、depth、goal 不做同步旋转。
- 扩散 action loss 使用原始错误 `pred_actions`；增强不会修复主监督坐标。
- critic 的增强同样在错误二维坐标中构造，因此也继承 train/deploy 坐标不一致。

## 数据端最小修复

不要改数据符号、删除右转样本或在 MPC 补偿。应统一一个公开动作契约：

```text
NavDP action/point-goal/trajectory = [forward, left, yaw]
```

对固定 `T_base_from_camera` 的新数据，`relative_pose()` 应直接返回 base-local
`[forward,left,up]`，不得再做 `[y,-x]`。修复前先增加契约测试：一条直线 `+X_base`
轨迹必须产生正的 action dim0、近零 dim1；其镜像轨迹必须只翻转 dim1/yaw。旧数据若仍需
支持，必须通过版本字段显式选择转换器，不能根据矩阵内容猜测。

## 未完成项

- `BLOCKED`：当前 `/media/ely/新加卷` 未挂载，未重新计算 300 条路线的全量 curvature、
  左/中/右幅值和 scene 级统计。
- `NOT RUN`：修复后的数据重建和重训；本轮按要求只诊断，不修改训练逻辑和数据。
