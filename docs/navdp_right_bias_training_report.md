# NavDP 持续右偏：训练、加载与部署审计

日期：2026-08-16（Asia/Shanghai）

## 最终判定

```text
RIGHT_BIAS_FIRST_APPEARS_AT = Dataset label adapter（训练前）；模型输出最早观测到 checkpoint-50
BIAS_STAGE = TRAINING
```

S0 和 S1 正常且一致；同一固定正前方目标在 accepted smoke checkpoint-50 已变成稳定右向，
之后正式训练的 checkpoint-500 至 checkpoint-10000 全部保持右向。训练最终文件与仿真部署
文件 SHA-256 完全相同，所以偏置不是 export/server/deployment 首次制造的。

根因：`CONFIRMED coordinate/label contract bug`。数据端提供固定
`T_base_from_camera` 后，Dataset 仍将正确 base-local `[forward,left]` 变为
`[left,-forward]`。有符号扩散 MSE 正常地学会了错误标签。

## 模型身份表

| 身份 | 路径 | SHA-256 | 格式/加载结果 |
|---|---|---|---|
| S0 | `/home/ely/Desktop/InternNav/navdp-cross-modal.ckpt` | `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947` | raw state_dict；1066 tensors |
| S1 | S0 经 `NavDPNet.from_pretrained`，更新前 | 与 S0 的 1048 个当前模型 tensors 逐元素一致 | 18 个 `decoder_layer.*` 旧模板被批准忽略；当前模型 strict load |
| S2 | `.../checkpoint-10000/pytorch_model.bin` | `b6b6f31478c34ebe01f6a3d6f0b3d4c552ea74e297bfcb7f5fbb05647e42e1ec` | raw state_dict；1048 tensors |
| 仿真部署 | `/home/ely/Desktop/NavDP_workspace/NavDP/pytorch_model.bin` | `b6b6f31478c34ebe01f6a3d6f0b3d4c552ea74e297bfcb7f5fbb05647e42e1ec` | 与 S2 字节级一致 |

- 中间正式 checkpoint：500 至 10000，每 500 step 一个，共 20 个；每个都有 model、optimizer、
  scheduler、rng 和 trainer_state。
- smoke checkpoint：50、100、105；其 100→105 resume 已由既有验收证明恢复 model、optimizer、
  scheduler 和 global step。
- 未发现 EMA 配置、EMA 权重或 export 权重。训练、server 和部署使用 raw state_dict。
- server 的 `strict=False` 对 S2 只缺少 18 个未使用的 `decoder_layer.*` 模板；实际推理使用的
  `decoder.layers.*`、encoders、action/critic heads 均完整加载。

完整 hash、key 集合、epoch/step 和每个 checkpoint 的参数差异见
`outputs/navdp_right_bias_audit/checkpoint_audit.json`。

## S0 / S1 / S2 固定输入对照

输入取自已记录的真实 Go2 server 模型输入：8 帧 `224x224` BGR、1 帧 metric depth；目标固定
为 `[forward=4,left=0,yaw=0]`。沿用部署类 `NavDP_Policy`、32 candidates、4 个固定随机种子。
轨迹坐标为 `x=forward, y=left`，所以 `y<0` 是右。

| 模型 | 选中轨迹终点 forward 均值 | 选中轨迹终点 left 均值 | 选中轨迹右向比例 | 所有候选终点 left 均值 |
|---|---:|---:|---:|---:|
| S0 | +3.324 m | +0.213 m | 0/4 | +0.025 m |
| S1 | +3.324 m | +0.213 m | 0/4 | +0.025 m |
| smoke step 50 | +1.132 m | -2.955 m | 4/4 | -1.932 m |
| formal step 500 | +1.046 m | -1.925 m | 4/4 | -1.276 m |
| S2 step 10000 | +0.730 m | -2.062 m | 4/4 | -1.949 m |

S0/S1 的同模型、同输入、同 RNG 原始输出差异为 0。S0、S1、S2 的首个固定 seed 原始
candidates 和 positive trajectories 已保存到
`outputs/navdp_right_bias_audit/s0_s1_s2_raw_trajectories.npz`。

正式 step 500–10000 的每个 checkpoint 都是 4/4 选中轨迹右向，且所有候选终点 left 均值
始终为负。由此可排除“只有 critic 排序错”：扩散生成器的候选分布本身已经右偏。

由于没有 step 1–49 权重，模型输出发生改变的精确首 step 无法回放；可严格报告的最早模型
证据为 step 50。因错误监督在第一次 optimizer step 前就存在，因果首发点明确位于 Dataset
label adapter。

## 训练运行时真实配置

| 项 | 实际值 |
|---|---|
| 初始化 | fresh pretrained S0；不是本次 run 的 resume |
| optimizer | `torch.optim.Adam`（自定义 Trainer override） |
| lr | `1e-4` |
| weight decay | 实际 0；配置中的 `1e-4` 被 override 忽略 |
| scheduler | `LinearLR(1.0 -> 0.5, total_iters=10000)` |
| batch / accumulation | 32 / 1 |
| max steps | 10000 |
| AMP / TF32 | false / false |
| TrainingArguments seed | 0 |
| DistributedSampler seed | 1234 |
| Dataset | 300 unique episodes，重复 50 次，总长 15000 |
| sampler | shuffle；三批全量各 5000 |
| drop_last | true |
| EMA | 无 |

checkpoint-500 optimizer 的 870 个 state step 均为 500，scheduler last_epoch=500；
checkpoint-10000 均为 10000。lr 分别为 `9.75e-5` 和 `5.0e-5`，没有 resume step 跳变。
训练日志 1000 个记录的 loss 从 6.9406 降到 1.8465，最小 1.4723；grad norm
`2.057..121.376`，loss/grad 均无 NaN/Inf。weight decay 配置失效是真实配置缺陷，但无法解释
固定方向偏置，归为非主因。

## freeze / optimizer / buffer

| 规则 | requires_grad | optimizer member | state tensors | S0→S2 改变 |
|---|---:|---:|---:|---:|
| `rgbd_encoder.rgb_model.*` | false | true | 175 / 22,056,576 values | 0 |
| 非 RGB `*mask_token*` | false | true | 3 / 1,152 values | 0 |
| 其余 trainable | true | true | 870 / 111,300,746 values | 870 tensors 全部改变 |

- optimizer 从 `model.parameters()` 构造，所以 frozen 参数也在 param group，但没有 gradient 和
  optimizer state；optimizer state_entries 正好为 870。
- 单步验收已证明 frozen 参数无 gradient 且 optimizer.step 后不变。
- S0→S2 对比证明全部 frozen state tensors bitwise 不变；未发现 frozen buffer 漂移。
- trainable 参数到 S2 的全局 delta L2 为 63.673；变化从 step 500 起连续累积。

详细逐 tensor 表：

- `outputs/navdp_right_bias_audit/freeze_optimizer_table.csv`
- `outputs/navdp_right_bias_audit/parameter_delta.csv`

## loss 与 goal conditioning

- diffusion target 是有符号 epsilon，`(pred-noise)^2.mean()`；不存在 unsigned loss。
- 训练 action 是 Dataset 的错误 `pred_actions`，所以 signed loss 会忠实学习负 dim1。
- 总 loss：`0.8*action + 0.2*critic + 0.5*aux`。
- no-goal 分支也用同一错误 action 标签，故即便不使用 point-goal，视觉直行样本仍监督右向。
- multi-goal 分支在 point/image/pixel embeddings 中按 batch index 组合。point-goal 自身也被变为
  `[left,-forward]`，而部署输入仍是 `[forward,left]`，形成额外 conditioning OOD。
- augmentation 只作用于 critic 的候选动作；左右对称的随机 yaw 不会抵消主 action 标签错误。

## 镜像与模态消融

严格镜像测试同时翻转图像、depth、goal left/yaw 和扩散初始噪声，再把输出镜回：

| 模型 | mean abs error | relative error | max abs error |
|---|---:|---:|---:|
| S0 | 0.123 m | 0.185 | 0.781 m |
| S2 | 0.871 m | 1.554 | 4.452 m |

S2 的镜像一致性显著恶化，符合单向错误监督。

S2 消融的选中终点 left 均值：

- 正前方目标：-2.062 m；
- 左侧 2 m 目标：-1.507 m；
- 右侧 2 m 目标：-2.839 m；
- zero RGB：-0.575 m（3/4 seeds 右）；
- zero depth：-2.272 m（4/4 右）；
- RGB/depth 全零：-0.605 m（3/4 右）。

goal 仍有一定调制能力，但整体右向基线压倒目标方向；不是单一视觉纹理造成，也不是 goal 完全
失效。S0 对左右目标分别输出 +1.939 m / -2.051 m，说明同一部署链原本保留正确 goal 符号。

## server / deploy / train 一致性

- 启动说明明确：原模型使用 `navdp-cross-modal.ckpt`，过拟合模型使用根目录
  `pytorch_model.bin`。
- 根目录部署文件与 training S2 SHA-256 相同。
- 既有完整链诊断已经在坐标变换前观察到 NavDP raw path 的负 y；本次固定输入又直接复现。
- server 不做 action 轴交换；Go2 轨迹转换保持负 left 符号，MPC 才据此发出右转。

因此：

```text
BIAS_STAGE = TRAINING
EXPORT = byte copy / no new bias
SERVER = preserves model output
DEPLOYMENT = preserves sign; not first cause
```

## 最小修复建议

1. 先修 Dataset 与新生成契约：对固定 `T_base_from_camera` 直接输出
   `[forward,left,up]`，删除/版本化 `[y,-x]` 旧适配。
2. 增加三项 hard gate：直行标签 `dim0>0, abs(dim1)<eps`；镜像只翻 dim1/yaw；
   S0→S1 raw trajectory bitwise/数值一致。
3. 从未更新的 S0 重新训练。现有 step 50–10000 均已吸收错误监督，不应继续 resume。
4. 首个 smoke 在 1、5、10、25、50 steps 保存并执行正前方 raw trajectory gate；出现负 left
   立即停止。
5. 修正 optimizer 配置事实源（Adam vs TrainingArguments AdamW、weight_decay 被忽略），但把它
   作为独立清理，不要误当作本次方向根因。

禁止采用 MPC 补偿、输出取反或删右转数据；这些会掩盖但不会恢复训练/部署坐标契约。

## 未完成项

- `BLOCKED`：step 1–49 无 checkpoint，无法给出首次模型输出右偏的精确 optimizer step。
- `BLOCKED`：训练数据盘未挂载，无法在本轮重扫全 300 episode；已有契约验收、真实 loader
  抽样和同版本逐帧 smoke 足以确认代数根因。
- `NOT RUN`：修复、重建、重训和修复后仿真/实机回归；本轮遵守“根因明确前不改训练逻辑”。
