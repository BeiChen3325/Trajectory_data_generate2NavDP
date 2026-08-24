# NavDP 右偏 checkpoint 时间线

固定测试：真实 Go2 server 输入、正前方 `[4,0,0]`、部署推理代码、32 candidates、4 个固定
seed。`selected left < 0` 表示右向。参数 delta 是相对 S0 的全模型 L2。

## 边界

| 状态 | selected endpoint `[forward,left]` m | right seeds | 判定 |
|---|---:|---:|---|
| S0 | `[+3.324,+0.213]` | 0/4 | 正常 |
| S1（加载后、更新前） | `[+3.324,+0.213]` | 0/4 | 与 S0 一致 |
| accepted smoke step 50 | `[+1.132,-2.955]` | 4/4 | 已稳定右偏 |
| accepted smoke step 100 | `[+1.229,-1.746]` | 4/4 | 右偏 |
| accepted resume step 105 | `[+1.262,-1.721]` | 4/4 | 右偏 |

smoke run 与正式 run 使用相同 S0、long300 Dataset、optimizer、scheduler 和 loss；它是训练前
验收运行，不是正式 run 的同一 optimizer 时间线。因此它证明缺陷最迟在第 50 次更新已经进入
模型，但不能提供正式 run 的 step 1–49 权重。

## 正式训练

| step | SHA-256 前缀 | 参数 delta L2 | 最近记录 loss | selected forward | selected left | right fraction |
|---:|---|---:|---:|---:|---:|---:|
| 500 | 24061d5baeb7 | 18.892 | 2.4934 | +1.046 | -1.925 | 1.00 |
| 1000 | 0de2e553e3c3 | 26.071 | 2.2915 | +0.760 | -2.345 | 1.00 |
| 1500 | 3d58dada98cd | 30.842 | 3.3957 | +0.625 | -2.111 | 1.00 |
| 2000 | 1f7171a582fa | 35.419 | 3.3572 | +0.574 | -1.693 | 1.00 |
| 2500 | 6804df805953 | 38.927 | 3.0348 | +0.499 | -2.349 | 1.00 |
| 3000 | 27b95321c01f | 42.262 | 2.6971 | +0.418 | -2.267 | 1.00 |
| 3500 | bb228a30eeb0 | 44.815 | 2.4286 | +0.499 | -1.974 | 1.00 |
| 4000 | f737e1e330ba | 47.229 | 2.5068 | +0.826 | -1.954 | 1.00 |
| 4500 | d398b8798c15 | 49.494 | 2.7243 | +0.602 | -1.921 | 1.00 |
| 5000 | 794fd8eb022b | 51.537 | 2.7570 | +0.900 | -1.856 | 1.00 |
| 5500 | 76d4377e9bec | 53.272 | 2.2968 | +0.496 | -2.083 | 1.00 |
| 6000 | 0c490b8d9a15 | 54.875 | 2.5312 | +0.519 | -1.721 | 1.00 |
| 6500 | 598a2e75d621 | 56.410 | 1.7621 | +0.759 | -2.404 | 1.00 |
| 7000 | e98af9d8f4cb | 57.736 | 2.1924 | +0.493 | -2.585 | 1.00 |
| 7500 | 202a8552a3ab | 58.869 | 2.3478 | +1.111 | -2.096 | 1.00 |
| 8000 | 3427c0476f6a | 59.961 | 2.0261 | +0.582 | -1.800 | 1.00 |
| 8500 | 4f5f46e90fbb | 60.998 | 2.4246 | +0.469 | -2.213 | 1.00 |
| 9000 | 3f0172d60d2b | 61.984 | 2.4266 | +0.577 | -2.142 | 1.00 |
| 9500 | c8cabf4630c3 | 62.861 | 1.8156 | +0.806 | -2.100 | 1.00 |
| 10000 | b6b6f31478c3 | 63.673 | 1.8465 | +0.730 | -2.062 | 1.00 |

每个正式 checkpoint 的 4 个 seed 都选择右向轨迹；每个 checkpoint 的全部候选终点 left
均值也为负。偏置并非训练后期才发生的 collapse，也不是 checkpoint-10000 导出时引入。

## 判定

```text
CAUSAL_FIRST_APPEARANCE = Dataset label adapter（optimizer step 之前）
MODEL_OUTPUT_FIRST_OBSERVED = accepted smoke checkpoint-50
FORMAL_RUN_FIRST_OBSERVED = checkpoint-500
EXACT_MODEL_STEP = within 1..50, not recoverable from existing checkpoints
BIAS_STAGE = TRAINING
```

完整 SHA-256、文件路径、trainer state、optimizer/scheduler 状态和模块参数 delta：

- `outputs/navdp_right_bias_audit/checkpoint_audit.json`
- `outputs/navdp_right_bias_audit/inference_timeline.json`
