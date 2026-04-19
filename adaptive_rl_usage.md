# Adaptive RL 训练脚本使用说明

新增脚本：`train_rl_adaptive.py`。它用于训练 `AdaptiveRL_Lightning`，并支持按模块做 ablation。

## 1) 快速开始

### 单奖励（QED）baseline（全部模块关闭）
```bash
python train_rl_adaptive.py \
  --config rl.json \
  --reward-name qed \
  --mode baseline \
  --project-name SOTMOL_QED_BASELINE
```

### 单奖励（QED）分别打开三个模块
```bash
# Module A: adaptive time sampling
python train_rl_adaptive.py --config rl.json --reward-name qed --mode module_a --project-name SOTMOL_QED_A

# Module B: reward routing
python train_rl_adaptive.py --config rl.json --reward-name qed --mode module_b --project-name SOTMOL_QED_B

# Module C: constraints
python train_rl_adaptive.py --config rl.json --reward-name qed --mode module_c --project-name SOTMOL_QED_C
```

### 多奖励（MPO preset）baseline vs 模块效果
```bash
# baseline（3个模块全关）
python train_rl_adaptive.py \
  --config rl.json \
  --preset mpo_task_1_tanimoto_qed_tpsa \
  --mode baseline \
  --project-name SOTMOL_MPO1_BASELINE

# 仅开 A / B / C
python train_rl_adaptive.py --config rl.json --preset mpo_task_1_tanimoto_qed_tpsa --mode module_a --project-name SOTMOL_MPO1_A
python train_rl_adaptive.py --config rl.json --preset mpo_task_1_tanimoto_qed_tpsa --mode module_b --project-name SOTMOL_MPO1_B
python train_rl_adaptive.py --config rl.json --preset mpo_task_1_tanimoto_qed_tpsa --mode module_c --project-name SOTMOL_MPO1_C

# 全部打开
python train_rl_adaptive.py --config rl.json --preset mpo_task_1_tanimoto_qed_tpsa --mode all --project-name SOTMOL_MPO1_ALL
```

## 2) 进阶参数覆盖

如果你想精确设置新增模块参数，创建 JSON 文件（例如 `adaptive_overrides.json`）：

```json
{
  "adaptive_time_sampling": true,
  "time_num_bins": 8,
  "time_tau": 0.7,
  "reward_routing_enabled": true,
  "constraints_enabled": true,
  "dual_lr": 0.02
}
```

然后：
```bash
python train_rl_adaptive.py \
  --config rl.json \
  --preset mpo_task_1_tanimoto_qed_tpsa \
  --mode all \
  --adaptive-overrides adaptive_overrides.json
```

注意：`--mode` 会先给出一组开关默认值，`--adaptive-overrides` 会在其后覆盖同名字段。

## 3) 如何验证三个模块“起作用”

建议固定相同 seed、epoch、batchsize，逐组对比：

1. **单奖励 QED**
   - baseline: `--mode baseline`
   - A/B/C 各自单开
   - 观察：`train-rl-reward-mean`、`train-rl-total-loss`，以及模块专属日志

2. **多奖励 MPO**
   - `--preset mpo_task_1_tanimoto_qed_tpsa`
   - baseline vs A/B/C vs all
   - 观察：
     - 组奖励日志：`train-rl-2d_soft-score-mean`
     - A 模块：`train-timeprob-2d_soft-bin*` 是否偏离均匀分布
     - B 模块：`train-rl-2d_soft-routed-loss` 与 baseline loss 形态差异
     - C 模块：`train-constraint-*-violation-mean` 是否下降、`train-constraint-*-lambda` 是否自适应变化

3. **与旧版 rl_diff baseline 对比**
   - 继续使用原有 `train_rl.py` 作为历史 baseline。
   - `train_rl.py` 与 `train_rl_adaptive.py --mode baseline` 的差异可用于确认新类在默认关闭模块时的一致性。

## 4) 常用参数

- `--preset`: 使用 reward preset（例如 `mpo_task_1_tanimoto_qed_tpsa`）
- `--reward-name`: 单奖励任务名（默认 `qed`）
- `--mode`: `baseline | module_a | module_b | module_c | all`
- `--adaptive-overrides`: 自定义 AdaptiveRL 的 JSON 覆盖配置
- `--epochs`, `--batchsize`, `--ngpus`, `--seed`

