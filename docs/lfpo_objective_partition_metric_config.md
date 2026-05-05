# LFPO-F `objective_config` / `partition_config` / `metric_config` 参数说明

本文档解释 `train_lfpo_rl.py` 中这三组配置的含义、控制范围、可选项与当前实现状态。

## 1. 配置入口

在训练脚本中，三组配置在构造 `MolGen_LFPOModel` 前定义并传入：

- `objective_config`
- `partition_config`
- `metric_config`

它们最终进入 `LIFT_Lightning`，分别控制：

- **objective_config**：分子打分规则（oracle/objective）
- **partition_config**：top-bottom 样本划分策略
- **metric_config**：oracle CSV 日志与离线指标输入

## 2. `objective_config`

示例：

```python
objective_config = {
  "aggregate": "official" if objective_name != "qed" else "geometric",
  "use_official_guacamol": True,
  "fallback_aggregate": "geometric",
  "feasibility": {"valid": True, "connected": True},
}
```

### 2.1 `aggregate`
最终 `score` 的聚合方式。

当前实现可选：
- `"official"`：优先调用 GuacaMol benchmark objective 分数（若可用）
- `"geometric"`：加权几何平均
- `"linear"`：加权线性和
- `"tchebycheff"`：Tchebycheff 形式

### 2.2 `use_official_guacamol`
是否尝试调用 GuacaMol 官方 scoring：
- `True`：尝试导入并调用官方 objective
- `False`：直接本地 fallback

### 2.3 `fallback_aggregate`
当前代码中该字段为**预留项**，尚未被实际读取。

### 2.4 `component_weights`
可选字典，控制聚合权重。

### 2.5 `pareto_component_names`
可选列表，指定用于 Pareto 的 component；默认全部 component。

### 2.6 `feasibility`
当前代码中该字段为**预留项**；当前 feasible 实际是 `valid & connected`。

## 3. `partition_config`

示例：

```python
partition_config = {
  "mode": args.partition_mode,
  "top_ratio": 0.25,
  "bottom_ratio": 0.25,
  "pareto_rank_max": 1,
  "top_candidate_quantile": 0.7,
  "diversity_mode": "scaffold",
  "bottom_mode": "unusable_region",
  "bottom_priority": ["invalid", "severe", "dominated", "low_score"],
  "bad_rank_quantile": 0.75,
  "low_score_quantile": 0.25,
}
```

- `mode`：`scalar_top_bottom` 或其余（走 feasible+pareto+diversity 分支）
- `top_ratio` / `bottom_ratio`：top/bottom 采样比例
- `pareto_rank_max`：top Pareto rank 上限
- `top_candidate_quantile`：Pareto 不足时的分位补齐阈值
- `diversity_mode`：`scaffold` / `fingerprint` / none
- `low_score_quantile`：low score bottom 阈值
- `bad_rank_quantile`：dominated 候选阈值
- `bottom_priority`：当前为预留项，代码优先级顺序写死
- `bottom_mode`：当前为预留项

## 4. `metric_config`

示例：

```python
metric_config = {
  "enabled": bool(args.oracle_log_path),
  "oracle_log_path": args.oracle_log_path if args.oracle_log_path else str(script_dir / "oracle_logs" / f"{objective_name}.csv"),
  "novelty_reference_path": str(script_dir / "train_smiles.txt"),
  "log_ref_train": True,
  "log_current_eval": True,
}
```

- `enabled`：是否启用 OracleLogger
- `oracle_log_path`：CSV 输出路径
- `novelty_reference_path`：参考集合，用于 `is_novel`
- `log_ref_train` / `log_current_eval`：控制 source 维度日志

注意：`enabled=bool(args.oracle_log_path)` 表示只有显式传 `--oracle_log_path` 才启用。

## 5. 推荐模板

### QED 兼容
```python
objective_name = "qed"
objective_config = {"aggregate": "geometric"}
partition_config = {"mode": "scalar_top_bottom", "top_ratio": 0.25, "bottom_ratio": 0.25}
metric_config = {"enabled": False}
```

### MPO fallback 稳定
```python
objective_name = "Ranolazine_MPO"
objective_config = {"aggregate": "geometric", "use_official_guacamol": False}
partition_config = {"mode": "feasible_pareto_diverse", "top_ratio": 0.25, "bottom_ratio": 0.25, "pareto_rank_max": 1, "diversity_mode": "scaffold"}
metric_config = {"enabled": True, "oracle_log_path": "./oracle_logs/Ranolazine_MPO.csv", "log_ref_train": True, "log_current_eval": True}
```

### MPO benchmark 对齐优先
```python
objective_config = {"aggregate": "official", "use_official_guacamol": True}
```

若官方包/API 不可用，会回退到本地评分。
