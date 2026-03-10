# GRPO Fine‑tuning for Flow Matching Models (Single‑Time Surrogate Version)

本指南说明：**当你已经拥有一个预训练的 Flow Matching 模型（endpoint x0 预测形式）时，如何按照 PackFlow 论文的方法进行 GRPO 强化学习微调。**

目标：在 **不计算真实 likelihood 的情况下**，对 Flow 模型进行 RL 对齐。

---

# 1. 方法核心思想

Flow 模型真实 likelihood 为：

log p(x0) = log p(x1) − ∫ div v(x_t,t) dt

该计算需要 **Jacobian trace 积分**，计算代价极高。

论文提出的关键近似：

log π(x) ≈ − L_flow

其中

L_flow = || vθ(x_t,t) − v* ||²

因此 policy ratio 可以写为：

r = exp(L_old − L_new)

这就是 **single‑time surrogate score**。

---

# 2. Flow Matching 基本符号

x0 : 生成样本

采样时间

 t ~ Uniform(0,1)

噪声

 ε ~ N(0,I)

OT 插值：

x_t = (1 − t)x0 + tε

目标速度场：

v* = ε − x0

Flow surrogate loss：

ℓθ = || vθ(x_t,t) − v* ||²

---

# 3. GRPO 微调整体流程

训练循环：

1 采样候选
2 计算奖励
3 计算 group advantages
4 计算 surrogate score
5 计算 policy ratio
6 PPO/GRPO 更新

---

# 4. Step‑by‑Step 训练步骤

## Step 1 — 采样候选

对于每个 conditioning context τ：

生成 K 个样本

x0^(1), x0^(2), …, x0^(K)

这些样本组成一个 **GRPO group**。

---

## Step 2 — 计算 reward

对每个样本计算奖励，例如

r = reward(x0)

例如：

r_E = − Energy(x0)

或

r = task_reward(x0)

---

## Step 3 — 计算 group advantages

在 group 内标准化 reward：

μ = mean(r)

σ = std(r)

A^(k) = (r^(k) − μ) / (σ + ε)


---

## Step 4 — 采样 single‑time surrogate 状态

对每个样本：

采样

t ~ Uniform(0,1)

ε ~ N(0,I)

构造

x_t = (1 − t)x0 + tε

---

## Step 5 — 计算目标速度

OT interpolation 下

v* = ε − x0

---

## Step 6 — endpoint 预测转 velocity

如果模型预测 x0_hat：

vθ(x_t,t) = ε − x0_hat

---

## Step 7 — surrogate loss

计算

ℓθ^(k) = || vθ(x_t,t) − v* ||²

该值近似

− log πθ(x0)

---

## Step 8 — 缓存旧策略 score

使用当前模型参数计算

ℓ_old^(k)

并缓存。

---

## Step 9 — 计算 policy ratio

重新计算

ℓ_new^(k)

ratio：

r^(k) = exp(ℓ_old^(k) − ℓ_new^(k))

---

## Step 10 — PPO / GRPO objective

L = min(

 r^(k) * A^(k),

 clip(r^(k), 1−ε, 1+ε) * A^(k)

)

---

## Step 11 — KL 正则

定义

Δ = ℓθ − ℓ_ref

Schulman KL estimator：

KL = exp(Δ) − Δ − 1

最终 loss：

Loss = −E[L] + β KL

---

## Step 12 — 参数更新

θ ← θ − η ∇ Loss

重复 PPO epochs。

---

# 5. 训练伪代码

```
for iteration:

    samples = generate_samples(policy, K)

    rewards = compute_rewards(samples)

    advantages = normalize(rewards)

    for sample:

        t ~ Uniform(0,1)
        ε ~ Normal()

        x_t = (1−t)x0 + tε

        compute ℓ_old

    for PPO epochs:

        recompute ℓ_new

        r = exp(ℓ_old − ℓ_new)

        compute clipped objective

        add KL regularization

        update θ
```

---

# 6. 实践建议

推荐参数

Group size K: 8–32

PPO epochs: 2–4

Clip ratio: 0.2

KL coefficient β: 1e−3 – 1e−2

---

# 7. 可替代 surrogate 方法

以下方法可用于实验比较。

---

# 7.1 Multi‑time surrogate

随机采样多个时间点

t1, t2, …, tM

定义

L = (1/M) Σ || vθ(x_{t_i},t_i) − v* ||²

优点

更低方差

缺点

计算成本增加

---

# 7.2 Trajectory surrogate

使用 ODE 采样轨迹

x_{t1}, x_{t2}, …, x_{tT}

定义

L = Σ || vθ(x_{t_i},t_i) − v* ||²

优点

与真实 sampling path 一致

缺点

更高计算成本

不是 unbiased likelihood estimator

---

# 8. 建议实验

对比三种 surrogate

1 single‑time surrogate
2 multi‑time surrogate
3 trajectory surrogate

评估指标

RL 稳定性
reward 提升
sample quality
训练速度

---

# 9. 总结

Flow 模型难以直接使用 RL，因为真实 likelihood 需要 divergence 积分。

PackFlow 方法通过

flow matching loss ≈ − log likelihood

实现 GRPO 微调。

Single‑time surrogate 提供

高效
稳定
易实现

的 RL 微调方案。

