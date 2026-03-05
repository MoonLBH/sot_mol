# Flow-GRPO：在已有 Flow Matching 预训练模型基础上的在线 RL 微调（逐步指南）

> 本文基于论文 **Flow-GRPO: Training Flow Matching Models via Online RL** 的方法脉络整理成“可复现的步骤清单”，目标是：**当你已经有一个预训练的 flow matching / rectified flow 模型后，如何按论文做在线 RL 再训练（fine-tune）**。

---

## 1. 这篇文章解决了什么问题（核心动机）

Flow Matching（尤其是 Rectified Flow）通常用 **确定性的 ODE 采样**来生成样本：给定随机种子与 prompt，轨迹几乎是确定的。  
而在线 RL（如 GRPO / PPO）需要：
- **随机性（stochasticity）**来探索，形成一批多样化轨迹；
- **可计算的策略概率** `pθ(x_{t-1} | x_t, c)`，以便构造 `ratio r_t`、KL 等项。

论文提出 **Flow-GRPO**，把 GRPO 引入 flow matching，并用两项关键策略解决上述矛盾：

1) **ODE-to-SDE 转换**：把确定性 ODE sampler 转成与原模型“各时刻边缘分布一致”的 **SDE 采样**，从而引入噪声、得到显式的高斯条件分布（可算 logprob / KL）。  
2) **Denoising Reduction**：在线 RL 收集数据时用更少的去噪步数（例如 T=10），但推理仍用原始完整步数（例如 T=40），显著提升采样效率。

---

## 2. 先决条件：你需要有什么（从“已有预训练 flow 模型”开始）

你需要具备以下组件：

### 2.1 预训练 Flow Matching 模型（策略网络）
- 一个速度场/向量场网络 `vθ(x_t, t, c)`（条件输入可为文本 prompt `c`）。
- 原始 rectified flow 的线性插值 noising 形式：  
  `x_t = (1 - t) x_0 + t x_1`，其中 `x_1 ~ N(0, I)`。

> 你可以把它视为“要被 RL 微调的 policy”。

### 2.2 参考策略（reference / ref policy）
- 通常就是“冻结的预训练模型参数” `v_ref`（或 `θ_ref`），用于 KL 约束，防止 reward hacking / 多样性塌缩。

### 2.3 奖励函数 R(x0, c)
- 论文在 T2I 场景用了 **可验证 reward**（GenEval、OCR）或 **模型 reward**（PickScore）。
- 对你自己的任务：只要奖励能对最终样本 `x_0`（以及条件 `c`）打分即可；过程中奖励为 0，仅在最后一步给 reward（稀疏终端奖励）。

---

## 3. 把“去噪过程”写成 MDP（RL 视角）

把 T 步逆向生成过程当成一个 MDP：

- **状态**：`s_t = (c, t, x_t)`
- **动作**：`a_t = x_{t-1}`
- **策略**：`πθ(a_t | s_t) = pθ(x_{t-1} | x_t, c)`
- **转移**：`x_t -> x_{t-1}`（在离散化采样器下）
- **奖励**：仅终止时刻给：`R_t = r(x_0, c)` 当 `t=0`，否则 0

这一步的关键意义：**我们要能写出 `pθ(x_{t-1}|x_t,c)` 的概率形式**，GRPO 才能计算 `ratio` 和 KL。

---

## 4. 关键技术 1：ODE-to-SDE（让策略变成“可取样、可算概率”的高斯）

### 4.1 原本的 ODE（确定性）
典型 Euler 离散（示意）：
`x_{t-1} = x_t + Δt * vθ(x_t, t, c)`  
它给不出 `p(x_{t-1}|x_t,c)`（因为是确定映射），也缺乏探索噪声。

### 4.2 构造等价的 reverse-time SDE（引入随机性，但保持边缘分布）
论文构造了一个 SDE，使其在每个 t 的边缘分布与原流程一致（论文给出证明在 Appendix A）。最终的 Euler–Maruyama 离散更新形式（概念上）是：

**SDE 采样更新：**
```
x_{t+Δt} = x_t
         + driftθ(x_t, t) * Δt
         + σ_t * sqrt(Δt) * ε,      ε ~ N(0, I)
```

其中 `σ_t` 控制随机性强度。论文采用：
`σ_t = a * sqrt(t / (1 - t))`  
`a` 是超参数（噪声/探索强度）。

### 4.3 关键收益：策略条件分布是“各向同性高斯”
有了上面这个离散化，`πθ(x_{t-1}|x_t,c)` 变成一个高斯分布：
- **均值**：由 `x_t` 与 `vθ(·)` 的 drift 决定
- **协方差**：`σ_t^2 * Δt * I`

于是你可以：
- 计算 **logprob** `log pθ(x_{t-1}|x_t,c)`（用于 ratio）
- 计算 **KL(πθ || π_ref)**（论文给出了闭式形式）

> 实现上你只需要明确：给定 `x_t`、`t`，你能算出 `μθ(t, x_t)` 与 `σ_t`，然后把采样写成 `x_{t-1} = μθ + σ_t * sqrt(Δt) * ε` 即可。

---

## 5. 关键技术 2：GRPO（组相对优势 + clipped objective + KL）

GRPO 的核心是：**对同一个 prompt，一次采样一组 G 个样本**，用组内相对方式估计 advantage，避免训练 value network。

### 5.1 采样一组轨迹（用旧策略）
对每个 prompt `c`：
1) 从 `x_T ~ N(0,I)` 开始
2) 用 **SDE sampler** 跑 T 步，得到 `G` 条轨迹：
   `τ_i = (x_T^i, x_{T-1}^i, ..., x_0^i)`

### 5.2 计算组内 reward 与优势（Advantage）
对每个样本 i：
- 终端 reward：`R_i = R(x_0^i, c)`
- 组归一化 advantage（论文公式）：
  `A_i = (R_i - mean(R)) / (std(R) + eps)`

> 注意：论文在公式中写的是 A_t^i，但 reward 只在终端给，所以每一步的 advantage 可直接复用同一个 `A_i`（实现上很常见）。

### 5.3 计算 ratio 与 clipped policy gradient loss
对每条轨迹、每个时间步，计算
`r_t^i(θ) = pθ(x_{t-1}^i | x_t^i, c) / p_{old}(x_{t-1}^i | x_t^i, c)`

然后用 PPO 风格的 clipping：
`min( r_t^i * A_i , clip(r_t^i, 1-ε, 1+ε) * A_i )`

### 5.4 加上 KL 约束（防 reward hacking / 多样性塌缩）
最终目标（概念形式）：
`J = J_clip - β * KL(πθ || π_ref)`

论文强调 KL 与早停不等价：合适的 KL 可以达到同等高 reward，同时保持图像质量/多样性，但训练更久。

---

## 6. 关键技术 3：Denoising Reduction（训练采样用更少步）

在线 RL 的瓶颈是“采样很贵”。论文发现：  
**RL 训练时生成的低质量（少步）样本仍然能提供足够的学习信号**。

做法：
- **训练数据收集**：用更小的步数 `T_train`（论文用 10）
- **最终推理/评测**：仍用原始完整步数 `T_eval`（论文用 40）

这使得训练速度大幅提升（论文报告可达 4× 级别）。

---

## 7. 一份“从 0 到 1”的实现流程（你可以直接照着搭 pipeline）

下面给出一个最小闭环的工程步骤清单（按论文逻辑排序）。

### Step 0：准备
- 冻结一份 reference 模型 `v_ref`（通常是预训练权重拷贝）。
- 决定微调参数化：论文用 **LoRA**（降低显存与稳定训练）。
- 实现奖励 `R(x0, c)` 与 prompt 数据集/采样器。

### Step 1：实现 SDE sampler（替换你原来的 ODE sampler）
输入：`vθ`、prompt `c`、初始 `x_T`、训练步数 `T_train`  
输出：整条轨迹 `x_T, x_{T-1}, ..., x_0` 以及每步需要的统计量（至少能重算 `μθ`、`σ_t`）。

**要点：**
- 你要明确 `t` 的离散时间网格（例如从 1 到 0 的等间隔或按原 scheduler）。
- 每步使用 Euler–Maruyama：`x_{t-1} = μθ + σ_t * sqrt(Δt) * ε`。

### Step 2：实现 logprob 计算（核心）
对每个 step：
- 给定 `x_t`、`t`、`c`，计算 `μθ` 与 `σ_t`，于是：
  `log pθ(x_{t-1}|x_t,c) = log N(x_{t-1}; μθ, σ_t^2 Δt I)`

同时需要 old policy 的 logprob（用 `θ_old` 的 `μ_old`）。

### Step 3：采样 G 条轨迹 + 计算 reward
对每个 prompt：
- 采样 `G` 次（不同噪声 ε / 不同 seed），得到 `G` 个终端样本 `x_0^i`
- 计算 reward `R_i`

### Step 4：组内 advantage
`A_i = (R_i - mean(R)) / (std(R)+eps)`

### Step 5：构造 GRPO loss（clip + KL）
- 用 logprob 得到 `ratio = exp(logp_new - logp_old)`
- `L_clip = - mean_{i,t}( min(ratio*A_i, clip(ratio,1-ε,1+ε)*A_i) )`
- `L_KL = mean_{i,t}( KL(πθ || π_ref) )`
- 总 loss：`L = L_clip + β * L_KL`（符号按你实现的“最小化 loss”定义调整）

### Step 6：参数更新与在线迭代
- 每轮：采样轨迹 → 算 reward/advantage → 反传更新 θ
- 更新 `θ_old ← θ`（或按固定间隔更新 old policy）

### Step 7：评测（用完整步数）
- 使用 `T_eval`（完整 scheduler）生成高质量样本做验证指标。

---

## 8. 论文给出的可直接复用的超参数（默认推荐起步点）

论文 Appendix B.2 给出的固定设置（除 β 外跨任务复用）：
- **训练采样步数**：`T_train = 10`
- **评测步数**：`T_eval = 40`
- **组大小**：`G = 24`
- **噪声强度**：`a = 0.7`（用于 `σ_t = a * sqrt(t/(1-t))`）
- **分辨率**：512（T2I）
- **KL 系数 β**：  
  - GenEval / Text Rendering：`β = 0.004`  
  - PickScore：`β = 0.001`
- **LoRA**：`r = 32`, `α = 64`

> 如果你不是图像任务，LoRA/分辨率可忽略，但 **G、a、β、T_train/T_eval 的思路仍通用**。

---

## 9. 实战注意事项（容易踩坑）

1) **噪声 a 太小 → 探索不足，训练慢**；太大 → 样本质量崩坏，reward 变 0，训练失败。论文建议“在不显著损害样本质量的前提下，取尽可能大的 a”。  
2) **无 KL 容易 reward hacking**：可能出现质量下降或多样性塌缩（不同 seed 输出趋同）。  
3) **KL 不是早停替代品**：要把 KL 当作“持续约束”，目标是让 KL 维持在一个小而稳定的区间。  
4) **reward 设计要能“看见”你想要的能力**：否则 RL 只会优化可钻空子的指标。  
5) **advantage 归一化**：std 很小会导致数值爆炸，记得加 `eps`。  
6) **训练步数减少只用于“收集轨迹”**：最终推理必须回到原步数，否则你评测的是低质量 sampler 而非 RL 对能力的提升。

---

## 10. 最简伪代码（框架级）

```python
# Inputs:
#   v_theta: trainable flow/velocity network
#   v_ref: frozen reference network
#   reward_fn: R(x0, c)
#   schedule: discrete times t_0=1 ... t_T=0 (T_train steps)
# Hyperparams: G, eps_clip, beta, a

while training:
    batch_prompts = sample_prompts(B)

    trajectories = []
    rewards = []

    # --- Collect data with old policy ---
    theta_old = stopgrad(copy(theta))  # or cached weights

    for c in batch_prompts:
        group_trajs = []
        group_rewards = []

        for i in range(G):
            x = sample_normal()  # x_T
            traj = [x]

            for (t, dt) in schedule:
                sigma_t = a * sqrt(t/(1-t))
                mu = drift_from_velocity(v_theta_old, x, t, c, sigma_t)   # compute μ_old
                x = mu + sigma_t * sqrt(dt) * randn_like(x)               # Euler–Maruyama
                traj.append(x)

            x0 = traj[-1]
            r = reward_fn(x0, c)

            group_trajs.append(traj)
            group_rewards.append(r)

        A = normalize(group_rewards)  # (r - mean)/std

        trajectories.append((c, group_trajs, A))
        rewards.append(group_rewards)

    # --- GRPO update ---
    loss = 0.0
    for (c, group_trajs, A) in trajectories:
        for i, traj in enumerate(group_trajs):
            for step in range(T_train):
                x_t = traj[step]
                x_tm1 = traj[step+1]
                t, dt = schedule[step]

                logp_new = logprob_gaussian(v_theta, x_t, x_tm1, t, c, a, dt)
                logp_old = logprob_gaussian(v_theta_old, x_t, x_tm1, t, c, a, dt)

                ratio = exp(logp_new - logp_old)
                pg = min(ratio * A[i], clip(ratio, 1-eps_clip, 1+eps_clip) * A[i])

                kl = kl_gaussian_closed_form(v_theta, v_ref, x_t, t, c, a, dt)

                loss += -(pg) + beta * kl

    loss /= (B * G * T_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # evaluate with full steps T_eval (original sampler schedule)
```

---

## 11. 你如何把它迁移到“你自己的 Flow Matching 模型”（不局限于 T2I）

把论文要点抽象后，你只需要满足 3 个条件：

1) **你的生成过程能写成多步迭代**：`x_t -> x_{t-1}`  
2) **你能把每一步变成一个随机的高斯转移**：`x_{t-1} ~ N(μθ(x_t), Σ_t)`  
   - ODE-to-SDE 给的是一种构造方式；你也可以用别的方式，只要边缘分布合理且能算 logprob
3) **你有终端 reward**：能评估最终样本质量/属性/任务完成度

满足这些，你就可以把 GRPO 套在任何 flow matching generator 上做在线 RL 微调。

---

## 参考：论文中直接对应的关键点（便于你回看原文）
- Flow-GRPO 概览（ODE-to-SDE + Denoising Reduction + GRPO 目标）：论文 Figure 2 与 Section 4  
- GRPO 形式与 advantage 计算：Section 4.1  
- ODE-to-SDE 与 Euler–Maruyama 以及 KL 闭式：Section 4.2  
- Denoising Reduction：Section 4.3  
- 默认超参数：Appendix B.2
