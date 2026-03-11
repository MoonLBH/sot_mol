# DiffusionNFT：基于预训练 Flow Matching 模型（预测 endpoint / x0 风格）的在线再训练微调指南

> 目标：把 **DiffusionNFT** 论文中的核心方法，改写成一个**可直接落地到你自己的预训练 flow match 模型**上的实现说明。默认你已经有一个预训练模型 `v_ref` / `model_ref`，它在给定 `x_t, t, cond` 时预测一个与 `x0` 等价的目标（例如直接预测 `x0`，或预测可线性换算到 `x0` 的 velocity / residual）。
>
> 本文档重点不是复述论文，而是回答：**如果我已经有自己的预训练 flow matching 模型，怎么按 DiffusionNFT 的思路做在线 RL 式微调？**

---

## 1. 先说结论：DiffusionNFT 到底在做什么

DiffusionNFT 的核心不是 PPO / GRPO 那种“沿采样轨迹做策略梯度”，而是：

1. 用旧模型 `model_old` 采样出一批最终样本 `x0_hat`
2. 给这些最终样本打奖励 `r_raw`
3. 把奖励归一化成 `r in [0, 1]`
4. 再把这些 `(cond, x0_hat, r)` 当成“带软标签的训练数据”
5. 在**前向加噪过程**上构造一个**正分支**和**负分支**的监督损失
6. 用这个损失更新当前训练模型 `model_theta`
7. 用 EMA 把 `model_theta` 缓慢同步给采样模型 `model_old`

所以它本质上是：

**“在线采样 + 奖励加权的正负双分支 flow matching 回归”**

而不是：

**“存整条 reverse trajectory，再做似然比 / clipped objective 的 RL”**。

这正是论文强调的点：它在**forward process**上优化，不需要 reverse process likelihood，也不依赖特定 SDE solver。fileciteturn0file0L3-L8 fileciteturn0file0L73-L88

---

## 2. 你要先把自己的模型放到哪个抽象里

DiffusionNFT 原文用的是 velocity parameterization：

- 前向加噪：`x_t = alpha_t * x0 + sigma_t * eps`
- 监督目标：`v = alpha_dot_t * x0 + sigma_dot_t * eps`
- 模型输出：`v_theta(x_t, cond, t)`

但如果你的预训练 flow matching 模型是 **“预测 endpoint / x0 风格”**，也完全可以做。关键不是必须预测 `v`，而是：

> 你必须有一个统一的训练 target 表达，并且能把“旧模型输出”和“当前模型输出”做线性插值/外推，构造出正负隐式策略。

### 2.1 最推荐的做法

把你的模型接口统一成预测 `x0`：

- `x0_old = model_old(x_t, t, cond)`
- `x0_theta = model_theta(x_t, t, cond)`

然后直接在 `x0` 空间里构造：

- 正分支：
  `x0_pos = (1 - beta) * x0_old + beta * x0_theta`
- 负分支：
  `x0_neg = (1 + beta) * x0_old - beta * x0_theta`

再对真实的监督目标 `x0_target = x0` 做回归。

这相当于把论文里的 `v^+_theta, v^-_theta` 全部改写到 `x0` 参数化下。

### 2.2 为什么这样改是合理的

论文里的隐式正负策略是：

- `v^+_theta = (1 - beta) * v_old + beta * v_theta`
- `v^-_theta = (1 + beta) * v_old - beta * v_theta`

其本质只是：

1. 旧模型是基准点
2. 当前模型相对旧模型的偏移，代表“改进方向”
3. 正分支沿这个方向前进一步
4. 负分支沿相反方向后退一步

所以只要你的参数化空间是**可训练且一致的输出空间**，这个构造就可以搬过去。

### 2.3 什么时候不建议直接搬

如果你的模型输出不是 `x0`，也不是一个与 `x0` 线性等价的量，而是例如：

- 某个 highly non-linear latent
- 多头输出中只有一部分和 `x0` 对齐
- predictor + corrector 的复合状态

那就不要直接在该空间做 `old/theta` 线性组合。此时应先把模型输出统一映射成：

- `x0_pred`，或
- 标准 flow matching target

然后只在这个统一空间中构造正负分支。

---

## 3. 论文方法翻成你可以落地的训练对象

下面假设：

- 你有条件输入 `cond`
- 你有预训练 flow matching 模型 `model_ref`
- 你可以用任意 solver 从它采样出最终样本 `x0_hat`
- 你有一个 reward function：`reward_fn(x0_hat, cond) -> scalar`

我们维护两个模型副本：

- `model_old`：**采样 / rollout 模型**
- `model_theta`：**被梯度更新的训练模型**

初始时：

- `model_old <- deepcopy(model_ref)`
- `model_theta <- deepcopy(model_ref)`

这和原文 Algorithm 1 一致：采样策略与训练策略解耦，训练后再对 `model_old` 做 soft EMA update。fileciteturn0file0L197-L207

---

## 4. 整体训练循环

## Step 0：准备组件

你至少需要这些模块：

- `sample_solver(model, cond, num_steps, ...) -> x0_hat`
- `reward_fn(x0_hat, cond) -> r_raw`
- `sample_time(batch_size) -> t`
- `sample_noise_like(x0) -> eps`
- `alpha(t), sigma(t)`
- 如果你训练的是 velocity，则还要 `alpha_dot(t), sigma_dot(t)`

如果你是 rectified flow / endpoint 风格，最常见就是：

- `x_t = (1 - t) * x0 + t * eps`

若模型预测 `x0`，则 target 直接是 `x0`。

---

## Step 1：用旧模型采样一组候选结果

对每个条件 `cond`，采样 `K` 个最终结果：

```python
samples = [sample_solver(model_old, cond) for _ in range(K)]
```

这里的关键点是：

- **只需要最终干净样本 `x0_hat`**
- **不需要保存整条采样轨迹**
- solver 可以是任意 black-box solver

这也是 DiffusionNFT 相比 GRPO/FlowGRPO 的核心便利之一。fileciteturn0file0L84-L88 fileciteturn0file0L167-L175

如果你自己的生成任务不是图像，而是分子/3D 结构/离散连续混合对象，只要最终能得到一个“可被奖励函数评估”的 clean sample，就可以套这一步。

---

## Step 2：对每个最终样本打分

```python
r_raw_i = reward_fn(x0_hat_i, cond)
```

奖励可以是：

- 单目标标量，如 QED / docking score / shape similarity
- 多目标聚合
- rank-based score
- 黑盒评价器给出的数值

DiffusionNFT 原文假设先得到 raw reward，再归一化成“optimality probability” `r in [0,1]`。fileciteturn0file0L177-L196

---

## Step 3：组内归一化，把 raw reward 变成 `r in [0, 1]`

对同一个条件 `cond` 下的 `K` 个样本：

```python
mean_raw = mean(r_raw_group)
r_norm_i = r_raw_i - mean_raw
r_i = 0.5 + 0.5 * clip(r_norm_i / Zc, -1.0, 1.0)
```

其中：

- `Zc` 是归一化尺度，可取：
  - 该 group 的标准差
  - 全局 reward std
  - 滑动平均 std
  - 一个手工常数

### 3.1 这个 `r` 的直觉

- `r > 0.5`：这个样本相对同组平均更好，应该更像“正样本”
- `r < 0.5`：相对更差，更像“负样本”
- `r = 1`：强正例
- `r = 0`：强负例
- `r = 0.5`：中性，正负分支权重一样

### 3.2 为什么不直接拿 reward 当 loss 权重

因为 DiffusionNFT 不是简单的 reward-weighted regression。

它不是只说“高奖励样本多学一点”，而是同时说：

- 好样本要朝正分支拟合
- 差样本要朝负分支拟合

也就是**显式利用负反馈**。论文里也强调 negative loss 很关键，去掉负分支会很快 collapse。fileciteturn0file0L227-L236

---

## Step 4：把 rollout 数据缓存下来

缓存的数据最小只需要：

```python
(cond, x0_hat, r)
```

注意这里的 `x0_hat` 是 **旧模型采样出的最终 clean sample**，不是 ground-truth data。

这一点很重要：

> DiffusionNFT 的在线训练数据不是来自真实数据集，而是来自当前/旧策略自己生成的样本，再由奖励函数打软标签。

所以它是 **online post-training / online RL-style finetuning**。

---

## Step 5：从缓存里取 `(cond, x0_hat, r)`，重新走 forward noising

训练时，不需要回放采样轨迹，只要对最终样本重新加噪即可。

如果你是 `x0` 预测风格：

```python
t = sample_time(batch)
eps = torch.randn_like(x0_hat)
x_t = alpha(t) * x0_hat + sigma(t) * eps
x0_target = x0_hat
```

如果你是 velocity 风格：

```python
v_target = alpha_dot(t) * x0_hat + sigma_dot(t) * eps
```

这就是论文所谓“在前向过程上做优化”。fileciteturn0file0L89-L107

---

## Step 6：分别跑旧模型和当前模型

如果你是 `x0` 参数化：

```python
with torch.no_grad():
    x0_old = model_old(x_t, t, cond)

x0_theta = model_theta(x_t, t, cond)
```

### 实现要点

- `model_old` 必须 `no_grad`
- `model_old` 是 teacher / reference / sampling policy
- 梯度只能更新 `model_theta`

---

## Step 7：构造隐式正负策略

这是 DiffusionNFT 的核心。

如果你采用 `x0` 参数化：

```python
x0_pos = (1 - beta) * x0_old + beta * x0_theta
x0_neg = (1 + beta) * x0_old - beta * x0_theta
```

如果你采用 velocity 参数化：

```python
v_pos = (1 - beta) * v_old + beta * v_theta
v_neg = (1 + beta) * v_old - beta * v_theta
```

### 7.1 这一步的直觉

设 `delta = x0_theta - x0_old`，则：

- `x0_pos = x0_old + beta * delta`
- `x0_neg = x0_old - beta * delta`

也就是说：

- 正分支沿当前模型相对旧模型的方向前进
- 负分支沿相反方向后退

这就是“negative-aware”的来源。

---

## Step 8：写出 DiffusionNFT 损失

如果你是 `x0` 参数化，最直接的 loss 是：

```python
loss_pos = mse(x0_pos, x0_target)   # 或逐元素/逐样本 mse
loss_neg = mse(x0_neg, x0_target)
loss = r * loss_pos + (1 - r) * loss_neg
loss = loss.mean()
```

更严格一点，要按样本维度写成：

```python
# per-sample
loss_pos_i = ((x0_pos - x0_target) ** 2).flatten(1).mean(dim=1)
loss_neg_i = ((x0_neg - x0_target) ** 2).flatten(1).mean(dim=1)
loss_i = r * loss_pos_i + (1 - r) * loss_neg_i
loss = loss_i.mean()
```

这就是论文 Eq. (5) 在 `x0` 参数化下的直接对应物。原文写的是：

- `r * ||v_pos - v||^2 + (1-r) * ||v_neg - v||^2`fileciteturn0file0L121-L129

### 8.1 为什么这个 loss 会推动模型改进

直觉上：

- 高奖励样本：`r` 大，更多压低 `loss_pos`
- 低奖励样本：`1-r` 大，更多压低 `loss_neg`

要同时满足这两个目标，`model_theta` 就会被推向一个相对 `model_old` 更偏向高奖励区域、远离低奖励区域的位置。

论文从 `pi+ / pi- / pi_old` 的分布关系出发证明了这个点。fileciteturn0file0L108-L129

---

## Step 9：可选——加入论文里的 adaptive weighting

论文指出，普通 diffusion / flow matching 会有时间权重 `w(t)`；他们实际更推荐一种**自适应 x0 回归归一化**方法，而不是手调 `w(t)`。fileciteturn0file0L208-L216

如果你本来就是 `x0` 预测模型，可以直接写成：

```python
err_pos = (x0_pos - x0_target)
err_neg = (x0_neg - x0_target)

scale = (x0_theta.detach() - x0_target).abs().mean(dim=tuple(range(1, x0_theta.ndim)), keepdim=False)
scale = scale.clamp_min(1e-6)

loss_pos_i = err_pos.flatten(1).pow(2).mean(dim=1) / scale
loss_neg_i = err_neg.flatten(1).pow(2).mean(dim=1) / scale
loss_i = r * loss_pos_i + (1 - r) * loss_neg_i
loss = loss_i.mean()
```

更实用的版本是：

```python
base_scale = (x0_old.detach() - x0_target).abs().flatten(1).mean(dim=1).clamp_min(1e-6)
loss_pos_i = ((x0_pos - x0_target) ** 2).flatten(1).mean(dim=1) / base_scale
loss_neg_i = ((x0_neg - x0_target) ** 2).flatten(1).mean(dim=1) / base_scale
loss = (r * loss_pos_i + (1 - r) * loss_neg_i).mean()
```

### 9.1 你需不需要这个 adaptive weighting

建议：

- 第一版实现：先**不用**，直接普通 MSE 跑通
- 训练出现明显不稳定、不同 `t` 段 loss 尺度差异过大时，再加

---

## Step 10：反向传播更新 `model_theta`

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

可以只更新：

- LoRA
- adapter
- 少数 head
- 或整个模型

论文视觉实验是 LoRA 微调。fileciteturn0file0L219-L223

对你自己的 flow matching 分子模型，更推荐先从：

- **只训 LoRA / 小模块**
- 学习率较小
- reward normalization 稳定

开始。

---

## Step 11：soft update 采样策略 `model_old`

每一轮 rollout+update 之后，不要直接把 `model_theta` 硬拷给 `model_old`，而是 EMA：

```python
for p_old, p_theta in zip(model_old.parameters(), model_theta.parameters()):
    p_old.data.mul_(eta).add_(p_theta.data, alpha=1 - eta)
```

原文形式：

```python
theta_old = eta_i * theta_old + (1 - eta_i) * theta
```

论文强调：

- `eta = 0`（完全 on-policy）前期快，但很容易崩
- `eta -> 1` 太稳，但太慢
- 更好的策略：`eta_i` 随训练逐渐增大。fileciteturn0file0L199-L207 fileciteturn0file0L232-L236

### 11.1 推荐调度

可直接这样写：

```python
def get_eta(i, eta_max=0.8, slope=0.01):
    return min(slope * i, eta_max)
```

或者更平滑：

```python
def get_eta(i, warmup=200, eta_start=0.0, eta_end=0.8):
    x = min(i / warmup, 1.0)
    return eta_start + x * (eta_end - eta_start)
```

---

## 5. 给 Codex 的最小实现骨架

下面这个版本是假设你用 `x0` 参数化的 flow matching 模型。

```python
import copy
import torch
import torch.nn.functional as F


class DiffusionNFTTrainer:
    def __init__(
        self,
        model_ref,
        optimizer,
        reward_fn,
        sample_solver,
        alpha_fn,
        sigma_fn,
        beta=1.0,
        K=8,
        device="cuda",
    ):
        self.model_theta = model_ref
        self.model_old = copy.deepcopy(model_ref).eval()
        for p in self.model_old.parameters():
            p.requires_grad_(False)

        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.sample_solver = sample_solver
        self.alpha_fn = alpha_fn
        self.sigma_fn = sigma_fn
        self.beta = beta
        self.K = K
        self.device = device

    @torch.no_grad()
    def rollout_one_group(self, cond):
        x0_list = []
        r_raw_list = []
        for _ in range(self.K):
            x0_hat = self.sample_solver(self.model_old, cond)
            r_raw = self.reward_fn(x0_hat, cond)
            x0_list.append(x0_hat)
            r_raw_list.append(r_raw)

        x0 = torch.stack(x0_list, dim=0)
        r_raw = torch.tensor(r_raw_list, device=x0.device, dtype=torch.float32)

        mean_raw = r_raw.mean()
        std_raw = r_raw.std(unbiased=False).clamp_min(1e-6)
        r = 0.5 + 0.5 * ((r_raw - mean_raw) / std_raw).clamp(-1.0, 1.0)
        return x0, r

    def sample_forward_noise(self, x0):
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device)
        while t.ndim < x0.ndim:
            t = t.unsqueeze(-1)
        eps = torch.randn_like(x0)
        alpha_t = self.alpha_fn(t)
        sigma_t = self.sigma_fn(t)
        x_t = alpha_t * x0 + sigma_t * eps
        return x_t, t, eps

    def nft_loss(self, x0_target, cond, r):
        x_t, t, _ = self.sample_forward_noise(x0_target)

        with torch.no_grad():
            x0_old = self.model_old(x_t, t, cond)

        x0_theta = self.model_theta(x_t, t, cond)

        beta = self.beta
        x0_pos = (1.0 - beta) * x0_old + beta * x0_theta
        x0_neg = (1.0 + beta) * x0_old - beta * x0_theta

        loss_pos = ((x0_pos - x0_target) ** 2).flatten(1).mean(dim=1)
        loss_neg = ((x0_neg - x0_target) ** 2).flatten(1).mean(dim=1)

        loss = r * loss_pos + (1.0 - r) * loss_neg
        return loss.mean()

    @torch.no_grad()
    def ema_update_old(self, eta):
        for p_old, p_theta in zip(self.model_old.parameters(), self.model_theta.parameters()):
            p_old.data.mul_(eta).add_(p_theta.data, alpha=1.0 - eta)

    def train_one_iteration(self, cond_batch, inner_steps=1, eta=0.5):
        buffer = []

        # 1) rollout
        for cond in cond_batch:
            x0_group, r_group = self.rollout_one_group(cond)
            for i in range(x0_group.shape[0]):
                buffer.append((cond, x0_group[i], r_group[i]))

        # 2) optimize
        self.model_theta.train()
        for _ in range(inner_steps):
            total_loss = 0.0
            for cond, x0_target, r in buffer:
                x0_target = x0_target.unsqueeze(0)
                r = r.unsqueeze(0) if r.ndim == 0 else r
                loss = self.nft_loss(x0_target, cond, r)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())

        # 3) soft update old policy
        self.ema_update_old(eta)
        return total_loss / max(len(buffer), 1)
```

---

## 6. 如果你的模型原本不是 `x0` 预测，而是 velocity / endpoint residual，怎么改

## 情况 A：模型预测 velocity

那就完全按论文写：

- `v_old = model_old(x_t, t, cond)`
- `v_theta = model_theta(x_t, t, cond)`
- `v_target = alpha_dot * x0 + sigma_dot * eps`
- `v_pos = (1-beta) * v_old + beta * v_theta`
- `v_neg = (1+beta) * v_old - beta * v_theta`
- `loss = r * mse(v_pos, v_target) + (1-r) * mse(v_neg, v_target)`

## 情况 B：模型预测 residual，但能线性还原 `x0`

例如：

- `pred = model(x_t, t, cond)`
- `x0_pred = convert_to_x0(pred, x_t, t)`

那推荐在 `x0_pred` 空间做 DiffusionNFT，而不是在 residual 空间硬做。

## 情况 C：模型同时输出离散与连续部分

比如分子生成里：

- 坐标 / 连续几何
- 原子类型 / 键 / 电荷等离散 token

你可以对每一类输出分别构造正负分支，然后把损失加总：

```python
loss = (
    lambda_coord * nft_mse(coord_pos, coord_neg, coord_target, r)
    + lambda_atom * nft_ce(atom_pos_logits, atom_neg_logits, atom_target, r)
    + lambda_bond * nft_ce(bond_pos_logits, bond_neg_logits, bond_target, r)
)
```

但第一版建议只在**主导输出空间**上先验证，例如只在连续 `x0`/坐标空间先做。

---

## 7. 在你自己的 flow matching 分子模型里，最自然的落地方式

如果你的模型已经是“从噪声状态逐步生成 3D 分子”的 flow matching 框架，那么 DiffusionNFT 的迁移方式可以概括成：

### 方案一：把最终生成分子当作 `x0_hat`

rollout：

- 用 `model_old` 生成完整分子 `mol_hat`
- 计算 reward，例如：
  - QED
  - SA / SCScore / Fsp3
  - docking / shape similarity
  - 多目标聚合 reward

training：

- 把 `mol_hat` 转回训练表征 `x0_hat`
- 对 `x0_hat` 重新加噪得到 `x_t`
- 用 DiffusionNFT loss 训练

这是最贴近论文的做法。

### 方案二：只对某个子空间做 NFT

比如你的模型由多个头组成：

- 坐标头
- 原子类别头
- 键类型头

你可以：

- 只对坐标头做 NFT loss
- 其余头继续普通监督 / anchor / consistency 正则

这是更稳的工程方案。

---

## 8. 你最需要注意的几个工程问题

## 8.1 `model_old` 必须稳定

如果 `model_old` 更新太快，reward 分布会剧烈漂移，训练非常容易炸。

建议：

- 先用 EMA
- `eta` 不要太小
- 初期 `beta` 也不要太大

## 8.2 奖励标准化极其关键

如果 reward 尺度变化很大，`r in [0,1]` 的映射会不稳定。

建议：

- 用 group 内均值中心化
- 用 group std 或 EMA std 归一化
- 对极端 reward 做截断

## 8.3 `beta` 太大会不稳

原文指出 `beta` 负责速度-稳定性的权衡，接近 1 通常较稳；更小的某些设置会更快。fileciteturn0file0L235-L236

对你的任务，建议从：

- `beta = 0.1`
- `beta = 0.3`
- `beta = 1.0`

三档试起。

## 8.4 先小步微调，不要一上来全模型大改

优先顺序建议：

1. 只训 LoRA / adapter
2. 低学习率
3. 小 rollout group
4. 高频评估 reward 漂移

## 8.5 负分支不要删

这是 DiffusionNFT 和普通 reward-weighted finetuning 最大区别之一。删掉负分支后，你做的就更像 RWR / RFT，而不是 DiffusionNFT。

---

## 9. 建议的最小实验配方

如果你要先做一个最小可行版本，建议：

### Version 1：最简版

- 参数化：`x0` 预测
- loss：普通 MSE 版本的正负分支
- reward 归一化：group mean/std
- old policy update：EMA
- 只更新 LoRA

### Version 2：稳定版

在 V1 基础上再加：

- adaptive weighting
- `eta_i` 递增
- reward clipping
- 保留一个 reference anchor 正则（可选）

### Version 3：完整版

- 多 reward 混合
- 多头输出一起做 NFT
- 更优 rollout solver
- replay buffer / recent buffer 混合采样

---

## 10. 你可以直接交给 Codex 的实现需求

下面这段可以直接作为需求说明。

---

请在现有预训练 flow matching 代码库中实现一个 **DiffusionNFT-style online finetuning trainer**，用于在黑盒 reward 下微调一个预测 `x0` 的生成模型。要求如下：

### 目标

已有：
- 一个预训练模型 `model_ref(x_t, t, cond) -> x0_pred`
- 一个采样器 `sample_solver(model, cond) -> x0_hat`
- 一个奖励函数 `reward_fn(x0_hat, cond) -> scalar`

需要实现：
- 维护两个模型：`model_old`（rollout）与 `model_theta`（trainable）
- 使用 DiffusionNFT 的正负隐式策略损失进行在线微调

### rollout 逻辑

1. 对每个条件 `cond`，使用 `model_old` 采样 `K` 个最终样本 `x0_hat`
2. 对每个 `x0_hat` 调用 `reward_fn` 得到 `r_raw`
3. 在 group 内做 reward 归一化：
   - `r_norm = r_raw - mean(r_raw_group)`
   - `r = 0.5 + 0.5 * clip(r_norm / Z, -1, 1)`
4. 缓存 `(cond, x0_hat, r)`

### training 逻辑

1. 从缓存中取出 `(cond, x0_hat, r)`
2. 采样时间 `t ~ Uniform(0,1)` 和噪声 `eps`
3. 构造前向加噪：`x_t = alpha(t) * x0_hat + sigma(t) * eps`
4. 计算：
   - `x0_old = model_old(x_t, t, cond)`（no_grad）
   - `x0_theta = model_theta(x_t, t, cond)`
5. 构造隐式正负策略：
   - `x0_pos = (1 - beta) * x0_old + beta * x0_theta`
   - `x0_neg = (1 + beta) * x0_old - beta * x0_theta`
6. 计算每样本 loss：
   - `loss_pos = mse(x0_pos, x0_hat)`
   - `loss_neg = mse(x0_neg, x0_hat)`
   - `loss = r * loss_pos + (1 - r) * loss_neg`
7. 对 `model_theta` 反向传播更新

### old policy 更新

每轮训练结束后，用 EMA 更新 rollout 模型：

- `theta_old = eta * theta_old + (1 - eta) * theta`

要求：
- `model_old` 永远不参与梯度
- 支持 `eta` 调度
- 支持 `beta` 配置
- 支持只训练 LoRA / adapter 参数

### 额外要求

- 请封装成清晰的 trainer 类
- 输出训练日志：reward mean/std，r mean，loss_pos，loss_neg，总 loss
- 提供一个最小训练脚本示例
- 所有张量维度和 batch 处理要写清楚
- 代码风格尽量简洁、可直接接入现有项目

---

## 11. 一句话总结

如果你已经有自己的预训练 flow matching 模型，而且它是 `x0` / endpoint 风格，那么实现 DiffusionNFT 的最核心改动只有三步：

1. **用旧模型在线采样 + 奖励打分**
2. **把最终样本重新加噪，构造 `old/theta` 的正负隐式分支**
3. **用 `r * 正分支损失 + (1-r) * 负分支损失` 更新当前模型，再 EMA 同步旧模型**

真正的关键不在 PPO 式 ratio，而在：

> **把奖励学习转成前向过程上的 negative-aware 监督学习。**

如果你的任务是分子生成，这个框架尤其适合，因为 reward 通常本来就是黑盒、不可微、且最终只依赖 clean sample。

