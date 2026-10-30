# PPO 完整推导（从策略梯度到最终损失）

> 用途：`paper_draft.md` / `idea-stage/IDEA_REPORT.md` 中 PPO 相关表述的理论支撑
> 推导日期：2026-10-22
> 参考文献：Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347

---

## 0. 记号约定

| 记号 | 含义 |
|------|------|
| $\pi_\theta(a\|s)$ | 策略网络，参数 $\theta$ |
| $\pi_{\theta_{\text{old}}}$ | 采样数据时的旧策略（每轮更新前冻结） |
| $r_t$ | 时刻 $t$ 的即时奖励（reward） |
| $\gamma$ | 折扣因子，$\gamma\in[0,1)$ |
| $V_\theta(s)$ | 价值网络，估计 $\mathbb{E}[\sum \gamma^{k-t} r_k \mid s_t=s]$ |
| $Q(s,a)$ | 动作价值 |
| $A(s,a)$ | 优势函数 $Q(s,a)-V(s)$ |
| $d^{\pi}(s)$ | 状态访问分布 $\sum_t \gamma^t P(s_t=s\mid\pi)$ |
| $\epsilon$ | clip 范围超参（论文默认 0.2） |
| $N, T, K, M$ | 并行环境数、每环境步数、每轮 epoch 数、minibatch 大小 |

⚠️ **术语陷阱**：PPO 原论文中 $r_t$ 同时被用作"单步奖励"和"重要性采样比值"。本文件用 $r_t$ 表示单步奖励，用 $r_t(\theta)$ 表示比值。

---

## 1. 策略梯度定理

目标函数：

$$J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right] = \sum_s d^{\pi_\theta}(s)\sum_a \pi_\theta(a|s)Q^{\pi_\theta}(s,a)$$

用 log-derivative trick（$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)$）：

$$\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s)\sum_a \pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)\,Q^{\pi_\theta}(s,a)$$

### 两个毛病

1. **依赖 $Q^{\pi_\theta}$** —— 需要跑完整轨迹估计，样本效率低、方差大。
2. **依赖 $d^{\pi_\theta}(s)$** —— 状态分布本身依赖 $\theta$，梯度里藏着一个我们不想算的量。

**毛病 1：减基线。** 减去只依赖状态的 $b(s)$ 不改变梯度：

$$\sum_a \pi(a|s)b(s)\nabla\log\pi(a|s) = b(s)\sum_a\nabla\pi(a|s) = b(s)\nabla 1 = 0$$

取 $b(s)=V^\pi(s)$ 得优势函数 $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s\sim d^{\pi_\theta},\,a\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a|s)A^\pi(s,a)\right]$$

减基线不改变梯度的期望，但显著降低方差（$A$ 的绝对值平均远小于 $Q$）。取 $V$ 而非 $Q$ 做基线还有额外好处：$A$ 可用 TD 残差在线学。

**毛病 2：重要性采样。** 用旧策略的数据算新策略的梯度：

$$\nabla_\theta J(\theta) \approx \nabla_\theta \underbrace{\mathbb{E}_{s\sim d^{\pi_{\text{old}}},\,a\sim\pi_{\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}A^{\pi_{\text{old}}}(s,a)\right]}_{\equiv\, L_{\text{surr}}(\theta)}$$

⚠️ **引入的假设（关键）**：$d^{\pi_\theta}(s) \approx d^{\pi_{\text{old}}}(s)$，且 $A^{\pi_{\text{old}}} \approx A^{\pi_\theta}$。

这个近似只在 $\theta \approx \theta_{\text{old}}$ 时成立。**PPO 全文就是在处理"如何在保持这个近似成立的前提下更新 $\theta$"。**

---

## 2. TRPO 的约束形式与自然梯度

带约束的优化问题：

$$\max_\theta\; L_{\text{surr}}(\theta) \quad \text{s.t.}\quad \mathbb{E}_{s\sim d^{\pi_{\text{old}}}}\left[D_{\text{KL}}\big(\pi_{\text{old}}(\cdot|s)\,\|\,\pi_\theta(\cdot|s)\big)\right] \le \delta$$

**为什么用 KL 而不是参数距离 $\|\theta-\theta_{\text{old}}\|$？** 参数空间距离与策略空间距离不是一回事：权重微小变化可能导致输出动作剧烈改变（大网络深层尤甚），而 KL 直接度量**行为**的改变量。这是策略空间而非参数空间的概念。

**拉格朗日形式**（TRPO 论文中约束用 $D_{\text{KL}}(\pi_{\text{old}}\|\pi_\theta)$，惩罚项用 $D_{\text{KL}}(\pi_\theta\|\pi_{\text{old}})$；局部展开时二者一阶等价，而反向 KL 有闭式估计）：

$$\mathcal{L}(\theta, c) = L_{\text{surr}}(\theta) - c\cdot\mathbb{E}_s\big[D_{\text{KL}}(\pi_{\text{old}}\|\pi_\theta)\big]$$

反向 KL 的闭式估计：

$$D_{\text{KL}}(\pi_\theta\|\pi_{\text{old}}) = \sum_a \pi_\theta(a|s)\log\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} = \mathbb{E}_{a\sim\pi_\theta}\big[\log\pi_\theta(a|s)-\log\pi_{\text{old}}(a|s)\big]$$

**一阶近似 + 自然梯度。** KL 在 $\theta_{\text{old}}$ 处的二阶展开：

$$D_{\text{KL}}(\pi_{\text{old}}\|\pi_\theta) \approx \frac{1}{2}\Delta\theta^\top F\,\Delta\theta,\qquad F = \nabla^2_\theta\,\mathbb{E}_s D_{\text{KL}}(\pi_{\text{old}}\|\pi_\theta)\big|_{\theta_{\text{old}}}$$

$F$ 即 Fisher 信息矩阵。$L_{\text{surr}}$ 一阶展开：$L_{\text{surr}}(\theta_{\text{old}}+\Delta\theta) \approx L_{\text{surr}}(\theta_{\text{old}}) + g^\top\Delta\theta$，$g=\nabla_\theta L_{\text{surr}}(\theta_{\text{old}})$。

于是变成带二次约束的二次优化：

$$\max_{\Delta\theta}\; g^\top\Delta\theta - \frac{c}{2}\Delta\theta^\top F\Delta\theta \quad \text{s.t.}\quad \frac{1}{2}\Delta\theta^\top F\Delta\theta \le \delta$$

拉格朗日 $\mathcal{L}' = g^\top\Delta\theta - \frac{c}{2}\Delta\theta^\top F\Delta\theta - \frac{\mu}{2}(\Delta\theta^\top F\Delta\theta - 2\delta)$，对 $\Delta\theta$ 求导置零：

$$g - (c+\mu)F\Delta\theta = 0 \;\Longrightarrow\; \Delta\theta = \frac{1}{c+\mu}F^{-1}g$$

代回约束，令 $\Delta\theta = \alpha F^{-1}g$：

$$\frac{1}{2}\alpha^2 g^\top F g = \delta \;\Longrightarrow\; \alpha = \sqrt{\frac{2\delta}{g^\top F g}}$$

$$\boxed{\Delta\theta = \sqrt{\frac{2\delta}{g^\top F g}}\;F^{-1}g}$$

**这是自然梯度步长**：方向 $F^{-1}g$（Fisher 归一化梯度），步长由 KL 半径 $\delta$ 决定。TRPO 的核心贡献：单调改进有理论保证。

**TRPO 的代价**：需计算并求逆（或近似求逆）Fisher 矩阵，且与 value network 不能共享参数（二阶信息限制）。

---

## 3. PPO 的简化 —— 惩罚项代替约束

$$L^{KL}(\theta) = \hat{\mathbb{E}}_t\left[D_{\text{KL}}(\pi_\theta(\cdot|s_t)\,\|\,\pi_{\text{old}}(\cdot|s_t))\right]$$

$$L^{KL+clip}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t\right] - c\,L^{KL}(\theta)$$

对 $L^{KL}$ 求梯度：

$$\nabla_\theta L^{KL}(\theta) = \hat{\mathbb{E}}_t\left[\nabla_\theta\log\pi_\theta(a_t|s_t) - \nabla_\theta\log\pi_{\text{old}}(a_t|s_t)\right] = \hat{\mathbb{E}}_t\left[\nabla_\theta\log\pi_\theta(a_t|s_t)\right]$$

（第二项在 $\theta=\theta_{\text{old}}$ 处为常数，$\nabla\log\pi_{\text{old}}$ 与 $\theta$ 无关）

**关键点：惩罚项的梯度形式和普通策略梯度一模一样**——都是 $\mathbb{E}[\nabla\log\pi_\theta(a|s)\cdot(\text{标量})]$。因此：

- 不需要 Fisher 矩阵
- 不需要二阶导
- 可与 value network 共享主干网络（PPO 论文的主要工程优势）

在 $L^{KL}$ 中的标量就是 $1$，所以 KL 惩罚项等价于给策略加了一个熵式正则，让新策略不要离旧策略太远。

**自适应 KL 变体**：用控制器根据实测 KL 调整 $c$，使 KL 稳定在目标值附近；实测 KL 超阈值时提前停止本轮更新（early stopping）。

---

## 4. 为什么还需要 clip

代理目标：

$$L_{\text{surr}}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t\right],\qquad r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

**问题**：目标无界。若某动作 $\hat{A}_t > 0$ 且很大，只要把 $\pi_\theta(a_t|s_t)$ 推到无穷大，$L_{\text{surr}}\to+\infty$。优化器会把大量步数花在"先把 $r_t$ 推过 1 再继续推"上——而一旦 $r_t$ 偏离 1 太多，第 1 步的假设（$d^{\pi_\theta}\approx d^{\pi_{\text{old}}}$）彻底失效，重要性采样估计方差爆炸。

更本质地说：**在 off-policy 数据上做重要性采样，方差随 $r_t$ 偏离 1 迅速增大。**

**PPO 的解法**：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}\big(r_t(\theta),\,1-\epsilon,\,1+\epsilon\big)\hat{A}_t\right)\right]$$

### 4.1 逐情况分析

令 $r_t$ 为比值，$A=\hat{A}_t$。$\text{clip}$ 后的值与 $r_t$ 无关，$\min$ 决定哪一项生效。

**情况 A：$A > 0$（好动作，想提高概率）**

| $r_t$ 范围 | $\min$ 取到 | 目标值 | 梯度 |
|---|---|---|---|
| $r_t \le 1+\epsilon$ | $r_t A$ | $r_tA$ | **有**（随 $r_t$ 增大） |
| $r_t > 1+\epsilon$ | $(1+\epsilon)A$ | $(1+\epsilon)A$ | **无**（常数） |

验证 $r_t>1+\epsilon$ 时：$r_tA > (1+\epsilon)A$（$A>0$），$\min$ 取较小的 $(1+\epsilon)A$。

**含义**：概率提升到 $1+\epsilon$ 倍封顶，再往上无额外收益 → 梯度归零，训练自动停止。

**情况 B：$A < 0$（坏动作，想降低概率）**

| $r_t$ 范围 | $\min$ 取到 | 目标值 | 梯度 |
|---|---|---|---|
| $r_t \ge 1-\epsilon$ | $r_t A$ | $r_tA$ | **有**（随 $r_t$ 减小而下降） |
| $r_t < 1-\epsilon$ | $(1-\epsilon)A$ | $(1-\epsilon)A$ | **无**（常数） |

验证 $r_t<1-\epsilon$ 时：$A<0$ 意味着乘以更小系数得到更小值，$(1-\epsilon)A < r_tA$，$\min$ 取 $(1-\epsilon)A$。

**含义**：概率降到 $1-\epsilon$ 倍封底，再往下无额外收益 → 梯度归零。

### 4.2 图示（PPO 论文 Figure 1）

```
L^CLIP
  ↑        A>0: 上升到 1+ε 后变平
  │              ╭──────────
  │             ╱
  │  A<0 ──────╯          A<0: 降到 1-ε 后变平
  │   ╲
  │    ╰─────────────
  └────────────────────────→ r_t
        1-ε      1   1+ε
```

### 4.3 悲观下界性质

$r_t>1+\epsilon$（$A>0$）时，$L^{CLIP}$ 取 $(1+\epsilon)A < r_tA$，即**低于**真实代理目标。$L^{CLIP}$ 是真实目标的**悲观（保守）估计**，永远不会因过度乐观的 off-policy 外推给出错误引导。

### 4.4 为什么用 $\min$ 而不是 $\max$

$\min$ 才能在两个方向都封住：好动作不能无限提升（$A>0$），坏动作不能无限压低（$A<0$）。若用 $\max$，$A<0$ 时会取 $r_tA$（不 clip），坏动作可无限压低——正是要避免的。

### 4.5 clip 与 KL 约束不等价

- Clip 约束**逐样本比值** $r_t\in[1-\epsilon,1+\epsilon]$
- KL 约束**分布层面的平均差异**

理论上可以出现"每个样本偏移都不超过 $\epsilon$，但聚合后 KL 很大"。**实践建议**：同时用 clip + KL early stopping，比单独用 clip 更稳。这也是 PPO 实现几乎都会加 KL 监控的原因。

### 4.6 多 epoch 复用同一批数据

传统策略梯度每样本用一次即扔。PPO 用同一批数据做 $K$ 个 epoch 的小批量更新。这看似违反第 1 步假设，但 clip 提供保护：比值越界则梯度归零，更新不会 runaway。

**隐患**：Engstrom et al. (2020, *Implementation Matters in Deep Policy Gradients*) 指出 PPO 的 $\min$ 形式在某些情况下**过度悲观**——丢弃"比值越界但方向正确"的有用梯度信号，导致学习变慢。他们提出 PPO-IBM（对 clip 项做 importance-weighted 回退）。

**实践建议**：若 PPO 学不动，第一件该试的是把 $\min$ 换成不 clip 但对 KL 做 early stopping 的版本（原论文 §4 的 adaptive KL + early stopping 变体）。

---

## 5. 优势函数估计 —— GAE

### 5.1 三种估计量

**TD(0)**（方差小、偏差大）：
$$\hat{A}_t^{TD(0)} = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**Monte Carlo**（方差大、偏差小）：
$$\hat{A}_t^{MC} = \sum_{k=t}^{T}\gamma^{k-t}r_k - V(s_t)$$

**n-step**（折中）：
$$\hat{A}_t^{(n)} = \sum_{k=t}^{t+n-1}\gamma^{k-t}r_k + \gamma^n V(s_{t+n}) - V(s_t)$$

### 5.2 GAE 的构造

定义 TD 残差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。GAE 把不同 $n$ 的估计量指数加权求和：

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}$$

### 5.3 递推形式（实现用）

$$\boxed{\hat{A}_t^{GAE(\gamma,\lambda)} = \delta_t + \gamma\lambda\,\hat{A}_{t+1}^{GAE(\gamma,\lambda)}}$$

倒序遍历 episode 即可。展开到 $T$ 时刻并截断：

$$\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l} + (\gamma\lambda)^{T-t}\hat{A}_T,\qquad \hat{A}_T = V(s_T)$$

### 5.4 $\lambda$ 的插值含义

- $\lambda=0$：退化为 TD(0)
- $\lambda=1$：退化为 TD($\lambda$)，进一步退化为 Monte Carlo
- $\lambda\in(0,1)$：指数加权混合

**偏差-方差**：偏差随 $\lambda$ 增大而增大（更多 bootstrap 假设），方差随 $\lambda$ 增大而减小（更多样本参与平均）。标准配置 $\gamma=0.99, \lambda=0.95$。

**为什么 GAE 与 PPO 绑定这么紧？** clip 后的梯度里 $\hat{A}_t$ 是唯一信号来源。若 $\hat{A}$ 噪声太大，clip 也救不了——clip 只约束更新幅度，不改善信噪比。GAE 提供低方差优势，正是让 clip 有效的前提。

---

## 6. 完整 PPO 目标函数

$$\boxed{L(\theta,\phi) = \underbrace{L^{CLIP}(\theta)}_{\text{策略}} - \underbrace{c_1 L^{VF}(\phi)}_{\text{价值}} + \underbrace{c_2 S(\theta)}_{\text{熵 bonus}}}$$

**策略项**
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]$$

**价值项**
$$L^{VF}(\phi) = \hat{\mathbb{E}}_t\left[\left(V_\phi(s_t)-\hat{R}_t\right)^2\right],\qquad \hat{R}_t = \sum_{l=0}^{T-t-1}\gamma^l r_{t+l} + \gamma^{T-t}V_\phi(s_T)$$

**熵项**
$$S(\theta) = \hat{\mathbb{E}}_t[-\mathcal{H}(\pi_\theta(\cdot|s_t))]$$

**完整梯度**
$$\nabla_\theta L = \hat{\mathbb{E}}_t\left[\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot\hat{A}_t + \nabla_\theta S(\theta)\right] - c_1\hat{\mathbb{E}}_t\left[\nabla_\phi(V_\phi(s_t)-\hat{R}_t)^2\right]$$

---

## 7. 完整算法

```
输入: N 个并行环境, 每个跑 T 步; 折扣 γ; GAE λ; clip ε; epoch 数 K; minibatch 大小 M
初始化: 策略网络 π_θ, 价值网络 V_φ, 旧策略缓存

for iteration = 1, 2, ...:
    # --- 阶段 1: 采样 ---
    for t = 1..T:
        a_t ~ π_θ_old(·|s_t);  执行动作;  观测 s_{t+1}, r_t
        缓存 (s_t, a_t, r_t, log π_θ_old(a_t|s_t))
    计算 bootstrap 值 V_φ(s_{T+1})

    # --- 阶段 2: 计算 GAE 优势 (倒序递推) ---
    for t = T..1:
        δ_t = r_t + γ V_φ(s_{t+1}) - V_φ(s_t)
        Â_t = δ_t + γλ Â_{t+1}          # Â_{T+1} = 0
    # 回报估计
    for t = 1..T:
        R̂_t = Â_t + V_φ(s_t)             # 技巧: 由优势+价值直接得到

    # --- 阶段 3: 多 epoch 小批量更新 ---
    for epoch = 1..K:
        打乱 N*T 条样本
        for 每个 minibatch (大小 M):
            计算 r_t(θ) = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))
            联合优化 L_clip, L_vf, S
        # 可选: 实测 KL 超阈值则 break (early stopping)

    # --- 阶段 4: 更新基准 ---
    π_θ_old ← π_θ
```

**阶段 2 的技巧**：$\hat{R}_t = \hat{A}_t + V_\phi(s_t)$，因为 $\hat{A}_t = \hat{R}_t - V_\phi(s_t)$，直接反推即可，省去显式计算 bootstrap 项。

---

## 8. 连续动作空间的具体形式（AD 场景）

自动驾驶用连续动作（转向角 $\delta$、加速度 $a$，或轨迹点）。

**策略参数化**：对角高斯
$$\mu_\theta(s) = f_\theta(s),\qquad \sigma_\theta(s)=\text{softplus}(g_\theta(s))\quad\text{或固定常数}$$

**采样**（重参数化技巧，保证可微）
$$a_t = \mu_\theta(s_t) + \sigma_\theta(s_t)\odot\epsilon_t,\qquad \epsilon_t\sim\mathcal{N}(0,I)$$

**log 概率**
$$\log\pi_\theta(a_t|s_t) = -\frac{1}{2}\sum_{k=1}^{K}\left[\left(\frac{a_t^{(k)}-\mu_\theta^{(k)}(s_t)}{\sigma_\theta^{(k)}(s_t)}\right)^2 + \log 2\pi\sigma_\theta^{2(k)}(s_t)\right]$$

**比值的稳定计算**
$$r_t(\theta) = \exp\big(\log\pi_\theta(a_t|s_t) - \log\pi_{\text{old}}(a_t|s_t)\big)$$

⚠️ 直接用概率相除会下溢（$10^{-300}$ 级别）。用 log 相减再 exp，$|\log r_t|$ 通常在 0.01~0.5 量级，安全。

**动作裁剪**：环境侧常把动作 clip 到物理可行范围（转向角 $\pm 30°$、加速度 $\pm 4$ m/s²）。⚠️ 这会引入"环境实际执行分布"与"策略输出分布"的偏差，是 AD 任务里 reward hacking 的常见来源。

**并行环境**：CARLA 单实例慢，通常开 $N=8\sim16$ 个环境（多进程）同时采样，正好契合 PPO 的多 epoch 复用特性——$N\cdot T$ 通常设 2048~8192。

---

## 9. 与本项目方向的交汇点

### 9.1 LLM 生成的奖励必须是 potential-based

Ng, Harada & Russell (1999, *Policy invariance under reward transformations*)：若塑形奖励

$$r' = r + \gamma\Phi(s') - \Phi(s)$$

（potential-based shaping），则**最优策略不变**。Devlin & Kudenko (*Potential-based reward shaping: a survey*) 推广表明：一般的 $r' = r + F(s,a,s')$ 保持策略不变当且仅当 $F$ 可分解为 $\gamma\Phi(s')-\Phi(s)$ 的形式。

**对本项目的意义**：如果 LLM 生成的是"安全距离 +1、压线 -1"这种**任意形式**的加分，它**改变了最优策略**，后果是：
1. 训出的策略可能只是在迎合 LLM 评分函数，而非真的更安全；
2. baseline（无塑形）与本方法（塑形）**优化的不是同一目标**，比较不公平；
3. 审稿人会直接问"你怎么保证 LLM 的奖励没把策略带偏"。

**建议**：要么把 LLM 塑形项约束成 potential-based 形式；要么在论文里明确论证"塑形只用于训练早期引导，最终评价阶段完全移除塑形奖励"（即方案里的"退场"设计）——后者更常见、更容易辩护。

### 9.2 clip 的 $\epsilon$ 与奖励尺度耦合

Clip 约束比值，与奖励绝对尺度无关；但**梯度大小**与奖励尺度成正比。

- 奖励尺度太大 → 优势值大 → 梯度大 → 即使有 clip，早期仍可能一步跨过信任域
- 奖励尺度太小 → 优势信号被数值噪声淹没

**监控指标**：实际被 clip 掉的样本比例 `frac_clipped`，健康值 0.05~0.3。>0.5 说明更新太激进或奖励尺度有问题；接近 0 说明 clip 形同虚设（可能是 $A$ 符号混乱或优势估计有问题）。

**对 LLM 组件的具体设计约束**：让 LLM 输出**结构化的奖励配置**（哪些子目标、每个的权重区间），而非直接输出数值，由固定代码做归一化和边界裁剪。这样 LLM 的自由度被限制在"语义层"，数值稳定性由代码保证。

### 9.3 逐步稀疏化奖励的实现含义

从"密集塑形奖励"过渡到"只依赖安全 + 效率奖励"，对应 $c_1$（价值损失系数）、$c_2$（熵系数）、塑形项权重 $\lambda_{\text{shape}}$ 的调度。

**陷阱**：中途改变奖励函数会破坏 PPO 的数据复用假设——旧数据里的 $\hat{A}_t$ 是用旧奖励算的，不能直接复用。

**正确做法**：
- 把奖励切换放在 iteration 边界（每轮重新采样 rollout）
- **价值网络**是针对旧奖励训的，奖励形式变了则 $V_\phi$ 的标度不再匹配——要么重训价值网络，要么至少重新校准价值输出

这一点值得在论文里作为实现细节明确写出，因为它是**真实存在的工程坑**，写清楚是加分项。

---

## 10. 小结

$$\boxed{\text{PPO} = \text{策略梯度} + \text{截断的信任域} + \text{GAE 优势} + \text{多 epoch 复用}}$$

| 层 | 解决的问题 |
|---|---|
| 减去基线 $V(s)$ | 降低策略梯度方差 |
| 重要性采样比 $r_t(\theta)$ | 复用旧数据、多次更新 |
| clip 到 $[1-\epsilon,1+\epsilon]$ | 限制单次更新的信任域，防止 off-policy 外推爆炸 |
| GAE | 给出低方差 $\hat{A}_t$，让 clip 有效 |
| 多 epoch + minibatch | 提高样本利用率 |
| KL early stopping | clip 不保证整体分布不漂移，额外保险 |

---

## 参考文献

- Schulman J, Wolski F, Radford A, Klimov I. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017.
- Schulman J, Levine S, Moritz P, Jordan M, Abbeel P. *Trust Region Policy Optimization*. arXiv:1502.05477, ICML 2015.
- Schulman J, Moritz P, Levine S, Jordan M, Abbeel P. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438, 2016.
- Engstrom S, Ilyas A, Celebi A. *Implementation Matters in Deep Policy Gradients*. NeurIPS 2020.
- Ng A Y, Harada D, Russell S. *Policy invariance under reward transformations: Theory and application to reward shaping*. ICML 1999.
- Devlin S, Kudenko D. *Potential-based reward shaping: a survey*. JAIR 2022.
