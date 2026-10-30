# 自动驾驶后训练提效 - 研究调研

**方向**: 自动驾驶后训练（Post-training）中的样本效率与训练效率提升
**日期**: 2026-10-21

---

## 一、文献 Landscape

### 核心挑战

| 问题 | 描述 | 代表工作 |
|------|------|----------|
| **稀疏奖励** | 安全驾驶的正奖励极少，碰撞等负样本也稀缺 | LLM-PPO, ARMS |
| **离线数据利用** | 如何高效利用大量预采集的驾驶数据 | Hierarchical Latent Skills |
| **IL+RL 融合** | 纯 IL 累积误差，纯 RL 探索成本高 | Fine-Tuning is Not Enough |
| **世界模型** | 减少真实环境交互，提升训练效率 | RAD-2, Latent World Model |
| **Diffusion+RL** | 扩散生成轨迹，RL 优化判别器 | RAD-2 |

### 代表性论文

1. **LLM-PPO Driver** (ASCC 2026): LLM 引导 reward shaping
2. **RAD-2** (arXiv 2026): Diffusion generator + RL discriminator + BEV-Warp 高效仿真
3. **TPSP** (arXiv 2026): Threat-guided policy-aware scene perturbation
4. **AutoVLA** (2025): Adaptive reasoning + RL fine-tuning for VLA
5. **Hierarchical Latent Skills** (arXiv 2023): 层级潜技能加速离线 RL
6. **Offline-Trained Encoders** (arXiv 2024): 视频预训练编码器 + RL
7. **Pseudo-Expert Regularized Offline RL** (CVPR 2026): 伪专家正则化
8. **BEV-Fused IL+RL** (2026): BEV 融合的模仿+强化学习

---

## 二、可做 Ideas

### 💡 Idea 1: LLM-Guided Curriculum Reward Shaping for E2E AD

**问题**: 端到端自动驾驶的奖励信号稀疏且延迟长，RL 训练极不稳定。

**核心思路**: 
- 用 LLM 分析当前驾驶场景状态，自动生成**层次化子目标奖励**（如"保持车道"→"与前车保持安全距离"→"安全变道"）
- 设计 curriculum：初期用密集的子目标奖励引导，后期逐步稀疏化直到只依赖安全奖励

**Novelty**: 
- 区别于手工设计 reward shaping，用 LLM 自动生成场景自适应的 reward curriculum
- 尚未有工作将 LLM-based reward shaping 与 E2E AD 的后训练结合

**可行性**: 中等（需要 LLM API + 仿真环境如 CARLA/NAVSIM）

---

### 💡 Idea 2: Diffusion Planner + RL Discriminator with World Model Acceleration

**问题**: RAD-2 等工作已证明 Diffusion+RL 有效，但训练仍需大量仿真数据。

**核心思路**:
- 在 RAD-2 框架基础上，引入**世界模型**来虚拟生成更多驾驶场景
- 用 world model 扩展安全关键场景（corner cases），减少真实仿真需求
- discriminator 不仅重排轨迹，还预测危险概率作为额外奖励信号

**Novelty**:
- RAD-2 只用 BEV-Warp 仿真加速，没有世界模型
- 引入 world model 增强安全关键场景的生成

**可行性**: 较高（有 RAD-2 开源代码可参考）

---

### 💡 Idea 3: Offline RL with Uncertainty-Aware Pseudo-Expert

**问题**: 离线 RL 面临分布偏移问题（OOD actions），现有方法（CQL、BCQ）效果有限。

**核心思路**:
- 学习一个**不确定性估计器**，识别离线数据中策略不确定的动作
- 对这些高不确定性状态，用逆强化学习或 LLM 生成伪专家动作
- 用伪专家动作作为额外监督，结合 CQL 等正则化

**Novelty**:
- 不同于单纯的价值函数正则化，而是用 uncertainty-guided pseudo-expert
- 结合 IRL/LLM 作为伪专家生成器

**可行性**: 中等（需要离线数据集 + 仿真验证）

---

### 💡 Idea 4: Cross-Domain Transfer with Skill Pretraining for AD

**问题**: 在一个domain（如晴天高速）训练的策略迁移到新domain（雨天城市场景）效率低。

**核心思路**:
- 参考 Hierarchical Latent Skills，提取**可迁移的驾驶技能**（如变道、避障、跟车）
- 在源域预训练技能编码器，然后在目标域只 fine-tune 技能组合层
- 设计技能级 curriculum：从简单技能组合逐步引入复杂场景

**Novelty**:
- Hierarchical Latent Skills 没有针对跨域迁移优化
- 结合 domain randomization + skill composition

**可行性**: 较高（技能提取方法有参考，工作量适中）

---

### 💡 Idea 5: Threat-Guided Data Augmentation for Safety-Critical Training

**问题**: 驾驶数据中安全关键场景（危险交互）天然稀少，RL 难以学会应对。

**核心思路**:
- 基于 TPSP (Threat-guided Policy-aware Scene Perturbation) 的思路
- 但改为**离线数据增强**模式：用预训练的危险检测器识别数据中的"近碰撞"帧
- 对这些帧进行对抗性扰动生成更多安全关键样本
- 用增强数据做 offline RL + safety constraint

**Novelty**:
- TPSP 是在线 RL 场景，改为离线增强
- 结合危险检测器做针对性数据生成

**可行性**: 较高（有 NAVSIM 等 benchmark 可用）

---

## 三、优先级建议

| 优先级 | Idea | 理由 |
|--------|------|------|
| ⭐⭐⭐ | **Idea 5**: Threat-Guided Data Augmentation | 工作量适中，有现成 benchmark，数据增强可解释性强 |
| ⭐⭐⭐ | **Idea 4**: Cross-Domain Transfer | 学术价值高，可与预训练模型结合 |
| ⭐⭐ | **Idea 2**: Diffusion+RL+World Model | 技术前沿，但工作量较大 |
| ⭐⭐ | **Idea 1**: LLM Reward Shaping | 创新性强，但依赖 LLM API 可能有 cost 问题 |
| ⭐ | **Idea 3**: Uncertainty-Aware Offline RL | 理论深度高，但验证难度大 |

---

## 四、下一步

建议先从 **Idea 5** 或 **Idea 4** 入手：
- Idea 5: 用现有数据做增强实验，快速验证可行性
- Idea 4: 与预训练模型（如 LMDrive, UniAD）结合，有代码可参考

需要我进一步细化哪个 idea 的实验方案？
