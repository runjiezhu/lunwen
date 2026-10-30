# LLM-Guided Curriculum Reward Shaping for E2E AD - 细化方案

**Idea**: 基于 LLM 的课程化 Reward Shaping 用于端到端自动驾驶后训练
**日期**: 2026-10-21
**状态**: 精化中

---

## 一、相关论文分析

### 📖 论文精读推荐 (Must Read)

#### 1. **LLM-PPO Driver** ⭐⭐⭐ (最相关!)
| 项目 | 内容 |
|------|------|
| **Paper** | LLM-PPO Driver: Improving Autonomous Driving via LLM-Guided Reward Shaping and Imitation Learning |
| **Venue** | ASCC 2026 |
| **URL** | https://doi.org/10.1109/ascc70886.2026.11702225 |
| **核心贡献** | 用 LLM 引导 reward shaping + imitation learning 结合 |
| **关键方法** | LLM 分析驾驶场景，自动生成细粒度 reward 指导 |
| **精读理由** | 🔥 **最直接相关**！是本 idea 最强的 baseline。创新点在于用 LLM 自动生成 reward，但还没有课程学习的概念。|

#### 2. **CurricuVLM** ⭐⭐⭐ (课程学习参考)
| 项目 | 内容 |
|------|------|
| **Paper** | CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models |
| **Venue** | arXiv 2502.15119, Transportation Research Part C 2026 |
| **URL** | https://arxiv.org/abs/2502.15119 |
| **核心贡献** | VLM 驱动的个性化课程学习，针对安全关键场景 |
| **关键方法** | VLM 分析 agent 行为弱点 → 动态生成定制化训练场景 |
| **精读理由** | 🔥 VLM/LLM 做 curriculum learning 的最佳实践，课程生成方法可直接迁移 |

#### 3. **Uncertainty-Aware LLM-Guided Policy Shaping** ⭐⭐ (LLM+RL 融合)
| 项目 | 内容 |
|------|------|
| **Paper** | Uncertainty-Aware LLM-Guided Policy Shaping for Sparse-Reward Reinforcement Learning |
| **Venue** | IEEE CAI 2026 |
| **URL** | https://doi.org/10.1109/cai68641.2026.11536354 |
| **核心贡献** | 不确定性感知的 LLM 引导策略塑形 |
| **关键方法** | 用 uncertainty 过滤 LLM 生成的误导性 reward |
| **精读理由** | 解决 LLM 生成 reward 可能错误的问题，给出了 uncertainty 量化的方法 |

---

### 📚 泛读推荐 (Read Abstract + Key Figures)

| # | Paper | Venue | Key Takeaway | 泛读理由 |
|---|-------|-------|--------------|----------|
| 4 | **Room-of-Adjust** (轻量级 reward shaping) | 2024 | 分离连续倾向奖励 + 离散里程碑奖励 | 简单有效的 reward shaping baseline |
| 5 | **CuRLA** (Curriculum Learning for AD) | 2025 | 课程学习 + RL 用于自动驾驶 | 课程学习在 AD 中的经典应用 |
| 6 | **DriveVLM-RL** | 2025 | VLM + RL 用于安全自动驾驶 | VLM 推理 + RL fine-tuning 的框架 |
| 7 | **Reward Relabelling for IL+RL** | 2024 | 稀疏奖励下 IL+RL 融合 | 稀疏奖励处理方法 |
| 8 | **Sparse Rewards as Preferences** | 2024 | Preference-based RL for sparse reward | Preference learning 方法 |
| 9 | **LLM as Reward Generator** (诊断驱动) | 2026 | 诊断驱动 refinement | LLM 生成 reward 的改进方向 |

---

## 二、Novelty 分析

### 现有工作的 Gap

| 工作 | Reward Shaping | Curriculum | LLM | AD |
|------|---------------|------------|-----|-----|
| LLM-PPO Driver | ✅ LLM 生成 | ❌ | ✅ | ✅ |
| CurricuVLM | ❌ | ✅ VLM 生成 | ✅ VLM | ✅ |
| 本 Idea | ✅ LLM 生成 | ✅ 课程化 | ✅ | ✅ |

**本 Idea 的核心创新**: 
> **把 LLM-PPO 的 reward shaping 和 CurricuVLM 的 curriculum learning 结合起来**，实现一个"LLM 驱动的课程化 reward shaping"框架。

### 具体创新点

1. **课程化的 Reward Shaping**: 不是一次性用 LLM 生成完整 reward，而是分阶段（课程）：
   - 阶段 1: 密集子目标奖励（保持车道、跟车距离等）
   - 阶段 2: 逐步稀疏化，引入安全关键场景
   - 阶段 3: 只依赖安全 + 效率奖励

2. **场景自适应的 Curriculum**: 用 LLM 分析当前训练阶段，自动调整课程难度

3. **Uncertainty 过滤**: 参考 Uncertainty-Aware LLM-Guided Policy Shaping，过滤 LLM 生成的错误 reward

---

## 三、实验方案设计

### 3.1 整体框架

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Curriculum Reward Shaping             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ 场景分析器    │ -> │ 课程规划器    │ -> │ Reward生成器 │ │
│  │ (LLM/VLM)    │    │ (LLM)        │    │ (LLM)        │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         ↑                                      ↓            │
│         │                    ┌──────────────┐ ↓            │
│         │                    │  PPO/IRL    │ ← Reward      │
│         │                    │  Agent      │ ← Sub-goals   │
│         │                    └──────────────┘              │
│         │                          ↓                        │
│         │                    ┌──────────────┐              │
│         └────────────── ←   │ 仿真环境     │              │
│                              │ CARLA/NAVSIM│              │
│                              └──────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Curriculum 设计

| 阶段 | 描述 | 奖励类型 | 难度 |
|------|------|----------|------|
| **Stage 0** | Warm-up | LLM 生成的密集 shaping reward | ⭐ |
| **Stage 1** | 基础场景 | 减少 shaping + 增加稀疏奖励 | ⭐⭐ |
| **Stage 2** | 复杂交互 | 安全关键场景 + 对抗场景 | ⭐⭐⭐ |
| **Stage 3** | 泛化 | 只用安全 + 效率奖励 | ⭐⭐⭐⭐ |

### 3.3 LLM Prompt 设计

```
System: 你是一个驾驶安全专家。对于给定的驾驶场景状态，生成层次化的子目标。
User: 场景: [当前车辆状态、周围障碍物、道路信息]
Task: 生成 3-5 个可达成的子目标，按难度排序
Output Format:
1. [子目标1] - reward: +X
2. [子目标2] - reward: +Y
...
```

### 3.4 数据集 & 环境

| 资源 | 推荐 | 说明 |
|------|------|------|
| **仿真环境** | CARLA Leaderboard 2 / NAVSIM | 开源、benchmark 完善 |
| **离线数据** | BDD100K, nuScenes | 预训练 encoder |
| **预训练模型** | LMDrive, UniAD, DriveVLM | 可做 baseline |

---

## 四、预期贡献

1. **方法贡献**: 提出 LLM 驱动的课程化 reward shaping 框架
2. **实证贡献**: 在 CARLA/NAVSIM 上验证有效性
3. **分析贡献**: 对比 LLM-PPO、CurricuVLM 等 baseline

---

## 五、风险 & 备选方案

| 风险 | 应对 |
|------|------|
| LLM API cost 高 | 用开源 VLM (LLaVA, CogVLM) 替代 |
| LLM reward 错误 | 加入 uncertainty filter |
| 课程设计主观 | 用 agent performance 自动调整课程节奏 |
| 仿真环境差异 | 多环境验证 (CARLA + NAVSIM) |

---

## 六、下一步行动

- [ ] 下载精读论文 (LLM-PPO Driver, CurricuVLM)
- [ ] 复现 LLM-PPO Driver 作为 baseline
- [ ] 设计 LLM prompt 模板
- [ ] 实现 curriculum scheduler
- [ ] 在 CARLA 上跑初始实验
