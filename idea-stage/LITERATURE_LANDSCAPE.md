# 自动驾驶后训练提效 - 研究调研

**方向**: 自动驾驶后训练（Post-training）中的样本效率与训练效率提升
**日期**: 2026-10-21

---

## 一、文献 Landscape

### 1. 核心问题与挑战

| 问题 | 描述 | 代表工作 |
|------|------|----------|
| **稀疏奖励 (Sparse Reward)** | 自动驾驶场景中安全驾驶的奖励极少，碰撞等负样本更是稀缺 | LLM-PPO Driver, ARMS |
| **离线数据利用 (Offline RL)** | 如何高效利用大量预采集的驾驶数据 | Hierarchical Latent Skills, Offline RL surveys |
| **Imitation + RL 融合** | 纯 IL 难以纠正累积误差，纯 RL 探索成本高 | Fine-Tuning is Not Enough (CVPR 2026) |
| **世界模型 (World Model)** | 减少真实环境交互，提升训练效率 | RAD-2, LMDrive, Latent World Model |
| **Diffusion Planner + RL** | 扩散模型生成轨迹，RL 优化判别器 | RAD-2 (diffusion generator + RL discriminator) |

### 2. 关键研究方向

#### 2.1 Reward Shaping & Sparse Reward

- **LLM-PPO Driver** (ASCC 2026): 用 LLM 引导 reward shaping，将稀疏的安全奖励细化为密集的子目标奖励
- **ARMS** (2026): 自动化的 reward shaping 方法，针对多智能体稀疏奖励场景
- **Autonomous Reward Shaping** (IEEE Access): 自生成轨迹的 reward shaping

#### 2.2 Offline RL & 离线数据高效利用

- **Offline-Trained Encoders** (arXiv 2024): 用大规模视频预训练编码器解决 DR

... [内容已截断，原长度 6897 字符]
