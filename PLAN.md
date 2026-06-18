# MVP 执行计划

## 目标

验证核心假设：**玩家通过自定义奖励函数，能否确实塑造出行为差异化的智能体。**

---

## 分层规划

### 第一层：核心技术验证（当前阶段暂时用python pytorch）

| 序号 | 模块 | 内容 | 验证目标 |
|------|------|------|----------|
| 1 | 最小游戏环境 | 极简 2D 对战环境（1v1，vs 规则 Bot），统一观测/动作接口 | 环境基础可用 |
| 2 | PPO 训练管线 | PPO + MLP，GPU 训练，内置优势归一化 | 训练框架可行 |
| 3 | 奖励配置系统 | 3~5 个基础奖励分项，支持权重调节 | 奖励可配置 |
| 4 | 行为差异验证 | 两组极端奖励参数训练，对比行为指标 | **核心假设验证** |

### 第二层：多智能体 + 训练体验

| 序号 | 模块 | 内容 |
|------|------|------|
| 5 | 多智能体联合训练 | 2v2 环境，每个智能体独立奖励函数 |
| 6 | 冻结模型机制 | 训练完成的模型冻结后作为陪练/对手 |
| 7 | 基础可视化 | 训练曲线 + 行为热力图 + 单帧奖励分解 |
| 8 | 预训练模型 | 训练 2~3 个官方基础模型 |

### 第三层：对战 + 完整体验

| 序号 | 模块 | 内容 |
|------|------|------|
| 9 | 在线对战 | 客户端推理 + 服务端裁决 |
| 10 | 域随机化 | 分层域随机化接入 |
| 11 | 消融对比 | 多实例并行训练 + 对比报告 |

---

## 第一层详细方案

### 1. 最小游戏环境

- 1v1 对战，对手为固定规则 Bot
- 观测：结构化浮点向量，~20-40 维
- 动作：移动为 4 个独立二值按键(W/A/S/D 各自 0/1，支持同时按下) + 技能释放(3选1)
- 对局有时间上限，双方有血量，击杀/存活/超时结算
- 简单的pygame环境(有加速度和惯性)

### 1.1 动作空间设计：Discrete 9 方向移动 + Multi-Binary 技能

移动用 9 选 1 Discrete，技能用 Multi-Binary（sigmoid），分别适配各自的分布特性：

```
动作输出 = 移动(9选1) + 技能1(0/1) + 技能2(0/1)
```

| 动作头 | 类型 | 维度 | 可选动作 | 采样方式 |
|--------|------|------|----------|----------|
| 移动头 | Discrete | 9 | 停 / W / A / S / D / WA / WD / SA / SD | softmax → Categorical |
| 技能头 | Multi-Binary | 2 | 技能1释放/不释放, 技能2释放/不释放 | sigmoid → Bernoulli |

**为什么移动用 Discrete，技能用 Multi-Binary：**

- 移动：W/S 互斥（不能同时上下），Discrete 9 枚举所有有效组合，杜绝 W+S 等无效输出
- 技能：不释放占比 >95%，Discrete 3 分类下模型可通过"总是预测不释放"获得高准确率；Multi-Binary + sigmoid 通过 `pos_weight` 放大释放动作（正样本）的损失权重，强制模型关注稀缺的释放时机
- 两个技能同时释放（很罕见）：在后期训练中增加惩罚项让模型避免浪费，不需要在动作空间中枚举

```python
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

MOVE_ACTIONS = ["STOP", "W", "A", "S", "D", "WA", "WD", "SA", "SD"]  # 9 个

class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.move_head = nn.Linear(hidden_dim, 9)   # 移动: 9选1
        self.skill_head = nn.Linear(hidden_dim, 2)  # 技能: 2×独立二分类

    def forward(self, obs):
        x = self.shared(obs)
        return self.move_head(x), self.skill_head(x)

    def get_action(self, obs):
        move_logits, skill_logits = self.forward(obs)

        # 移动: 9选1 Discrete
        move_dist = Categorical(logits=move_logits)
        move_action = move_dist.sample()                                 # [B]
        move_log_prob = move_dist.log_prob(move_action)                  # [B]

        # 技能: 独立 Bernoulli, sigmoid 转为释放概率
        skill_probs = torch.sigmoid(skill_logits)                        # [B, 2]
        skill_dist = Bernoulli(probs=skill_probs)
        skill_action = skill_dist.sample()                               # [B, 2]
        skill_log_prob = skill_dist.log_prob(skill_action).sum(-1)       # [B]

        log_prob = move_log_prob + skill_log_prob
        return move_action, skill_action, log_prob
```

**技能损失加权（pos_weight）：** PPO 更新时对技能头的 BCELoss 设置 `pos_weight=释放帧占比的倒数`，例如不释放占 95%，则 `pos_weight=19`，让一次释放错误的损失相当于 19 次不释放错误，强制模型认真对待释放决策。

**双技能同时释放惩罚：** 训练后期对 `skill_action[:,0] * skill_action[:,1] == 1` 的帧施加额外负奖励，引导模型避免浪费技能。

### 2. PPO 训练管线

- 算法：PPO + 纯 MLP（MVP 先不加 GRU）
- 内置优势归一化，验证奖励容错性
- 训练超参数：clip_epsilon=0.2, n_epochs=10, rollout_steps=2048
- 使用 Matplotlib 记录训练曲线

### 3. 奖励配置系统

基础奖励分项：

| 分项 | 说明 |
|------|------|
| `damage_dealt` | 对敌方造成的伤害 |
| `damage_taken` | 自身受到的伤害（通常为负权重） |
| `survival_time` | 每帧存活奖励 |
| `distance_to_enemy` | 与敌方的距离 |
| `kill` | 击杀敌方 |

通过权重向量配置，支持归一化处理。

### 4. 核心假设验证实验

两组极端配置：

| 配置 | damage_dealt | damage_taken | survival | kill | 预期行为 |
|------|-------------|-------------|----------|------|----------|
| 激进型 | +2.0 | -0.1 | 0 | +10.0 | 主动追击、频繁攻击 |
| 保守型 | +0.5 | -5.0 | +1.0 | +2.0 | 保持距离、避免承伤 |

验证指标：
- 平均每局击杀数
- 平均与敌方距离
- 平均存活时间
- 训练曲线收敛性与稳定性

通过标准：两组在关键指标上有统计学显著差异（p < 0.05），行为模式肉眼可辨识。

---

## 执行顺序

```
搭建最小环境 + 规则Bot
    ↓
实现 PPO 训练管线（纯 MLP + GPU）
    ↓
实现奖励配置（5 个基础分项）
    ↓
训练激进型模型 ─┬─ 行为对比分析 → 核心假设验证
训练保守型模型 ─┘
    ↓ (验证通过后)
加 GRU + 基础域随机化
    ↓
多智能体联合训练
```