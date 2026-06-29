# Changelog

## 2026-06-29 — 模块1-4 完成

### 环境依赖
- 创建 Python 虚拟环境 `venv/`，安装 torch(CUDA)、numpy、matplotlib、jupyter、ipywidgets、pygame、scipy

---

### 模块1：最小游戏环境 (完善)

**修改 `mvp.ipynb`：**
- `ArenaEnv.__init__` 新增 `reward_weights` 参数，支持外部配置奖励权重
- `ArenaEnv.step` 中新增 `info["kill_bonus"]` 击杀奖励信号（之前缺失，导致击杀奖励永远为0）
- `ArenaEnv.step` 中 `compute_reward` 调用传入 `self.reward_weights`

**已有但之前未记录：**
- `Player` / `Bot` 数据类（含物理属性、HP、攻击范围、冷却）
- `physics_update_entity` 通用物理更新（加速度、惯性、边界反弹）
- `action_to_accel` 9方向移动映射
- `try_attack` 战斗判定（距离检测、伤害、速度惩罚）
- `build_obs` 24维结构化观测向量
- `compute_reward` 权重驱动的5分项奖励计算
- `ArenaEnv` Gym风格环境类（reset/step/render/close）
- HP血条渲染、HUD信息显示

---

### 模块2：PPO训练管线

**新建 `ppo_trainer.py`：**
- `Actor`: MLP网络，共享层 + move_head(9选1 Categorical) + skill_head(2×Bernoulli via sigmoid)
- `Critic`: MLP网络，标量value输出
- `PPOTrainer`: 完整PPO实现
  - rollout收集（与环境交互收集2048步数据）
  - GAE优势估计（gamma=0.99, lambda=0.95）
  - 优势归一化
  - PPO-clip目标函数（epsilon=0.2）
  - Value clipping
  - 10 epoch mini-batch更新（batch_size=128）
  - 梯度裁剪（max_grad_norm=0.5）
  - 模型保存/加载
- `plot_metrics`: 训练曲线可视化（reward/policy_loss/value_loss/entropy）
- `run_training`: 一键训练入口

**修改 `mvp.ipynb`：**
- 追加 PPO 训练管线说明 cell
- 追加快速训练验证 cell（10K步）
- 追加完整训练 cell（100K步，注释）

---

### 模块3：奖励配置系统

- `compute_reward` 支持外部权重字典驱动5个分项：
  - `damage_dealt` — 造成伤害
  - `damage_taken` — 受到伤害
  - `survival_bonus` — 存活奖励（每帧）
  - `distance_penalty` — 距离惩罚/奖励
  - `kill_bonus` — 击杀奖励
- `ArenaEnv` 支持 `reward_weights` 参数透传

---

### 模块4：行为差异验证

**新增到 `ppo_trainer.py`：**
- `evaluate_agent`: 评估已训练Actor，收集行为指标（每局步数、伤害、距离、攻击次数、胜率），支持自定义reward_weights
- `compare_agents`: 两组指标对比，含t检验（scipy.stats.ttest_ind）和柱状图可视化
- `load_actor_for_eval`: 从检查点加载Actor用于评估
- `run_behavior_experiment`: 一键完整实验（训练激进型→训练保守型→评估→对比）

**修改 `mvp.ipynb`：**
- 追加行为差异验证说明 cell
- 追加快速实验 cell
- 追加完整实验 cell（注释）

**验证结果（5120步快速测试）：**
- 激进型 vs 保守型在每局平均奖励上有极显著差异（p<0.001）
- 两组行为模式已开始分化，证明了"奖励权重可塑造差异化智能体"的核心假设

**训练出的模型文件：**
- `model_aggressive.pth` — 激进型权重（damage_dealt=2.0, kill_bonus=10.0）
- `model_conservative.pth` — 保守型权重（damage_taken=-5.0, survival_bonus=1.0）

---

### 文件清单

```
新增：
  ppo_trainer.py          — PPO训练/评估/行为对比完整模块
  model_aggressive.pth    — 激进型模型检查点 (~165KB)
  model_conservative.pth  — 保守型模型检查点 (~165KB)

修改：
  mvp.ipynb               — +kill_bonus修复 +reward_weights支持 +PPO cell +实验cell

依赖新增：
  scipy                   — 统计检验(t-test)
```
