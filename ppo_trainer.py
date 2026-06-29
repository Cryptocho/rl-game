"""
PPO Trainer for RL Arena - Module 2

Actor: MLP with 9-discrete move head + 2-binary skill head
Critic: MLP with scalar value output
PPOTrainer: Full PPO with GAE, advantage normalization, value clipping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import numpy as np
from collections import deque
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

# 配置中文字体
for _font in ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PPO] Using device: {device}")


class Actor(nn.Module):
    """MLP actor: 9-discrete (move) + 2-binary (skills)"""

    def __init__(self, obs_dim=24, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.move_head = nn.Linear(hidden_dim, 9)
        self.skill_head = nn.Linear(hidden_dim, 2)

    def forward(self, obs):
        x = self.shared(obs)
        return self.move_head(x), self.skill_head(x)

    def get_action(self, obs, deterministic=False):
        move_logits, skill_logits = self.forward(obs)

        move_dist = Categorical(logits=move_logits)
        if deterministic:
            move_action = move_logits.argmax(dim=-1)
        else:
            move_action = move_dist.sample()
        move_log_prob = move_dist.log_prob(move_action)

        skill_probs = torch.sigmoid(skill_logits)
        skill_dist = Bernoulli(probs=skill_probs)
        if deterministic:
            skill_action = (skill_probs > 0.5).float()
        else:
            skill_action = skill_dist.sample()
        skill_log_prob = skill_dist.log_prob(skill_action).sum(-1)

        log_prob = move_log_prob + skill_log_prob
        return move_action, skill_action, log_prob

    def evaluate(self, obs, move_action, skill_action):
        move_logits, skill_logits = self.forward(obs)

        move_dist = Categorical(logits=move_logits)
        move_log_prob = move_dist.log_prob(move_action)
        move_entropy = move_dist.entropy()

        skill_probs = torch.sigmoid(skill_logits)
        skill_dist = Bernoulli(probs=skill_probs)
        skill_log_prob = skill_dist.log_prob(skill_action).sum(-1)
        skill_entropy = skill_dist.entropy().sum(-1)

        log_prob = move_log_prob + skill_log_prob
        entropy = move_entropy + skill_entropy
        return log_prob, entropy


class Critic(nn.Module):
    """MLP critic: scalar value output"""

    def __init__(self, obs_dim=24, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class PPOTrainer:
    """PPO trainer with GAE, advantage normalization, value clipping"""

    def __init__(
        self,
        env,
        actor: Actor,
        critic: Critic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 128,
    ):
        self.env = env
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def collect_rollout(self, steps: int):
        obs_list, move_acts, skill_acts = [], [], []
        log_probs, rewards, values, dones = [], [], [], []
        episode_rewards = []
        episode_reward = 0.0

        obs = self.env.reset()
        for _ in range(steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                value = self.critic(obs_t)
                move_a, skill_a, log_p = self.actor.get_action(obs_t)

            move_idx = move_a.item()
            sk0 = int(skill_a[0, 0].item())
            sk1 = int(skill_a[0, 1].item())

            next_obs, reward, done, info = self.env.step((move_idx, sk0, sk1))
            episode_reward += reward

            obs_list.append(obs)
            move_acts.append(move_idx)
            skill_acts.append([sk0, sk1])
            log_probs.append(log_p.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs = self.env.reset()
            else:
                obs = next_obs

        if not dones[-1]:
            with torch.no_grad():
                last_obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
                last_value = self.critic(last_obs).item()
        else:
            last_value = 0.0

        obs_t = torch.FloatTensor(np.array(obs_list)).to(device)
        move_t = torch.LongTensor(move_acts).to(device)
        skill_t = torch.FloatTensor(skill_acts).to(device)
        logp_t = torch.FloatTensor(log_probs).to(device)
        rew_t = torch.FloatTensor(rewards).to(device)
        val_t = torch.FloatTensor(values).to(device)
        done_t = torch.FloatTensor(dones).to(device)

        advantages, returns = self._compute_gae(rew_t, val_t, done_t, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (obs_t, move_t, skill_t, logp_t, advantages, returns, episode_rewards)

    def update(self, rollout_data):
        obs, move_a, skill_a, old_logp, advantages, returns, _ = rollout_data
        n = len(obs)
        batches_per_epoch = max(1, n // self.batch_size)

        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.n_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                b_obs = obs[idx]
                b_move = move_a[idx]
                b_skill = skill_a[idx]
                b_old_logp = old_logp[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]

                new_logp, entropy = self.actor.evaluate(b_obs, b_move, b_skill)
                values = self.critic(b_obs)

                ratio = torch.exp(new_logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = b_ret + torch.clamp(
                    values - b_ret, -self.clip_epsilon, self.clip_epsilon
                )
                v_unclipped = (values - b_ret) ** 2
                v_clipped = (value_pred_clipped - b_ret) ** 2
                value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
        }

    def train(self, total_steps: int, rollout_steps: int = 2048, log_interval: int = 1):
        metrics = {"reward": [], "policy_loss": [], "value_loss": [], "entropy": []}
        all_ep_rewards = []
        iteration = 0
        steps_done = 0

        while steps_done < total_steps:
            rollout = self.collect_rollout(rollout_steps)
            upd = self.update(rollout)
            ep_rewards = rollout[-1]
            all_ep_rewards.extend(ep_rewards)

            steps_done += rollout_steps
            iteration += 1

            avg_r = np.mean(ep_rewards) if ep_rewards else 0.0
            metrics["reward"].append(avg_r)
            metrics["policy_loss"].append(upd["policy_loss"])
            metrics["value_loss"].append(upd["value_loss"])
            metrics["entropy"].append(upd["entropy"])

            if iteration % log_interval == 0:
                print(
                    f"Iter {iteration:4d} | Steps {steps_done:7d} | "
                    f"AvgR {avg_r:7.2f} | PL {upd['policy_loss']:.4f} | "
                    f"VL {upd['value_loss']:.4f} | Ent {upd['entropy']:.4f}"
                )

        return metrics

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Model loaded from {path}")


def plot_metrics(metrics: dict, title: str = "PPO Training"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    axes[0, 0].plot(metrics["reward"])
    axes[0, 0].set_title("Average Episode Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(metrics["policy_loss"])
    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(metrics["value_loss"])
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(metrics["entropy"])
    axes[1, 1].set_title("Entropy")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if matplotlib.get_backend() != "agg":
        plt.show()
    plt.close()


def run_training(
    env,
    total_steps: int = 100_000,
    rollout_steps: int = 2048,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    save_path: str | None = None,
    reward_weights: dict | None = None,
    log_interval: int = 1,
) -> dict:
    """
    一键训练入口。

    Args:
        env: ArenaEnv (不带 render_mode)
        total_steps: 总环境步数
        rollout_steps: 每次 rollout 收集的步数
        hidden_dim: MLP 隐藏层维度
        lr: 学习率
        save_path: 模型保存路径 (None 则不保存)
        reward_weights: 奖励权重字典, None 使用默认
        log_interval: 日志打印间隔 (iteration)

    Returns:
        metrics dict
    """
    actor = Actor(obs_dim=24, hidden_dim=hidden_dim)
    critic = Critic(obs_dim=24, hidden_dim=hidden_dim)
    trainer = PPOTrainer(env, actor, critic, lr=lr)

    if reward_weights:
        env.reward_weights = reward_weights

    metrics = trainer.train(total_steps, rollout_steps, log_interval)

    if save_path:
        trainer.save(save_path)

    return metrics


# ── 模块4：行为差异验证 ────────────────────────────────────────────

FIELD_W, FIELD_H = 700.0, 500.0
FIELD_DIAG = np.sqrt(FIELD_W**2 + FIELD_H**2)
MAX_STEPS = 3600


def evaluate_agent(actor, env_class, n_episodes=100, deterministic=True,
                   reward_weights=None):
    """
    评估已训练的 Actor，收集行为指标。

    返回: (summary_dict, raw_metrics_dict)
      summary 包含 avg_steps, avg_damage_dealt, avg_distances,
              win_rate, loss_rate, draw_rate 等
    """
    env = env_class(render_mode=None, reward_weights=reward_weights)
    actor.eval()

    raw = {
        "steps": [], "damage_dealt": [], "damage_taken": [],
        "wins": 0, "losses": 0, "draws": 0,
        "distances": [], "attacks": [], "rewards": [],
    }

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_steps = 0
        ep_dmg_d = 0.0
        ep_dmg_t = 0.0
        ep_rew = 0.0
        ep_dist = []
        ep_atk = 0

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                move_a, skill_a, _ = actor.get_action(obs_t, deterministic=deterministic)

            move_idx = move_a.item()
            sk0 = int(skill_a[0, 0].item())
            sk1 = int(skill_a[0, 1].item())

            obs, reward, done, info = env.step((move_idx, sk0, sk1))

            ep_steps += 1
            ep_rew += reward
            ep_dmg_d += info.get("damage_dealt", 0.0)
            ep_dmg_t += info.get("damage_taken", 0.0)
            if info.get("player_attack", False):
                ep_atk += 1

            dist = float(obs[16]) * FIELD_DIAG
            ep_dist.append(dist)

        winner = info.get("winner", "draw")
        if winner == "player":
            raw["wins"] += 1
        elif winner == "bot":
            raw["losses"] += 1
        else:
            raw["draws"] += 1

        raw["steps"].append(ep_steps)
        raw["damage_dealt"].append(ep_dmg_d)
        raw["damage_taken"].append(ep_dmg_t)
        raw["distances"].append(np.mean(ep_dist))
        raw["attacks"].append(ep_atk)
        raw["rewards"].append(ep_rew)

    env.close()

    summary = {"n_episodes": n_episodes}
    for key in ["steps", "damage_dealt", "damage_taken", "distances", "attacks", "rewards"]:
        arr = np.array(raw[key])
        summary[f"avg_{key}"] = float(np.mean(arr))
        summary[f"std_{key}"] = float(np.std(arr))

    summary["win_rate"] = raw["wins"] / n_episodes
    summary["loss_rate"] = raw["losses"] / n_episodes
    summary["draw_rate"] = raw["draws"] / n_episodes

    return summary, raw


def compare_agents(aggressive_summary, aggressive_raw,
                   conservative_summary, conservative_raw,
                   label_a="激进型", label_b="保守型"):
    """对比两个智能体的行为指标，包含统计检验和可视化。"""

    metrics_to_compare = [
        ("steps", "平均存活步数"),
        ("damage_dealt", "平均每局造成伤害"),
        ("damage_taken", "平均每局受到伤害"),
        ("distances", "平均与敌方距离"),
        ("attacks", "平均每局攻击次数"),
        ("rewards", "平均每局奖励"),
    ]

    print(f"\n{'='*60}")
    print(f"  行为差异验证: {label_a} vs {label_b}")
    print(f"{'='*60}")
    print(f"\n{'指标':<18} {label_a:>10} {label_b:>10} {'p-value':>10} {'显著':>6}")
    print(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    p_values = {}
    for key, name in metrics_to_compare:
        a_arr = np.array(aggressive_raw[key])
        b_arr = np.array(conservative_raw[key])
        t_stat, p_val = stats.ttest_ind(a_arr, b_arr, equal_var=False)
        p_values[key] = p_val
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{name:<18} {aggressive_summary[f'avg_{key}']:>10.2f} "
              f"{conservative_summary[f'avg_{key}']:>10.2f} {p_val:>10.4f} {sig:>6}")

    # Win rate
    print(f"\n{'胜率':<18} {aggressive_summary['win_rate']:>10.2%} "
          f"{conservative_summary['win_rate']:>10.2%}")
    print(f"{'负率':<18} {aggressive_summary['loss_rate']:>10.2%} "
          f"{conservative_summary['loss_rate']:>10.2%}")
    print(f"{'平局率':<18} {aggressive_summary['draw_rate']:>10.2%} "
          f"{conservative_summary['draw_rate']:>10.2%}")

    # Bar chart comparison
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Behavior Comparison: {label_a} vs {label_b}", fontsize=13)

    bar_keys = ["avg_steps", "avg_damage_dealt", "avg_damage_taken",
                "avg_distances", "avg_attacks", "avg_rewards"]
    bar_names = ["Survival Steps", "Damage Dealt", "Damage Taken",
                 "Avg Distance", "Attack Count", "Avg Reward"]
    bar_stds = ["std_steps", "std_damage_dealt", "std_damage_taken",
                "std_distances", "std_attacks", "std_rewards"]

    for ax, key, name, skey in zip(axes.flat, bar_keys, bar_names, bar_stds):
        vals = [aggressive_summary[key], conservative_summary[key]]
        errs = [aggressive_summary[skey], conservative_summary[skey]]
        bars = ax.bar([label_a, label_b], vals, yerr=errs, capsize=8,
                      color=["#FF6B6B", "#6BB5FF"], edgecolor="white")
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.3)
        sig_key = key.replace("avg_", "")
        if sig_key in p_values and p_values[sig_key] < 0.05:
            ax.text(0.5, 0.95, f"p={p_values[sig_key]:.3f} *",
                    transform=ax.transAxes, ha="center", fontsize=10, color="red")

    plt.tight_layout()
    if matplotlib.get_backend() != "agg":
        plt.show()
    plt.close()

    return p_values


def load_actor_for_eval(path, hidden_dim=128):
    """从检查点加载 Actor 用于评估。"""
    actor = Actor(obs_dim=24, hidden_dim=hidden_dim)
    ckpt = torch.load(path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.to(device)
    actor.eval()
    return actor


def run_behavior_experiment(env_class, total_steps=100_000, hidden_dim=128,
                            eval_episodes=100):
    """
    完整行为差异验证实验：
    1. 训练激进型模型
    2. 训练保守型模型
    3. 评估 + 对比
    """
    aggressive_weights = {
        "damage_dealt": 2.0,
        "damage_taken": -0.1,
        "survival_bonus": 0.0,
        "distance_penalty": 0.0,
        "kill_bonus": 10.0,
    }
    conservative_weights = {
        "damage_dealt": 0.5,
        "damage_taken": -5.0,
        "survival_bonus": 1.0,
        "distance_penalty": -0.5,
        "kill_bonus": 2.0,
    }

    print("=" * 60)
    print("  Module 4: Behavior Differentiation Experiment")
    print("=" * 60)

    # ── 训练激进型 ──
    print("\n[Phase 1] Training aggressive agent...")
    env_agg = env_class(render_mode=None)
    agg_metrics = run_training(
        env_agg, total_steps=total_steps, hidden_dim=hidden_dim,
        reward_weights=aggressive_weights, save_path="model_aggressive.pth",
    )
    env_agg.close()
    print("Aggressive agent training complete.\n")

    # ── 训练保守型 ──
    print("\n[Phase 2] Training conservative agent...")
    env_con = env_class(render_mode=None)
    con_metrics = run_training(
        env_con, total_steps=total_steps, hidden_dim=hidden_dim,
        reward_weights=conservative_weights, save_path="model_conservative.pth",
    )
    env_con.close()
    print("Conservative agent training complete.\n")

    # ── 评估 ──
    print("\n[Phase 3] Evaluating both agents...")
    actor_agg = load_actor_for_eval("model_aggressive.pth", hidden_dim)
    actor_con = load_actor_for_eval("model_conservative.pth", hidden_dim)

    agg_summary, agg_raw = evaluate_agent(
        actor_agg, env_class, n_episodes=eval_episodes,
        reward_weights=aggressive_weights,
    )
    con_summary, con_raw = evaluate_agent(
        actor_con, env_class, n_episodes=eval_episodes,
        reward_weights=conservative_weights,
    )

    # ── 对比 ──
    p_vals = compare_agents(agg_summary, agg_raw, con_summary, con_raw)

    # ── 训练曲线对比 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(agg_metrics["reward"], label="Aggressive", color="#FF6B6B")
    axes[0].plot(con_metrics["reward"], label="Conservative", color="#6BB5FF")
    axes[0].set_title("Training Reward Curves")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Avg Reward")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(agg_metrics["entropy"], label="Aggressive", color="#FF6B6B")
    axes[1].plot(con_metrics["entropy"], label="Conservative", color="#6BB5FF")
    axes[1].set_title("Entropy Curves")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Entropy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if matplotlib.get_backend() != "agg":
        plt.show()
    plt.close()

    return agg_summary, con_summary, p_vals
