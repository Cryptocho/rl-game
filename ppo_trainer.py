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
import matplotlib.pyplot as plt

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
    plt.show()


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
