"""PPO 训练器: 双价值头 + AMP + 梯度累积."""
import numpy as np
import torch
import torch.nn as nn

from curiosity_ppo.ppo.rollout_buffer import RolloutBuffer
from curiosity_ppo.ppo.gae import compute_gae
from curiosity_ppo.utils.amp import AMPManager


class PPOTrainer:
    """PPO 训练器：双价值头 + AMP + 梯度累积"""

    def __init__(self, actor_critic, config, device='cpu', amp_manager=None, obs_preprocess=None):
        self.actor_critic = actor_critic.to(device)
        self.config = config
        self.device = device
        self.amp = amp_manager or AMPManager(enabled=config.use_amp, device=device)
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.ppo.lr)
        self.batch_size = config.ppo.batch_size
        self.accumulation_steps = config.ppo.accumulation_steps
        self.ppo_epochs = config.ppo.ppo_epochs
        self.clip_range = config.ppo.clip_range
        self.ent_coef = config.ppo.ent_coef
        self.vf_coef = config.ppo.vf_coef
        self.max_grad_norm = config.ppo.max_grad_norm
        self.obs_preprocess = obs_preprocess

    def update(self, buffer: RolloutBuffer):
        """执行 PPO 更新，返回训练指标 dict"""
        metrics = {'policy_loss': 0, 'value_ext_loss': 0, 'value_int_loss': 0,
                   'entropy': 0, 'clip_fraction': 0, 'n_updates': 0}
        total_batches = 0

        for epoch in range(self.ppo_epochs):
            for step, batch in enumerate(buffer.get_mini_batches(self.batch_size)):
                with self.amp.autocast():
                    obs = batch['obs']
                    if self.obs_preprocess:
                        obs = self.obs_preprocess(obs)
                    logits, v_ext, v_int = self.actor_critic(obs)
                    v_ext = v_ext.squeeze(-1)
                    v_int = v_int.squeeze(-1)

                    dist = torch.distributions.Categorical(logits=logits)
                    logprobs = dist.log_prob(batch['actions'])
                    entropy = dist.entropy().mean()

                    # PPO ratio
                    ratio = torch.exp(logprobs - batch['logprobs'])
                    # 合并双轨优势
                    advantages = batch['advantages_ext'] + batch['advantages_int']
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # PPO clipped objective
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()

                    # 双价值损失
                    v_ext_loss = 0.5 * ((v_ext - batch['returns_ext']) ** 2).mean()
                    v_int_loss = 0.5 * ((v_int - batch['returns_int']) ** 2).mean()

                    loss = policy_loss + self.vf_coef * (v_ext_loss + v_int_loss) - self.ent_coef * entropy
                    loss = loss / self.accumulation_steps

                self.amp.scale_loss(loss).backward()

                if (step + 1) % self.accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        self.amp.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.amp.step(self.optimizer)
                    self.optimizer.zero_grad()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_ext_loss'] += v_ext_loss.item()
                metrics['value_int_loss'] += v_int_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['clip_fraction'] += clip_fraction.item()
                total_batches += 1

        for k in ['policy_loss', 'value_ext_loss', 'value_int_loss', 'entropy', 'clip_fraction']:
            metrics[k] /= max(total_batches, 1)
        metrics['n_updates'] = total_batches
        return metrics
