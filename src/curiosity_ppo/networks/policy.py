"""Actor-Critic 策略网络, 带外在/内在双价值头 (PPO + RND/ICM)."""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """共享编码器 + Actor / critic_ext / critic_int 三头.

    - actor: Linear(embed_dim, 512) -> ReLU -> Linear(512, action_dim)  输出 logits
    - critic_ext: Linear(embed_dim, 512) -> ReLU -> Linear(512, 1)      外在价值
    - critic_int: Linear(embed_dim, 512) -> ReLU -> Linear(512, 1)      内在价值
    """

    def __init__(self, encoder: nn.Module, action_dim: int, embed_dim: int = 512):
        super().__init__()
        self.encoder = encoder

        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        self.critic_ext = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.critic_int = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, obs: torch.Tensor):
        """返回 (logits, v_ext, v_int).

        - logits: (N, action_dim)
        - v_ext:  (N, 1)
        - v_int:  (N, 1)
        """
        h = self.encoder(obs)
        logits = self.actor(h)
        v_ext = self.critic_ext(h)
        v_int = self.critic_int(h)
        return logits, v_ext, v_int

    def get_action(self, obs: torch.Tensor):
        """采样动作, 返回 (action, logprob, v_ext.squeeze(-1), v_int.squeeze(-1)).

        action 为 long, 用于索引; logprob / value 已 squeeze 到 (N,).
        """
        logits, v_ext, v_int = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, v_ext.squeeze(-1), v_int.squeeze(-1)
