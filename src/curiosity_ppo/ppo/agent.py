"""CuriosityPPOAgent — 端到端好奇心 PPO 智能体

集成环境、网络、好奇心模块、PPO 训练器，实现完整训练循环:
collect_rollout → compute_gae → ppo_update + curiosity_update → log → checkpoint
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from curiosity_ppo.config import Config
from curiosity_ppo.networks.encoders import NatureDQNEncoder, CrafterEncoder
from curiosity_ppo.networks.policy import ActorCritic
from curiosity_ppo.networks.rnd import RNDNet
from curiosity_ppo.networks.icm import ICMNet
from curiosity_ppo.curiosity.icm_module import ICMCuriosity
from curiosity_ppo.curiosity.rnd_module import RNDCuriosity
from curiosity_ppo.curiosity.episodic_memory import EpisodicMemory
from curiosity_ppo.curiosity.ngu_fusion import NGUFusion
from curiosity_ppo.curiosity.reward_norm import RunningMeanStd, RewardNormalizer
from curiosity_ppo.ppo.rollout_buffer import RolloutBuffer
from curiosity_ppo.ppo.gae import compute_gae
from curiosity_ppo.ppo.ppo_trainer import PPOTrainer
from curiosity_ppo.utils.amp import AMPManager
from curiosity_ppo.utils.vram import get_vram_usage, empty_cache
from curiosity_ppo.utils.logger import TrainLogger
from curiosity_ppo.utils.checkpoint import save_checkpoint
from curiosity_ppo.utils.seed import set_seed


class CuriosityPPOAgent:
    """端到端好奇心 PPO 智能体

    Args:
        vec_env: 向量化环境 (DummyVecEnv 或 SubprocVecEnv)
        config: Config 对象
        device: 计算设备
    """

    def __init__(self, vec_env, config: Config, device='cpu', logger=None):
        self.vec_env = vec_env
        self.config = config
        self.device = device
        self.logger = logger or TrainLogger(
            project=config.wandb_project,
            config=config,
            use_wandb=False,
        )

        set_seed(config.seed)

        # 推断环境参数
        obs_shape = vec_env.observation_space.shape  # e.g. (64, 64, 3) or (4, 84, 84)
        n_actions = vec_env.action_space.n
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        # 判断通道位置: gymnasium 默认 HWC, 需转为 CHW
        if len(obs_shape) == 3:
            self.in_channels = obs_shape[2]  # HWC → C
            self.is_image = True
        else:
            self.in_channels = obs_shape[0]
            self.is_image = False

        # 根据环境选择编码器
        env_name = config.env.name.lower()
        if 'crafter' in env_name or 'minigrid' in env_name:
            encoder = CrafterEncoder(in_channels=self.in_channels, out_dim=config.icm.feature_dim)
            embed_dim = config.icm.feature_dim
            rnd_encoder_cls = CrafterEncoder
        else:
            encoder = NatureDQNEncoder(in_channels=self.in_channels, out_dim=512)
            embed_dim = 512
            rnd_encoder_cls = NatureDQNEncoder

        # Actor-Critic (双价值头)
        self.actor_critic = ActorCritic(encoder, n_actions, embed_dim=embed_dim).to(device)

        # AMP
        self.amp = AMPManager(enabled=config.use_amp, device=device)

        # PPO 训练器 (图像观测需要 HWC→CHW 转换)
        obs_preprocess = None
        if self.is_image:
            def obs_preprocess(x):
                return x.permute(0, 3, 1, 2)  # HWC → CHW
        self.ppo_trainer = PPOTrainer(self.actor_critic, config, device, self.amp, obs_preprocess=obs_preprocess)

        # 好奇心网络
        self.icm_net = None
        self.rnd_net = None
        self.icm_curiosity = None
        self.rnd_curiosity = None
        self.episodic_memory = None
        self.ngu_fusion = None
        self.icm_optimizer = None
        self.rnd_optimizer = None

        if config.icm.enabled:
            self.icm_net = ICMNet(
                in_channels=self.in_channels,
                action_dim=n_actions,
                feature_dim=config.icm.feature_dim,
                hidden_dim=config.icm.hidden_dim,
            ).to(device)
            self.icm_curiosity = ICMCuriosity(self.icm_net, eta=config.icm.eta, device=device)
            self.icm_optimizer = torch.optim.Adam(self.icm_net.parameters(), lr=config.ppo.lr)

        if config.rnd.enabled:
            self.rnd_net = RNDNet(
                in_channels=self.in_channels,
                output_dim=config.rnd.output_dim,
                encoder_cls=rnd_encoder_cls,
            ).to(device)
            # RND 归一化器使用 CHW 形状 (观测已由 _to_tensor 转换)
            rnd_obs_shape = (self.in_channels, obs_shape[0], obs_shape[1]) if self.is_image else obs_shape
            obs_normalizer = RunningMeanStd(shape=rnd_obs_shape) if config.rnd.obs_normalize else None
            self.rnd_curiosity = RNDCuriosity(
                self.rnd_net,
                obs_normalizer=obs_normalizer,
                L=config.episodic.L,
                device=device,
            )
            self.rnd_optimizer = torch.optim.Adam(self.rnd_net.predictor.parameters(), lr=config.ppo.lr)

        if config.episodic.enabled:
            self.episodic_memory = EpisodicMemory(
                capacity=config.episodic.capacity,
                dim=embed_dim,
                k=config.episodic.k,
                epsilon=config.episodic.epsilon,
                L=config.episodic.L,
            )

        # NGU 融合
        self.ngu_fusion = NGUFusion(
            config=config,
            icm=self.icm_curiosity,
            rnd=self.rnd_curiosity,
            episodic=self.episodic_memory,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=config.ppo.n_steps,
            n_envs=config.env.n_envs,
            obs_shape=obs_shape,
            action_dim=n_actions,
            device=device,
        )

        # 训练状态
        self.global_step = 0
        self.episode_count = 0
        self.current_obs = None

    def _to_tensor(self, obs):
        """numpy obs → torch tensor, HWC→CHW if image"""
        obs = np.asarray(obs, dtype=np.float32)
        t = torch.from_numpy(obs).to(self.device)
        if self.is_image and len(t.shape) == 4:  # (n_envs, H, W, C) → (n_envs, C, H, W)
            t = t.permute(0, 3, 1, 2)
        elif self.is_image and len(t.shape) == 3:  # (H, W, C) → (1, C, H, W)
            t = t.permute(2, 0, 1).unsqueeze(0)
        return t

    def _compute_intrinsic_reward(self, obs_np, action_np, next_obs_np, done_np):
        """计算内在奖励（每个环境独立）"""
        n_envs = obs_np.shape[0]
        int_rewards = np.zeros(n_envs, dtype=np.float32)

        for i in range(n_envs):
            s_t = self._to_tensor(obs_np[i:i+1])
            a = torch.tensor([action_np[i]], device=self.device, dtype=torch.long)
            s_next = self._to_tensor(next_obs_np[i:i+1])

            # 获取可控性嵌入 (用于情景记忆)
            controllable_emb = None
            if self.icm_curiosity:
                with torch.no_grad():
                    controllable_emb = self.icm_curiosity.get_embedding(s_t).cpu().numpy()[0]
            elif self.rnd_net:
                with torch.no_grad():
                    controllable_emb = self.rnd_net.target(s_next).cpu().numpy()[0]

            int_rewards[i] = self.ngu_fusion.compute(
                s_t=s_t, a=a, s_next=s_next, controllable_emb=controllable_emb
            )

            # 添加到情景记忆
            if self.episodic_memory and controllable_emb is not None:
                self.episodic_memory.add(controllable_emb)

            # episode 结束时重置情景记忆
            if done_np[i] and self.episodic_memory:
                self.episodic_memory.reset()

        return int_rewards

    def collect_rollout(self):
        """收集一轮 rollout 数据"""
        self.buffer.reset()
        if self.current_obs is None:
            self.current_obs = self.vec_env.reset()

        for step in range(self.config.ppo.n_steps):
            obs_tensor = self._to_tensor(self.current_obs)

            # 采样动作
            with torch.no_grad():
                action, logprob, v_ext, v_int = self.actor_critic.get_action(obs_tensor)

            action_np = action.cpu().numpy()
            # Step 环境
            next_obs, ext_reward, done, info = self.vec_env.step(action_np)

            # 计算内在奖励
            int_reward = self._compute_intrinsic_reward(
                self.current_obs, action_np, next_obs, done
            )

            # 存入 buffer
            self.buffer.add(
                obs=self.current_obs,
                action=action_np,
                logprob=logprob.cpu().numpy(),
                reward_ext=ext_reward,
                reward_int=int_reward,
                value_ext=v_ext.cpu().numpy(),
                value_int=v_int.cpu().numpy(),
                done=done,
            )

            self.current_obs = next_obs
            self.global_step += self.config.env.n_envs
            self.episode_count += int(sum(done))

        # 计算最后一步的 value 用于 GAE bootstrap
        with torch.no_grad():
            last_obs_tensor = self._to_tensor(self.current_obs)
            _, last_v_ext, last_v_int = self.actor_critic(last_obs_tensor)
            last_v_ext = last_v_ext.squeeze(-1).cpu().numpy()
            last_v_int = last_v_int.squeeze(-1).cpu().numpy()

        return last_v_ext, last_v_int

    def compute_advantages(self, last_v_ext, last_v_int):
        """计算双轨 GAE"""
        # 外在: 情景制, done 处截断
        adv_ext, ret_ext = compute_gae(
            rewards=self.buffer.rewards_ext,
            values=self.buffer.values_ext,
            last_value=last_v_ext,
            dones=self.buffer.dones,
            gamma=self.config.ppo.gamma_ext,
            gae_lambda=self.config.ppo.gae_lambda,
        )
        self.buffer.advantages_ext = adv_ext
        self.buffer.returns_ext = ret_ext

        # 内在: 非情景制, dones 全 0 (跨 episode)
        dones_int = np.zeros_like(self.buffer.dones)
        adv_int, ret_int = compute_gae(
            rewards=self.buffer.rewards_int,
            values=self.buffer.values_int,
            last_value=last_v_int,
            dones=dones_int,
            gamma=self.config.ppo.gamma_int,
            gae_lambda=self.config.ppo.gae_lambda,
        )
        self.buffer.advantages_int = adv_int
        self.buffer.returns_int = ret_int

    def update_curiosity(self):
        """更新好奇心网络 (ICM + RND)

        使用连续 (s_t, a, s_{t+1}) 三元组训练 ICM，
        使用观测训练 RND predictor。
        """
        metrics = {}

        # 构建连续 (s_t, a, s_{t+1}) 对
        n_steps, n_envs = self.buffer.actions.shape
        if n_steps < 2:
            return metrics

        # obs[t] → s_t, obs[t+1] → s_{t+1}
        s_t_all = self.buffer.obs[:-1]       # (n_steps-1, n_envs, *obs_shape)
        s_next_all = self.buffer.obs[1:]     # (n_steps-1, n_envs, *obs_shape)
        a_all = self.buffer.actions[:-1]     # (n_steps-1, n_envs)

        total = (n_steps - 1) * n_envs
        s_t_flat = s_t_all.reshape(total, *self.obs_shape)
        s_next_flat = s_next_all.reshape(total, *self.obs_shape)
        a_flat = a_all.reshape(total)

        # 随机采样 mini-batch
        batch_size = min(self.config.ppo.batch_size, total)
        indices = np.random.permutation(total)[:batch_size]

        s_t_batch = self._to_tensor(s_t_flat[indices])
        s_next_batch = self._to_tensor(s_next_flat[indices])
        a_batch = torch.tensor(a_flat[indices], device=self.device, dtype=torch.long)

        # ICM 更新 (inverse + forward loss)
        if self.icm_net is not None and self.config.icm.enabled:
            with self.amp.autocast():
                inverse_loss, forward_loss, phi_t = self.icm_net(s_t_batch, a_batch, s_next_batch)
                icm_loss = inverse_loss + forward_loss

            self.icm_optimizer.zero_grad()
            self.amp.scale_loss(icm_loss).backward()
            if self.config.ppo.max_grad_norm > 0:
                self.amp.unscale_(self.icm_optimizer)
                torch.nn.utils.clip_grad_norm_(self.icm_net.parameters(), self.config.ppo.max_grad_norm)
            self.icm_optimizer.step()

            metrics['icm_inverse_loss'] = inverse_loss.item()
            metrics['icm_forward_loss'] = forward_loss.item()

        # RND 更新 (predictor only, target frozen)
        if self.rnd_net is not None and self.config.rnd.enabled:
            with self.amp.autocast():
                pred_out, target_out = self.rnd_net(s_next_batch)
                rnd_loss = F.mse_loss(pred_out, target_out)

            self.rnd_optimizer.zero_grad()
            self.amp.scale_loss(rnd_loss).backward()
            if self.config.ppo.max_grad_norm > 0:
                self.amp.unscale_(self.rnd_optimizer)
                torch.nn.utils.clip_grad_norm_(self.rnd_net.predictor.parameters(), self.config.ppo.max_grad_norm)
            self.rnd_optimizer.step()

            metrics['rnd_loss'] = rnd_loss.item()

        return metrics

    def train_step(self):
        """执行一次完整的训练步骤"""
        # 1. 收集 rollout
        last_v_ext, last_v_int = self.collect_rollout()

        # 2. 计算 GAE
        self.compute_advantages(last_v_ext, last_v_int)

        # 3. PPO 更新
        ppo_metrics = self.ppo_trainer.update(self.buffer)

        # 4. 好奇心网络更新
        curiosity_metrics = self.update_curiosity()

        # 5. 清理缓存
        empty_cache()

        # 6. 汇总指标
        all_metrics = {**ppo_metrics, **curiosity_metrics}
        all_metrics['global_step'] = self.global_step
        all_metrics['episode_count'] = self.episode_count

        # VRAM 监控
        allocated, peak = get_vram_usage()
        all_metrics['vram_allocated_mb'] = allocated
        all_metrics['vram_peak_mb'] = peak

        # 奖励统计
        all_metrics['ext_reward_mean'] = float(self.buffer.rewards_ext.mean())
        all_metrics['int_reward_mean'] = float(self.buffer.rewards_int.mean())

        return all_metrics

    def train(self, total_steps=None, checkpoint_interval=10000, checkpoint_dir='results/checkpoints'):
        """完整训练循环"""
        total_steps = total_steps or self.config.env.total_steps
        log_interval = max(1, self.config.ppo.n_steps * self.config.env.n_envs)

        while self.global_step < total_steps:
            metrics = self.train_step()
            self.logger.log(metrics, step=self.global_step)

            # 定期保存检查点
            if self.global_step % checkpoint_interval < log_interval:
                path = f'{checkpoint_dir}/step_{self.global_step}.pt'
                state = {
                    'actor_critic': self.actor_critic.state_dict(),
                    'ppo_optimizer': self.ppo_trainer.optimizer.state_dict(),
                }
                if self.icm_net:
                    state['icm_net'] = self.icm_net.state_dict()
                if self.rnd_net:
                    state['rnd_net'] = self.rnd_net.state_dict()
                save_checkpoint(path, state, extra={'step': self.global_step, 'metrics': metrics})

        self.logger.finish()

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        """推理时选择动作"""
        obs_tensor = self._to_tensor(obs)
        logits, _, _ = self.actor_critic(obs_tensor)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action.cpu().numpy()

    def save(self, path):
        """保存模型"""
        state = {
            'actor_critic': self.actor_critic.state_dict(),
            'config': self.config,
        }
        if self.icm_net:
            state['icm_net'] = self.icm_net.state_dict()
        if self.rnd_net:
            state['rnd_net'] = self.rnd_net.state_dict()
        save_checkpoint(path, state, extra={'step': self.global_step})

    def load(self, path):
        """加载模型"""
        from curiosity_ppo.utils.checkpoint import load_checkpoint
        ckpt = load_checkpoint(path, self.device)
        self.actor_critic.load_state_dict(ckpt['agent_state']['actor_critic'])
        if self.icm_net and 'icm_net' in ckpt['agent_state']:
            self.icm_net.load_state_dict(ckpt['agent_state']['icm_net'])
        if self.rnd_net and 'rnd_net' in ckpt['agent_state']:
            self.rnd_net.load_state_dict(ckpt['agent_state']['rnd_net'])
        self.global_step = ckpt.get('step', 0)
