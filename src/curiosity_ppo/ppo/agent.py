"""CuriosityPPOAgent — 端到端好奇心 PPO 智能体

集成环境、网络、好奇心模块、PPO 训练器，实现完整训练循环:
collect_rollout → compute_gae → ppo_update + curiosity_update → log → checkpoint
"""
import math
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
import sys as _sys


def _maybe_compile(model):
    """torch.compile 包装 (Windows 无 Triton 时自动跳过, 零影响)。"""
    if _sys.platform == "win32":
        return model  # Windows: Triton 不可用, inductor 后端会在首次前向时报错
    try:
        return torch.compile(model)
    except Exception:
        return model


class CuriosityPPOAgent:
    """端到端好奇心 PPO 智能体

    Args:
        vec_env: 向量化环境 (DummyVecEnv 或 SubprocVecEnv)
        config: Config 对象
        device: 计算设备
        logger: 训练日志器
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

        # cuDNN 自动选择最优卷积算法 (数学等价, 零数据改动, 提速 5~15%)
        torch.backends.cudnn.benchmark = True

        # 推断环境参数
        obs_shape = vec_env.observation_space.shape
        n_actions = vec_env.action_space.n
        n_envs = config.env.n_envs
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.n_envs = n_envs

        # 判断通道位置: gymnasium 默认 HWC, 需转为 CHW
        # 但 Atari FrameStack(channels_first=True) 已输出 CHW (如 (4,84,84)),
        # 其最后一维=4 <= 通道数, 不应再转置; 仅当最后一维 > 4 才视为 HWC.
        if len(obs_shape) == 3 and obs_shape[2] <= 4:
            self.in_channels = obs_shape[2]
            self.is_image = True
        elif len(obs_shape) == 3:
            self.in_channels = obs_shape[0]
            self.is_image = False
        else:
            self.in_channels = obs_shape[0]
            self.is_image = False

        # 根据环境选择编码器
        env_name = config.env.name.lower()
        use_crafter_encoder = 'crafter' in env_name or 'minigrid' in env_name
        if use_crafter_encoder:
            encoder = CrafterEncoder(in_channels=self.in_channels, out_dim=config.icm.feature_dim)
            embed_dim = config.icm.feature_dim
            rnd_encoder_cls = CrafterEncoder
            icm_encoder_cls = CrafterEncoder
        else:
            encoder = NatureDQNEncoder(in_channels=self.in_channels, out_dim=512)
            embed_dim = 512
            rnd_encoder_cls = NatureDQNEncoder
            icm_encoder_cls = NatureDQNEncoder

        # 策略网络观测归一化 (RunningMeanStd, 在线更新)
        self.policy_obs_rms = RunningMeanStd(shape=obs_shape) if self.is_image else None

        # Actor-Critic (双价值头)
        self.actor_critic = ActorCritic(encoder, n_actions, embed_dim=embed_dim).to(device)
        if config.use_compile:
            # torch.compile: 纯计算图优化, 数学等价, 零数据改动 (提速 10~30%)
            # Windows 无 Triton 时 _maybe_compile 自动跳过
            self.actor_critic = _maybe_compile(self.actor_critic)

        # AMP
        self.amp = AMPManager(enabled=config.use_amp, device=device)

        # PPO 训练器 (obs_preprocess: HWC→CHW + 归一化)
        agent = self  # closure 捕获
        if self.is_image:
            def obs_preprocess(x):
                x = x.permute(0, 3, 1, 2)  # HWC → CHW
                if agent.policy_obs_rms:
                    # 归一化: (obs - mean) / (std + 1e-8), mean/std 是 HWC 形状
                    mean = torch.tensor(agent.policy_obs_rms.mean, device=x.device, dtype=x.dtype)
                    std = torch.tensor(agent.policy_obs_rms.std, device=x.device, dtype=x.dtype)
                    # mean/std 是 HWC, 需转为 CHW
                    mean = mean.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                    std = std.permute(2, 0, 1).unsqueeze(0)
                    x = (x - mean) / (std + 1e-8)
                return x
        else:
            obs_preprocess = None
        self.ppo_trainer = PPOTrainer(
            self.actor_critic, config, device, self.amp,
            obs_preprocess=obs_preprocess,
            total_steps=config.env.total_steps,
        )

        # 好奇心网络
        self.icm_net = None
        self.rnd_net = None
        self.icm_curiosity = None
        self.rnd_curiosity = None
        self.ngu_fusion = None
        self.icm_optimizer = None
        self.icm_forward_optimizer = None
        self.rnd_optimizer = None

        if config.icm.enabled:
            # ICM 编码器与环境匹配
            self.icm_net = ICMNet(
                in_channels=self.in_channels,
                action_dim=n_actions,
                feature_dim=config.icm.feature_dim,
                hidden_dim=config.icm.hidden_dim,
                encoder_cls=icm_encoder_cls,
            ).to(device)
            if config.use_compile:
                self.icm_net = _maybe_compile(self.icm_net)
            self.icm_curiosity = ICMCuriosity(self.icm_net, eta=config.icm.eta, device=device)
            # 双优化器: inverse → encoder+inverse_model, forward → forward_model only
            self.icm_optimizer = torch.optim.Adam(
                list(self.icm_net.encoder.parameters()) +
                list(self.icm_net.inverse_model.parameters()),
                lr=config.ppo.lr,
            )
            self.icm_forward_optimizer = torch.optim.Adam(
                self.icm_net.forward_model.parameters(),
                lr=config.ppo.lr,
            )

        if config.rnd.enabled:
            self.rnd_net = RNDNet(
                in_channels=self.in_channels,
                output_dim=config.rnd.output_dim,
                predictor_hidden=config.rnd.predictor_hidden,
                encoder_cls=rnd_encoder_cls,
            ).to(device)
            if config.use_compile:
                self.rnd_net = _maybe_compile(self.rnd_net)
            rnd_obs_shape = (self.in_channels, obs_shape[0], obs_shape[1]) if self.is_image else obs_shape
            obs_normalizer = RunningMeanStd(shape=rnd_obs_shape) if config.rnd.obs_normalize else None
            self.rnd_curiosity = RNDCuriosity(
                self.rnd_net,
                obs_normalizer=obs_normalizer,
                L=config.episodic.L,
                device=device,
            )
            self.rnd_optimizer = torch.optim.Adam(self.rnd_net.predictor.parameters(), lr=config.ppo.lr)

        # 每个 env 独立的 EpisodicMemory
        # 嵌入维度: ICM 启用时用 ICM feature_dim, 否则用 RND output_dim
        if config.icm.enabled:
            episodic_dim = config.icm.feature_dim
        elif config.rnd.enabled:
            episodic_dim = config.rnd.output_dim
        else:
            episodic_dim = embed_dim

        if config.episodic.enabled:
            self.episodic_memories = [
                EpisodicMemory(
                    capacity=config.episodic.capacity,
                    dim=episodic_dim,
                    k=config.episodic.k,
                    epsilon=config.episodic.epsilon,
                    L=config.episodic.L,
                )
                for _ in range(n_envs)
            ]
        else:
            self.episodic_memories = None

        # 内在奖励归一化器 (防止 ICM/RND reward 爆炸 → NaN)
        self.int_reward_normalizer = RewardNormalizer()

        # NGU 融合 (episodic=None, 通过 episodic_override 逐环境传入)
        self.ngu_fusion = NGUFusion(
            config=config,
            icm=self.icm_curiosity,
            rnd=self.rnd_curiosity,
            episodic=None,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=config.ppo.n_steps,
            n_envs=n_envs,
            obs_shape=obs_shape,
            action_dim=n_actions,
            device=device,
        )

        # 训练状态
        self.global_step = 0
        self.episode_count = 0
        self.current_obs = None

    def _to_tensor(self, obs):
        """numpy obs → torch tensor, HWC→CHW if image (不做归一化)"""
        obs = np.asarray(obs, dtype=np.float32)
        t = torch.from_numpy(obs).to(self.device)
        if self.is_image and len(t.shape) == 4:  # (n_envs, H, W, C) → (n_envs, C, H, W)
            t = t.permute(0, 3, 1, 2)
        elif self.is_image and len(t.shape) == 3:  # (H, W, C) → (1, C, H, W)
            t = t.permute(2, 0, 1).unsqueeze(0)
        return t

    def _normalize_policy_obs(self, obs_tensor):
        """策略网络观测归一化 (CHW 格式)"""
        if self.policy_obs_rms is None:
            return obs_tensor
        mean = torch.tensor(self.policy_obs_rms.mean, device=obs_tensor.device, dtype=obs_tensor.dtype)
        std = torch.tensor(self.policy_obs_rms.std, device=obs_tensor.device, dtype=obs_tensor.dtype)
        if self.is_image and len(mean.shape) == 3:
            # HWC → CHW
            mean = mean.permute(2, 0, 1)
            std = std.permute(2, 0, 1)
        return (obs_tensor - mean) / (std + 1e-8)

    def _compute_intrinsic_reward(self, obs_np, action_np, next_obs_np, done_np):
        """Batch 计算内在奖励 (一次前向所有 env)

        每个 env 使用独立的 EpisodicMemory
        """
        n_envs = obs_np.shape[0]
        int_rewards = np.zeros(n_envs, dtype=np.float32)

        # Batch 前向: 所有 env 一次传入网络
        s_t_batch = self._to_tensor(obs_np)         # (n_envs, C, H, W)
        a_batch = torch.tensor(action_np, device=self.device, dtype=torch.long)
        s_next_batch = self._to_tensor(next_obs_np)  # (n_envs, C, H, W)

        # Batch 获取可控性嵌入
        controllable_embs = None
        if self.icm_curiosity:
            with torch.no_grad():
                controllable_embs = self.icm_curiosity.get_embedding(s_t_batch).cpu().numpy()
        elif self.rnd_net:
            with torch.no_grad():
                controllable_embs = self.rnd_net.target(s_next_batch).cpu().numpy()

        # Batch ICM 前向 (如果启用)
        r_icm_batch = np.zeros(n_envs, dtype=np.float32)
        if self.config.icm.enabled and self.icm_curiosity:
            with torch.no_grad():
                inverse_loss, forward_loss, _ = self.icm_net(s_t_batch, a_batch, s_next_batch)
                # forward_loss 是标量 (mean over batch), 广播到每个 env
                raw_icm = (self.config.icm.eta * forward_loss).item()
                r_icm_batch[:] = min(raw_icm, 10.0)  # 裁剪 ICM 原始奖励 (防爆炸)

        # Per-env: episodic memory + RND alpha (kNN 是 CPU numpy, 快)
        for i in range(n_envs):
            epi = self.episodic_memories[i] if self.episodic_memories else None
            emb = controllable_embs[i] if controllable_embs is not None else None

            # NGU 融合 (episodic_override 实现多环境隔离)
            r_ngu = 0.0
            if self.config.episodic.enabled and epi is not None and emb is not None:
                r_epi = epi.compute_reward(emb)
                if self.config.rnd.enabled and self.rnd_curiosity:
                    alpha = self.rnd_curiosity.compute_alpha(s_next_batch[i:i+1])
                else:
                    alpha = 1.0
                r_ngu = r_epi * min(max(alpha, 1.0), float(self.config.episodic.L))
            elif self.config.rnd.enabled and self.rnd_curiosity:
                r_ngu = self.rnd_curiosity.compute_reward(s_next_batch[i:i+1])

            int_rewards[i] = r_icm_batch[i] + r_ngu

            # 添加到对应 env 的情景记忆
            if epi is not None and emb is not None:
                epi.add(emb)

            # 只重置该 env 的情景记忆
            if done_np[i] and epi is not None:
                epi.reset()

        # 内在奖励归一化 + 裁剪 (防止 ICM/RND 爆炸 → NaN)
        # RewardNormalizer: reward / (running_std + 1e-8), clip 到 [-10, 10]
        normalized = self.int_reward_normalizer(int_rewards)
        return normalized

    def collect_rollout(self):
        """收集一轮 rollout 数据"""
        self.buffer.reset()
        if self.current_obs is None:
            self.current_obs = self.vec_env.reset()

        for step in range(self.config.ppo.n_steps):
            obs_tensor = self._to_tensor(self.current_obs)

            # 更新观测归一化统计并归一化
            if self.policy_obs_rms is not None:
                self.policy_obs_rms.update(self.current_obs)
            obs_norm = self._normalize_policy_obs(obs_tensor)

            # 采样动作
            with torch.no_grad():
                action, logprob, v_ext, v_int = self.actor_critic.get_action(obs_norm)

            action_np = action.cpu().numpy()
            # Step 环境
            next_obs, ext_reward, done, info = self.vec_env.step(action_np)

            # Batch 计算内在奖励
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
            # 在 rollout 的每一步 (共 n_steps 次) 累加 n_envs 个环境步,
            # 整轮累加 = n_steps * n_envs, 即本 rollout 实际处理的环境步数。
            self.global_step += self.config.env.n_envs
            self.episode_count += int(sum(done))

        # 计算最后一步的 value 用于 GAE bootstrap
        with torch.no_grad():
            last_obs_tensor = self._to_tensor(self.current_obs)
            last_obs_norm = self._normalize_policy_obs(last_obs_tensor)
            _, last_v_ext, last_v_int = self.actor_critic(last_obs_norm)
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

        过滤跨 episode 的 (s_t, s_{t+1}) 对:
            当 dones[t]=True 时, obs[t+1] 是 reset 后的新 episode 首帧,
            用它作为 s_next 会给 ICM 注入噪声梯度。
        """
        metrics = {}

        n_steps, n_envs = self.buffer.actions.shape
        if n_steps < 2:
            return metrics

        # 构建有效转换掩码 — dones[t]=False 的 (s_t, s_{t+1}) 对才有效
        # dones[t] 表示 step t 是否结束, 如果结束则 obs[t+1] 是新 episode
        valid_mask = self.buffer.dones[:-1] == 0  # (n_steps-1, n_envs)

        # obs[t] → s_t, obs[t+1] → s_{t+1}
        s_t_all = self.buffer.obs[:-1]       # (n_steps-1, n_envs, *obs_shape)
        s_next_all = self.buffer.obs[1:]     # (n_steps-1, n_envs, *obs_shape)
        a_all = self.buffer.actions[:-1]     # (n_steps-1, n_envs)

        # 展平并筛选有效对
        total = (n_steps - 1) * n_envs
        s_t_flat = s_t_all.reshape(total, *self.obs_shape)
        s_next_flat = s_next_all.reshape(total, *self.obs_shape)
        a_flat = a_all.reshape(total)
        valid_flat = valid_mask.reshape(total)

        valid_indices = np.where(valid_flat)[0]
        if len(valid_indices) == 0:
            return metrics

        # 随机采样 mini-batch (仅从有效对中)
        batch_size = min(self.config.ppo.batch_size, len(valid_indices))
        sampled = np.random.choice(valid_indices, size=batch_size, replace=False)

        s_t_batch = self._to_tensor(s_t_flat[sampled])
        s_next_batch = self._to_tensor(s_next_flat[sampled])
        a_batch = torch.tensor(a_flat[sampled], device=self.device, dtype=torch.long)

        # ICM 更新 — 双优化器: inverse_loss → encoder + inverse_model,
        # forward_loss → forward_model only (梯度隔离, 防编码器发散)
        if self.icm_net is not None and self.config.icm.enabled:
            with self.amp.autocast():
                inverse_loss, forward_loss, phi_t = self.icm_net(
                    s_t_batch, a_batch, s_next_batch,
                    detach_phi_t_for_forward=True,
                )
                # forward_loss clip: 防止极端值导致梯度爆炸
                forward_loss_clipped = torch.clamp(forward_loss, max=10.0)

            # 优化器 1: inverse_loss → encoder + inverse_model (训练可控特征)
            self.icm_optimizer.zero_grad()
            self.amp.scale_loss(inverse_loss).backward(retain_graph=True)
            if self.config.ppo.max_grad_norm > 0:
                self.amp.unscale_(self.icm_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.icm_net.encoder.parameters()) +
                    list(self.icm_net.inverse_model.parameters()),
                    self.config.ppo.max_grad_norm,
                )
            self.amp.step(self.icm_optimizer)

            # 优化器 2: forward_loss → forward_model only (探索信号, 不影响编码器)
            self.icm_forward_optimizer.zero_grad()
            self.amp.scale_loss(forward_loss_clipped).backward()
            if self.config.ppo.max_grad_norm > 0:
                self.amp.unscale_(self.icm_forward_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.icm_net.forward_model.parameters(),
                    self.config.ppo.max_grad_norm,
                )
            self.amp.step(self.icm_forward_optimizer)
            self.amp.update()

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
            self.amp.update()  # 同上, 重置 GradScaler 状态

            metrics['rnd_loss'] = rnd_loss.item()

        return metrics

    def train_step(self):
        """执行一次完整的训练步骤"""
        # 学习率衰减
        self.ppo_trainer.update_lr(self.global_step)

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

        # NaN/inf 安全检测: 若任一核心 loss 为 NaN/inf, 重置好奇心网络并跳过本轮
        nan_detected = False
        for key in ('policy_loss', 'value_ext_loss', 'icm_forward_loss', 'rnd_loss'):
            val = all_metrics.get(key, 0.0)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                all_metrics[key] = 0.0
                nan_detected = True

        if nan_detected:
            all_metrics['nan_detected'] = True
            # 重置 buffer
            self.buffer.reset()
            empty_cache()
            # 重置好奇心网络到初始状态 (防止 NaN 权重持续传播)
            if self.icm_net is not None:
                for layer in self.icm_net.encoder.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                for layer in self.icm_net.inverse_model.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                for layer in self.icm_net.forward_model.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                # 重置 ICM 优化器状态 (旧 momentum 等已失效)
                self.icm_optimizer = torch.optim.Adam(
                    list(self.icm_net.encoder.parameters()) +
                    list(self.icm_net.inverse_model.parameters()),
                    lr=self.config.ppo.lr,
                )
                self.icm_forward_optimizer = torch.optim.Adam(
                    self.icm_net.forward_model.parameters(),
                    lr=self.config.ppo.lr,
                )
                # 重置内在奖励归一化器
                self.int_reward_normalizer = RewardNormalizer()
            if self.rnd_net is not None:
                for layer in self.rnd_net.predictor.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                self.rnd_optimizer = torch.optim.Adam(
                    self.rnd_net.predictor.parameters(), lr=self.config.ppo.lr,
                )
            # 重置情景记忆 (旧的嵌入来自 NaN 模型)
            if self.episodic_memories:
                for mem in self.episodic_memories:
                    mem.reset()
            all_metrics['ext_reward_mean'] = 0.0
            all_metrics['int_reward_mean'] = 0.0
            return all_metrics

        # VRAM 监控
        allocated, peak = get_vram_usage()
        all_metrics['vram_allocated_mb'] = allocated
        all_metrics['vram_peak_mb'] = peak

        # 奖励统计
        all_metrics['ext_reward_mean'] = float(self.buffer.rewards_ext.mean())
        all_metrics['int_reward_mean'] = float(self.buffer.rewards_int.mean())

        return all_metrics

    def evaluate(self, n_episodes=100, max_steps=10000):
        """训练中评测, 返回 success rate (到达终点的 episode 比例)

        使用确定性贪心策略 (argmax)。success 判定: 累积外在奖励(含塑形)
        含 goal_reward(=1.0), 即 >=1.0 视为到达终点 (拿钥匙+开门=0.8 < 1.0)。

        Args:
            n_episodes: 评测 episode 数量.
            max_steps: 最大评测步数上限, 防止无限循环.
        """
        eval_obs = self.vec_env.reset()
        eval_rewards = np.zeros(self.n_envs, dtype=np.float32)
        completed = 0
        successes = 0
        step_count = 0

        while completed < n_episodes and step_count < max_steps:
            obs_tensor = self._to_tensor(eval_obs)
            obs_norm = self._normalize_policy_obs(obs_tensor)
            with torch.no_grad():
                logits, _, _ = self.actor_critic(obs_norm)
                action = logits.argmax(dim=-1)
            eval_obs, reward, done, info = self.vec_env.step(action.cpu().numpy())
            eval_rewards += reward
            step_count += 1
            for i in range(self.n_envs):
                if done[i]:
                    if eval_rewards[i] >= 1.0:  # 到达终点 (goal_reward=1.0)
                        successes += 1
                    completed += 1
                    eval_rewards[i] = 0.0
                    if completed >= n_episodes:
                        break

        return successes / completed if completed > 0 else 0.0

    def train(self, total_steps=None, checkpoint_interval=10000, checkpoint_dir='results/checkpoints',
              eval_interval=50000, n_eval_episodes=100):
        """完整训练循环

        定期评测, 记录 eval_score
        学习率线性衰减
        """
        total_steps = total_steps or self.config.env.total_steps
        log_interval = max(1, self.config.ppo.n_steps * self.config.env.n_envs)

        while self.global_step < total_steps:
            metrics = self.train_step()
            self.logger.log(metrics, step=self.global_step)

            # 定期评测
            if self.global_step % eval_interval < log_interval:
                eval_reward = self.evaluate(n_episodes=n_eval_episodes)
                self.logger.log({'eval_score': eval_reward}, step=self.global_step)

            # 定期保存检查点
            if self.global_step % checkpoint_interval < log_interval:
                path = f'{checkpoint_dir}/step_{self.global_step}.pt'
                state = {
                    'actor_critic': self.actor_critic.state_dict(),
                    'ppo_optimizer': self.ppo_trainer.optimizer.state_dict(),
                }
                if self.icm_net:
                    state['icm_net'] = self.icm_net.state_dict()
                    if self.icm_optimizer is not None:
                        state['icm_optimizer'] = self.icm_optimizer.state_dict()
                    if self.icm_forward_optimizer is not None:
                        state['icm_forward_optimizer'] = self.icm_forward_optimizer.state_dict()
                if self.rnd_net:
                    state['rnd_net'] = self.rnd_net.state_dict()
                    if self.rnd_optimizer is not None:
                        state['rnd_optimizer'] = self.rnd_optimizer.state_dict()
                save_checkpoint(path, state, extra={'step': self.global_step, 'metrics': metrics})

        self.logger.finish()

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        """推理时选择动作"""
        obs_tensor = self._to_tensor(obs)
        obs_norm = self._normalize_policy_obs(obs_tensor)
        logits, _, _ = self.actor_critic(obs_norm)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action.cpu().numpy()

    def save(self, path):
        """保存模型 (含优化器状态, 便于续训)"""
        state = {
            'actor_critic': self.actor_critic.state_dict(),
            'config': self.config,
            'ppo_optimizer': self.ppo_trainer.optimizer.state_dict(),
        }
        # 观测归一化统计量: 评测/续训必须恢复, 否则输入分布漂移导致性能假阴性
        if self.policy_obs_rms is not None:
            state['policy_obs_rms'] = self.policy_obs_rms
        if self.icm_net:
            state['icm_net'] = self.icm_net.state_dict()
            if self.icm_optimizer is not None:
                state['icm_optimizer'] = self.icm_optimizer.state_dict()
            if self.icm_forward_optimizer is not None:
                state['icm_forward_optimizer'] = self.icm_forward_optimizer.state_dict()
        if self.rnd_net:
            state['rnd_net'] = self.rnd_net.state_dict()
            if self.rnd_optimizer is not None:
                state['rnd_optimizer'] = self.rnd_optimizer.state_dict()
        save_checkpoint(path, state, extra={'step': self.global_step})

    def load(self, path):
        """加载模型 (含优化器状态, 支持断点续训)"""
        from curiosity_ppo.utils.checkpoint import load_checkpoint
        ckpt = load_checkpoint(path, self.device)
        st = ckpt.get('agent_state', {})
        self.actor_critic.load_state_dict(st['actor_critic'])
        if self.icm_net and 'icm_net' in st:
            self.icm_net.load_state_dict(st['icm_net'])
        if self.rnd_net and 'rnd_net' in st:
            self.rnd_net.load_state_dict(st['rnd_net'])
        # 恢复优化器状态 (缺失时静默跳过, 兼容旧检查点)
        if 'ppo_optimizer' in st and hasattr(self.ppo_trainer, 'optimizer'):
            self.ppo_trainer.optimizer.load_state_dict(st['ppo_optimizer'])
        if self.icm_optimizer is not None and 'icm_optimizer' in st:
            self.icm_optimizer.load_state_dict(st['icm_optimizer'])
        if self.icm_forward_optimizer is not None and 'icm_forward_optimizer' in st:
            self.icm_forward_optimizer.load_state_dict(st['icm_forward_optimizer'])
        if self.rnd_optimizer is not None and 'rnd_optimizer' in st:
            self.rnd_optimizer.load_state_dict(st['rnd_optimizer'])
        # 恢复观测归一化统计量 (评测/续训关键)
        if 'policy_obs_rms' in st and self.policy_obs_rms is not None:
            self.policy_obs_rms = st['policy_obs_rms']
        self.global_step = ckpt.get('step', 0)
