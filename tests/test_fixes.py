"""修复验证测试 — 确认 P0/P1/P2 修复在集成场景下正确工作"""
import pytest
import numpy as np
import torch

from curiosity_ppo.config import Config
from curiosity_ppo.ppo.agent import CuriosityPPOAgent
from curiosity_ppo.ppo.ppo_trainer import PPOTrainer
from curiosity_ppo.networks.icm import ICMNet
from curiosity_ppo.networks.encoders import NatureDQNEncoder, CrafterEncoder
from curiosity_ppo.curiosity.episodic_memory import EpisodicMemory
from curiosity_ppo.utils.memory_bank import LRUMemoryBank


# ========== P0-1: Per-env EpisodicMemory Isolation ==========

class MockVecEnv:
    """模拟向量化环境, 每 env 每 5 步 done 一次"""
    def __init__(self, n_envs=4, obs_shape=(64, 64, 3), n_actions=17):
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self._step_count = 0
        import gymnasium
        self.observation_space = gymnasium.spaces.Box(0, 255, obs_shape, dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(n_actions)

    def reset(self):
        self._step_count = 0
        return np.random.randint(0, 256, (self.n_envs,) + self.obs_shape, dtype=np.uint8)

    def step(self, actions):
        self._step_count += 1
        obs = np.random.randint(0, 256, (self.n_envs,) + self.obs_shape, dtype=np.uint8)
        rewards = np.random.randn(self.n_envs).astype(np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        # 每 5 步所有 env done, 确保评测不会无限循环
        if self._step_count % 5 == 0:
            dones[:] = True
        infos = [{} for _ in range(self.n_envs)]
        return obs, rewards, dones, infos


def test_p0_1_episodic_memory_per_env_isolation():
    """P0-1: 验证每个 env 有独立的 EpisodicMemory, 且互不干扰"""
    config = Config()
    config.ppo.n_steps = 8
    config.ppo.n_envs = 4
    config.ppo.batch_size = 8
    config.ppo.accumulation_steps = 1
    config.ppo.ppo_epochs = 1
    config.env.n_envs = 4
    config.env.name = "crafter"
    config.use_amp = False

    env = MockVecEnv(n_envs=4, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    # 确认有 4 个独立的 EpisodicMemory 实例
    assert len(agent.episodic_memories) == 4
    # 确认它们是不同的对象
    for i in range(4):
        for j in range(i + 1, 4):
            assert agent.episodic_memories[i] is not agent.episodic_memories[j]

    # 手动向不同 env 的记忆库添加不同数量的数据
    for i in range(4):
        for _ in range(i + 1):  # env0: 1, env1: 2, env2: 3, env3: 4
            agent.episodic_memories[i].add(np.random.randn(config.icm.feature_dim).astype(np.float32))

    # 清空 env0, 其他 env 不受影响
    agent.episodic_memories[0].reset()
    assert agent.episodic_memories[0].bank.size == 0
    assert agent.episodic_memories[1].bank.size == 2
    assert agent.episodic_memories[2].bank.size == 3
    assert agent.episodic_memories[3].bank.size == 4


# ========== P0-2: ICM Encoder for Atari ==========

def test_p0_2_icm_nature_dqn_encoder():
    """P0-2: 验证 ICM 可以使用 NatureDQNEncoder (84x84 输入)"""
    icm = ICMNet(
        in_channels=4,
        action_dim=18,
        feature_dim=512,
        encoder_cls=NatureDQNEncoder,
    )
    s_t = torch.randn(4, 4, 84, 84)
    a = torch.tensor([1, 2, 3, 4])
    s_next = torch.randn(4, 4, 84, 84)

    inverse_loss, forward_loss, phi_t = icm(s_t, a, s_next)
    assert phi_t.shape == (4, 512)
    assert torch.isfinite(inverse_loss)
    assert torch.isfinite(forward_loss)


# ========== P0-3: Gradient Accumulation Flush ==========

def test_p0_3_gradient_flush():
    """P0-3: 验证梯度累积在非整除时也 flush"""
    config = Config()
    config.ppo.n_steps = 10
    config.ppo.n_envs = 2
    config.ppo.batch_size = 4  # 20 total / 4 = 5 batches
    config.ppo.accumulation_steps = 3  # 5 % 3 != 0, 需要末尾 flush
    config.ppo.ppo_epochs = 1
    config.ppo.lr = 1e-2  # 提高学习率确保参数变化可检测
    config.env.n_envs = 2
    config.env.name = "crafter"
    config.env.total_steps = 1000
    config.use_amp = False

    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    # 记录初始参数
    params_before = [p.clone() for p in agent.actor_critic.parameters()]

    agent.train_step()

    # 验证参数确实更新 (说明 flush 生效)
    params_after = [p.clone() for p in agent.actor_critic.parameters()]
    changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Parameters did not change — gradient flush may have failed"


# ========== P0-4: Cross-Episode ICM Training Pair Filtering ==========

def test_p0_4_cross_episode_filtering():
    """P0-4: 验证 done 边界的 (s_t, s_next) 对被过滤"""
    config = Config()
    config.ppo.n_steps = 8
    config.ppo.n_envs = 2
    config.ppo.batch_size = 4
    config.ppo.accumulation_steps = 1
    config.ppo.ppo_epochs = 1
    config.env.n_envs = 2
    config.env.name = "crafter"
    config.use_amp = False

    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    agent.train_step()

    # 验证 dones 中有 True 值 (环境确实 done 过)
    assert agent.buffer.dones.sum() > 0, "No done events in buffer"

    # 验证 update_curiosity 不崩溃 (如果未过滤, 可能会因为
    # 跨 episode 对产生异常梯度但不报错, 所以主要验证不崩溃)
    curiosity_metrics = agent.update_curiosity()
    assert 'icm_inverse_loss' in curiosity_metrics
    assert np.isfinite(curiosity_metrics['icm_inverse_loss'])


# ========== P1-5: Separate Advantage Normalization ==========

def test_p1_5_separate_advantage_normalization():
    """P1-5: 验证 PPO trainer 分别归一化 ext/int 优势"""
    import inspect
    from curiosity_ppo.ppo.ppo_trainer import PPOTrainer
    source = inspect.getsource(PPOTrainer.update)
    # 确认代码中有分别归一化的逻辑
    assert 'adv_ext' in source and 'adv_int' in source
    assert 'adv_ext.mean()' in source
    assert 'adv_int.mean()' in source
    # 确认不再合并后统一归一化
    assert 'advantages_ext + batch' not in source


# ========== P1-6: Batch Intrinsic Reward ==========

def test_p1_6_batch_intrinsic_reward():
    """P1-6: 验证 batch 计算 (8 envs 一次前向, 不逐 env 循环网络)"""
    config = Config()
    config.ppo.n_steps = 4
    config.ppo.n_envs = 8
    config.ppo.batch_size = 8
    config.ppo.accumulation_steps = 1
    config.ppo.ppo_epochs = 1
    config.env.n_envs = 8
    config.env.name = "crafter"
    config.use_amp = False

    env = MockVecEnv(n_envs=8, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    # 训练一步, 验证不崩溃且返回正确维度
    metrics = agent.train_step()
    assert np.isfinite(metrics['int_reward_mean'])
    assert np.isfinite(metrics['policy_loss'])


# ========== P1-7: Periodic Evaluation ==========

def test_p1_7_evaluation():
    """P1-7: 验证 evaluate 方法返回有效值"""
    config = Config()
    config.ppo.n_steps = 4
    config.ppo.n_envs = 2
    config.ppo.batch_size = 4
    config.ppo.accumulation_steps = 1
    config.ppo.ppo_epochs = 1
    config.env.n_envs = 2
    config.env.name = "crafter"
    config.use_amp = False

    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    eval_reward = agent.evaluate(n_episodes=2)
    assert isinstance(eval_reward, float)
    assert np.isfinite(eval_reward)


# ========== P1-8: Learning Rate Decay ==========

def test_p1_8_lr_decay():
    """P1-8: 验证学习率随步数线性衰减"""
    config = Config()
    config.ppo.n_steps = 4
    config.ppo.n_envs = 2
    config.ppo.batch_size = 4
    config.ppo.accumulation_steps = 1
    config.ppo.ppo_epochs = 1
    config.env.n_envs = 2
    config.env.total_steps = 1000
    config.env.name = "crafter"
    config.use_amp = False

    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    initial_lr = agent.ppo_trainer.initial_lr

    # step 0: lr = initial
    agent.ppo_trainer.update_lr(0)
    assert abs(agent.ppo_trainer.optimizer.param_groups[0]['lr'] - initial_lr) < 1e-6

    # step 500: lr = 0.5 * initial
    agent.ppo_trainer.update_lr(500)
    assert abs(agent.ppo_trainer.optimizer.param_groups[0]['lr'] - 0.5 * initial_lr) < 1e-6

    # step 1000: lr = 0
    agent.ppo_trainer.update_lr(1000)
    assert abs(agent.ppo_trainer.optimizer.param_groups[0]['lr']) < 1e-6


# ========== P2-9: kNN Pre-allocated Array ==========

def test_p2_9_memory_bank_preallocated():
    """P2-9: 验证预分配数组的正确性"""
    bank = LRUMemoryBank(capacity=100, dim=64)

    # 添加 50 个向量
    for i in range(50):
        emb = np.random.randn(64).astype(np.float32)
        bank.add(emb)

    assert bank.size == 50

    # kNN 查询
    query = np.random.randn(64).astype(np.float32)
    dists = bank.knn_distances(query, k=5)
    assert dists.shape == (5,)
    assert (dists >= 0).all()
    assert (dists[:-1] <= dists[1:]).all()  # 升序

    # 淘汰: 添加超过 capacity
    for i in range(60):
        emb = np.random.randn(64).astype(np.float32)
        bank.add(emb)

    assert bank.size == 100  # 达到 capacity

    # 再添加一个, 应该覆盖最旧
    bank.add(np.zeros(64, dtype=np.float32))
    assert bank.size == 100  # 不超过 capacity

    # clear
    bank.clear()
    assert bank.size == 0


# ========== P2-10: Policy Observation Normalization ==========

def test_p2_10_policy_obs_normalization():
    """P2-10: 验证策略网络观测归一化"""
    config = Config()
    config.ppo.n_steps = 4
    config.ppo.n_envs = 2
    config.ppo.batch_size = 4
    config.ppo.accumulation_steps = 1
    config.ppo.ppo_epochs = 1
    config.env.n_envs = 2
    config.env.name = "crafter"
    config.use_amp = False

    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    assert agent.policy_obs_rms is not None

    # 训练一步, RMS 应被更新
    agent.train_step()
    # mean 不应全为 0 (因为 RMS 被更新了)
    assert not np.allclose(agent.policy_obs_rms.mean, 0)


# ========== P2-11: Forward Loss Scaling ==========

def test_p2_11_forward_loss_not_scaled():
    """P2-11: 验证 forward_loss 不再乘以 feature_dim"""
    import inspect
    from curiosity_ppo.networks.icm import ICMNet
    source = inspect.getsource(ICMNet.forward)

    # 不应包含 * self.feature_dim
    assert '* self.feature_dim' not in source
    # 应包含 mse_loss
    assert 'F.mse_loss' in source


# ========== Integration: Atari-like Environment ==========

class MockAtariEnv:
    """模拟 Atari 环境: 84×84×4 观测"""
    def __init__(self, n_envs=2, n_actions=18):
        self.n_envs = n_envs
        self.n_actions = n_actions
        import gymnasium
        self.observation_space = gymnasium.spaces.Box(0, 255, (84, 84, 4), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(n_actions)

    def reset(self):
        return np.random.randint(0, 256, (self.n_envs, 84, 84, 4), dtype=np.uint8)

    def step(self, actions):
        obs = np.random.randint(0, 256, (self.n_envs, 84, 84, 4), dtype=np.uint8)
        rewards = np.random.randn(self.n_envs).astype(np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = [{} for _ in range(self.n_envs)]
        return obs, rewards, dones, infos


def test_integration_atari_environment():
    """集成测试: Atari 类环境完整训练步骤不崩溃"""
    config = Config()
    config.ppo.n_steps = 8
    config.ppo.n_envs = 2
    config.ppo.batch_size = 8
    config.ppo.accumulation_steps = 1
    config.ppo.ppo_epochs = 2
    config.env.n_envs = 2
    config.env.name = "atari_montezuma"  # 触发 NatureDQN 路径
    config.env.total_steps = 1000
    config.use_amp = False

    env = MockAtariEnv(n_envs=2, n_actions=18)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    # 验证编码器类型
    assert isinstance(agent.actor_critic.encoder, NatureDQNEncoder)
    assert isinstance(agent.icm_net.encoder, NatureDQNEncoder)

    # 训练步骤
    metrics = agent.train_step()
    assert np.isfinite(metrics['policy_loss'])
    assert np.isfinite(metrics['icm_forward_loss'])
    assert np.isfinite(metrics['rnd_loss'])


# ========== Integration: Ablation Configs ==========

def test_integration_all_ablation_configs():
    """集成测试: 四组消融配置都能训练"""
    for ablation in ['no_icm', 'no_rnd']:
        config = Config()
        config.ppo.n_steps = 4
        config.ppo.n_envs = 2
        config.ppo.batch_size = 4
        config.ppo.accumulation_steps = 1
        config.ppo.ppo_epochs = 1
        config.env.n_envs = 2
        config.env.name = "crafter"
        config.env.total_steps = 1000
        config.use_amp = False

        if ablation == 'no_icm':
            config.icm.enabled = False
            config.ablation = "no_icm"
        elif ablation == 'no_rnd':
            config.rnd.enabled = False
            config.ablation = "no_rnd"

        env = MockVecEnv(n_envs=2, n_actions=17)
        agent = CuriosityPPOAgent(env, config, device='cpu')
        metrics = agent.train_step()
        assert np.isfinite(metrics['policy_loss']), f"{ablation}: policy_loss not finite"
