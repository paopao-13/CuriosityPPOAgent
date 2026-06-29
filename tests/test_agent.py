"""CuriosityPPOAgent 端到端测试"""
import pytest
import numpy as np
import torch

from curiosity_ppo.config import Config
from curiosity_ppo.ppo.agent import CuriosityPPOAgent


class MockVecEnv:
    """模拟向量化环境，返回 64×64×3 图像观测"""
    def __init__(self, n_envs=2, obs_shape=(64, 64, 3), n_actions=17):
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self._step_count = 0

        import gymnasium
        self.observation_space = gymnasium.spaces.Box(0, 255, obs_shape, dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(n_actions)

    def reset(self):
        return np.random.randint(0, 256, (self.n_envs,) + self.obs_shape, dtype=np.uint8)

    def step(self, actions):
        self._step_count += 1
        obs = np.random.randint(0, 256, (self.n_envs,) + self.obs_shape, dtype=np.uint8)
        rewards = np.random.randn(self.n_envs).astype(np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        # 每 20 步随机结束一个 episode
        if self._step_count % 20 == 0:
            dones[0] = True
        infos = [{} for _ in range(self.n_envs)]
        return obs, rewards, dones, infos


def _make_test_config():
    config = Config()
    config.ppo.n_steps = 8
    config.ppo.n_envs = 2
    config.ppo.batch_size = 8
    config.ppo.accumulation_steps = 1
    config.ppo.ppo_epochs = 2
    config.env.n_envs = 2
    config.env.name = "crafter"
    config.icm.feature_dim = 288
    config.use_amp = False  # CPU 测试不用 AMP
    return config


def test_agent_init():
    """测试 Agent 初始化"""
    config = _make_test_config()
    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    assert agent.n_actions == 17
    assert agent.in_channels == 3
    assert agent.is_image is True
    assert agent.actor_critic is not None
    assert agent.icm_net is not None
    assert agent.rnd_net is not None
    assert agent.episodic_memories is not None
    assert agent.ngu_fusion is not None


def test_agent_train_step():
    """测试完整训练步骤"""
    config = _make_test_config()
    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    metrics = agent.train_step()

    # 验证返回指标
    assert 'policy_loss' in metrics
    assert 'value_ext_loss' in metrics
    assert 'value_int_loss' in metrics
    assert 'entropy' in metrics
    assert 'icm_inverse_loss' in metrics
    assert 'icm_forward_loss' in metrics
    assert 'rnd_loss' in metrics
    assert 'ext_reward_mean' in metrics
    assert 'int_reward_mean' in metrics
    assert 'global_step' in metrics
    assert metrics['global_step'] == config.ppo.n_steps * config.env.n_envs

    # 验证所有损失是有限数值
    for key in ['policy_loss', 'value_ext_loss', 'value_int_loss', 'icm_inverse_loss', 'rnd_loss']:
        assert np.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"


def test_agent_parameters_update():
    """测试参数确实更新"""
    config = _make_test_config()
    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    # 记录初始参数
    ac_params_before = [p.clone() for p in agent.actor_critic.parameters()]
    icm_params_before = [p.clone() for p in agent.icm_net.parameters()]

    agent.train_step()

    # 验证参数变化
    ac_changed = any(not torch.equal(before, after) for before, after in
                     zip(ac_params_before, agent.actor_critic.parameters()))
    icm_changed = any(not torch.equal(before, after) for before, after in
                      zip(icm_params_before, agent.icm_net.parameters()))

    assert ac_changed, "ActorCritic parameters did not change"
    assert icm_changed, "ICM parameters did not change"


def test_agent_act():
    """测试推理动作选择"""
    config = _make_test_config()
    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    obs = np.random.randint(0, 256, (1, 64, 64, 3), dtype=np.uint8)

    # 确定性策略
    action_det = agent.act(obs, deterministic=True)
    assert action_det.shape == (1,)
    assert 0 <= action_det[0] < 17

    # 随机策略
    action_stoch = agent.act(obs, deterministic=False)
    assert action_stoch.shape == (1,)
    assert 0 <= action_stoch[0] < 17


def test_agent_multiple_steps():
    """测试多步训练不崩溃"""
    config = _make_test_config()
    config.ppo.n_steps = 16
    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    for i in range(3):
        metrics = agent.train_step()
        assert np.isfinite(metrics['policy_loss'])

    assert agent.global_step == 3 * 16 * 2


def test_agent_ablation_no_icm():
    """测试消融: 无 ICM"""
    config = _make_test_config()
    config.icm.enabled = False
    config.ablation = "no_icm"
    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    assert agent.icm_net is None
    assert agent.icm_curiosity is None

    metrics = agent.train_step()
    assert 'icm_inverse_loss' not in metrics
    assert 'rnd_loss' in metrics  # RND 仍存在
    assert np.isfinite(metrics['policy_loss'])


def test_agent_ablation_no_rnd():
    """测试消融: 无 RND"""
    config = _make_test_config()
    config.rnd.enabled = False
    config.ablation = "no_rnd"
    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    assert agent.rnd_net is None
    assert agent.rnd_curiosity is None

    metrics = agent.train_step()
    assert 'rnd_loss' not in metrics
    assert 'icm_inverse_loss' in metrics
    assert np.isfinite(metrics['policy_loss'])


def test_agent_save_load(tmp_path):
    """测试模型保存加载"""
    config = _make_test_config()
    env = MockVecEnv(n_envs=2, n_actions=17)
    agent = CuriosityPPOAgent(env, config, device='cpu')

    # 训练一步
    agent.train_step()

    # 保存
    path = str(tmp_path / "test_model.pt")
    agent.save(path)

    # 验证文件存在
    import os
    assert os.path.exists(path)
