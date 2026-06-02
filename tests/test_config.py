import pytest
from curiosity_ppo.config import Config, load_config


def test_default_config():
    config = Config()
    assert config.ppo.lr == 1e-4
    assert config.ppo.clip_range == 0.1
    assert config.rnd.output_dim == 512
    assert config.icm.feature_dim == 288
    assert config.episodic.k == 5


def test_yaml_override(tmp_path):
    yaml_content = """
ppo:
  lr: 0.0003
  clip_range: 0.2
"""
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    config = load_config(str(yaml_file))
    assert config.ppo.lr == 0.0003
    assert config.ppo.clip_range == 0.2
    # 未覆盖的保持默认
    assert config.rnd.output_dim == 512


def test_ablation_config():
    config = Config()
    config.ablation = "no_icm"
    assert config.ablation == "no_icm"


def test_full_config_fields():
    config = Config()
    assert config.ppo.gamma_ext == 0.999
    assert config.ppo.gamma_int == 0.99
    assert config.ppo.batch_size == 128
    assert config.ppo.accumulation_steps == 4
    assert config.icm.eta == 0.2
    assert config.episodic.L == 5
    assert config.episodic.capacity == 10000
    assert config.use_amp is True
    assert config.seed == 42
