from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class PPOConfig:
    lr: float = 1e-4
    clip_range: float = 0.1
    ppo_epochs: int = 4
    gamma_ext: float = 0.999
    gamma_int: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 128
    accumulation_steps: int = 4
    n_steps: int = 128
    n_envs: int = 8


@dataclass
class RNDConfig:
    enabled: bool = True
    output_dim: int = 512
    predictor_hidden: int = 512
    obs_normalize: bool = True
    reward_normalize: bool = True


@dataclass
class ICMConfig:
    enabled: bool = True
    feature_dim: int = 288
    eta: float = 0.2
    hidden_dim: int = 256


@dataclass
class EpisodicConfig:
    enabled: bool = True
    k: int = 5
    epsilon: float = 1e-3
    L: int = 5
    capacity: int = 10000


@dataclass
class EnvConfig:
    name: str = "crafter"
    n_envs: int = 8
    total_steps: int = 1000000
    vec_env_type: str = "dummy"  # "dummy" | "subproc"


@dataclass
class Config:
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rnd: RNDConfig = field(default_factory=RNDConfig)
    icm: ICMConfig = field(default_factory=ICMConfig)
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ablation: str = "full"
    use_amp: bool = True
    use_compile: bool = False  # torch.compile 网络 (数学等价提速, 零数据改动)
    seed: int = 42
    wandb_project: str = "curiosity-ppo"


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    config = Config()
    if data is None:
        return config
    for section_name, section_data in data.items():
        if hasattr(config, section_name) and isinstance(section_data, dict):
            section = getattr(config, section_name)
            for key, value in section_data.items():
                if hasattr(section, key):
                    setattr(section, key, value)
        elif hasattr(config, section_name):
            setattr(config, section_name, section_data)
    return config
