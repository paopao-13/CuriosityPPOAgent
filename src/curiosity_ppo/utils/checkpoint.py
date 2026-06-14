"""模型保存/加载工具"""
import os
import torch


def save_checkpoint(path, agent_state, extra=None):
    """保存检查点
    agent_state: dict of {name: state_dict}
    extra: dict of additional info (step, metrics, etc.)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {'agent_state': agent_state}
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(path, device='cpu'):
    """加载检查点"""
    return torch.load(path, map_location=device, weights_only=False)


def get_latest_checkpoint(checkpoint_dir):
    """获取最新的检查点路径"""
    if not os.path.exists(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    return os.path.join(checkpoint_dir, files[-1])
