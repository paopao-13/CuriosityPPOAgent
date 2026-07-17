"""AMP (Automatic Mixed Precision) FP16 封装 (免费 GPU 安全版)."""
import torch


class AMPManager:
    """FP16 自动混合精度封装, 默认禁用以避免 unscale 顺序问题."""

    def __init__(self, enabled=False, device='cpu'):
        self.enabled = False

    def autocast(self):
        return torch.autocast(device_type='cpu', enabled=False)

    def scale_loss(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass
