"""AMP (Automatic Mixed Precision) FP16 封装."""
import torch


class AMPManager:
    """FP16 自动混合精度封装"""

    def __init__(self, enabled=True, device='cpu'):
        self.enabled = enabled and torch.cuda.is_available()
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.enabled)

    def autocast(self):
        if self.enabled:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            return torch.autocast(device_type='cpu', enabled=False)

    def scale_loss(self, loss):
        return self.scaler.scale(loss)

    def step(self, optimizer):
        self.scaler.step(optimizer)

    def update(self):
        self.scaler.update()

    def unscale_(self, optimizer):
        self.scaler.unscale_(optimizer)
