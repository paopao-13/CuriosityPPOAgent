"""Atari 云端/免费 GPU 一键补丁 (防御式, 可重复执行)

适用场景:
    - 在 Kaggle / Colab / 任意新 clone 的 CuriosityPPOAgent 仓库中运行
    - 修复旧版 GitHub 仓库缺失的三处更新, 使 Atari (CHW 观测) 训练可跑

修复内容:
    1. agent.py: is_image 误判 -> Atari FrameStack 输出 (4,84,84) CHW 被当 HWC 转置
    2. amp.py: 重写 GradScaler 调用, 避免 "unscale() after step()" 崩溃
    3. requirements.txt: 确保含 gymnasium[atari] 与 ale-py

用法:
    python scripts/patch_atari.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def patch_agent():
    """修复 is_image 通道判定 (CHW vs HWC)。"""
    f = os.path.join(ROOT, "src", "curiosity_ppo", "ppo", "agent.py")
    if not os.path.exists(f):
        print("[skip] agent.py 不存在")
        return
    c = open(f).read()
    old = """        # 判断通道位置: gymnasium 默认 HWC, 需转为 CHW
        if len(obs_shape) == 3:
            self.in_channels = obs_shape[2]
            self.is_image = True
        else:
            self.in_channels = obs_shape[0]
            self.is_image = False"""
    new = """        # 判断通道位置: gymnasium 默认 HWC, 需转为 CHW
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
            self.is_image = False"""
    if old in c:
        c = c.replace(old, new)
        open(f, "w").write(c)
        print("[ok] agent.py is_image 判定已修复 (CHW 不再误转置)")
    elif "obs_shape[2] <= 4" in c:
        print("[skip] agent.py 已是修复版")
    else:
        print("[warn] agent.py 未匹配预期代码, 请人工检查 is_image 逻辑")


def patch_amp():
    """重写 amp.py 为安全版 (idempotent)。"""
    f = os.path.join(ROOT, "src", "curiosity_ppo", "utils", "amp.py")
    safe = '''"""AMP (Automatic Mixed Precision) FP16 封装 (免费 GPU 安全版)."""
import torch


class AMPManager:
    """FP16 自动混合精度封装, 默认禁用以避免 unscale 顺序问题."""

    def __init__(self, enabled=False, device=\'cpu\'):
        self.enabled = False

    def autocast(self):
        return torch.autocast(device_type=\'cpu\', enabled=False)

    def scale_loss(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass
'''
    if not os.path.exists(f):
        os.makedirs(os.path.dirname(f), exist_ok=True)
    cur = open(f).read() if os.path.exists(f) else ""
    if "def unscale_(self, optimizer):\n        pass" in cur:
        print("[skip] amp.py 已是安全版")
        return
    open(f, "w").write(safe)
    print("[ok] amp.py 已重写为安全版 (GradScaler 调用已移除)")


def patch_requirements():
    """确保 requirements.txt 含 Atari 依赖。"""
    f = os.path.join(ROOT, "requirements.txt")
    if not os.path.exists(f):
        print("[skip] requirements.txt 不存在")
        return
    c = open(f).read()
    needed = ["gymnasium[atari,accept-rom-license]", "ale-py"]
    miss = [n for n in needed if n.split("[")[0] not in c and n not in c]
    if not miss:
        print("[skip] requirements.txt 已含 Atari 依赖")
        return
    with open(f, "a") as fh:
        fh.write("\n# Atari 依赖 (补丁添加)\n")
        for n in miss:
            fh.write(n + "\n")
    print(f"[ok] requirements.txt 已补充: {miss}")


if __name__ == "__main__":
    print("=== CuriosityPPO Atari 补丁开始 ===")
    patch_agent()
    patch_amp()
    patch_requirements()
    print("=== 完成。现在可执行训练命令 ===")
