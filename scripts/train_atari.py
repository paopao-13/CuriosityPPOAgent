"""快捷启动: Atari Montezuma's Revenge 训练

用法:
    python scripts/train_atari.py
    python scripts/train_atari.py --total-steps 5000000 --use-wandb
"""
import os
import subprocess
import sys

script_dir = os.path.dirname(__file__)
config = os.path.join(script_dir, "..", "experiments", "atari_montezuma_full.yaml")
subprocess.run(
    [sys.executable, os.path.join(script_dir, "train.py"), "--config", config]
    + sys.argv[1:]
)
