"""快捷启动: MiniGrid DoorKey 训练

用法:
    python scripts/train_minigrid.py
    python scripts/train_minigrid.py --total-steps 750000 --use-wandb
"""
import os
import subprocess
import sys

script_dir = os.path.dirname(__file__)
config = os.path.join(script_dir, "..", "experiments", "minigrid_doorkey_full.yaml")
subprocess.run(
    [sys.executable, os.path.join(script_dir, "train.py"), "--config", config]
    + sys.argv[1:]
)
