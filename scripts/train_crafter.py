"""快捷启动: Crafter 训练

用法:
    python scripts/train_crafter.py
    python scripts/train_crafter.py --total-steps 500000 --use-wandb
"""
import os
import subprocess
import sys

script_dir = os.path.dirname(__file__)
config = os.path.join(script_dir, "..", "experiments", "crafter_full.yaml")
subprocess.run(
    [sys.executable, os.path.join(script_dir, "train.py"), "--config", config]
    + sys.argv[1:]
)
