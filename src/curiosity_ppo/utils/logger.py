"""训练日志工具 - 支持 Wandb 和控制台输出"""
import json
import os
import time
from collections import defaultdict


class TrainLogger:
    """统一的训练日志器，支持 Wandb 和控制台"""

    def __init__(self, project="curiosity-ppo", config=None, use_wandb=False, run_name=None):
        self.use_wandb = use_wandb
        self.wandb_run = None
        self.metrics_history = defaultdict(list)
        self.start_time = time.time()
        self.step = 0

        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=project,
                    config=config.__dict__ if hasattr(config, '__dict__') else dict(config) if config else {},
                    name=run_name,
                )
            except Exception as e:
                print(f"Warning: Wandb init failed: {e}, falling back to console")
                self.use_wandb = False

    def log(self, metrics, step=None):
        """记录指标"""
        if step is not None:
            self.step = step
        else:
            self.step += 1

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)

        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log(metrics, step=self.step)
        else:
            # 控制台输出
            elapsed = time.time() - self.start_time
            parts = [f"step={self.step}"]
            for key, value in metrics.items():
                if isinstance(value, float):
                    parts.append(f"{key}={value:.4f}")
                else:
                    parts.append(f"{key}={value}")
            parts.append(f"elapsed={elapsed:.0f}s")
            print(" | ".join(parts))

    def log_histogram(self, name, values, step=None):
        """记录直方图"""
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log({name: wandb.Histogram(values)}, step=step or self.step)

    def finish(self):
        """结束日志"""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

    def save_history(self, path):
        """保存指标历史到 JSON"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=2)
