"""导出 Actor-Critic 策略网络为 ONNX 格式

加载检查点, 导出 actor 部分 (encoder + actor 头 → logits) 为 ONNX,
支持动态 batch 轴, 便于在 C++ / 移动端 / 浏览器中部署推理.

用法:
    python scripts/export_onnx.py \
        --checkpoint results/checkpoints/last.pt \
        --output results/onnx/minigrid_doorkey.onnx

    # 指定环境与配置 (默认按 --env 推断)
    python scripts/export_onnx.py --checkpoint last.pt --env crafter \
        --output crafter.onnx --config experiments/crafter_full.yaml

    # 自定义 opset 版本 / 关闭 ONNX Runtime 校验
    python scripts/export_onnx.py --checkpoint last.pt --env minigrid_doorkey \
        --output policy.onnx --opset 14 --no-verify
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# 将 src 加入模块搜索路径, 使脚本可独立运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from curiosity_ppo.config import load_config

# env 名称 → 默认实验配置文件
_ENV_CONFIG_MAP = {
    "crafter": "crafter_full.yaml",
    "atari": "atari_montezuma_full.yaml",
    "montezuma": "atari_montezuma_full.yaml",
    "minigrid": "minigrid_doorkey_full.yaml",
    "doorkey": "minigrid_doorkey_full.yaml",
}


def _infer_config_path(env_name):
    name = env_name.lower()
    for key, fname in _ENV_CONFIG_MAP.items():
        if key in name:
            return os.path.join(
                os.path.dirname(__file__), "..", "experiments", fname
            )
    raise ValueError(f"无法为环境 '{env_name}' 推断配置文件, 请用 --config 指定")


def make_single_env(env_name, seed):
    """创建 n_envs=1 的向量化环境 (用于获取观测/动作空间与样例观测)."""
    name = env_name.lower()
    if "crafter" in name:
        from curiosity_ppo.envs.crafter_env import make_crafter_env

        return make_crafter_env(n_envs=1, seed=seed)
    elif "atari" in name or "montezuma" in name:
        from curiosity_ppo.envs.atari_env import make_atari_env

        return make_atari_env(n_envs=1, seed=seed)
    elif "minigrid" in name or "doorkey" in name:
        from curiosity_ppo.envs.minigrid_env import make_minigrid_env

        return make_minigrid_env(n_envs=1, seed=seed)
    else:
        raise ValueError(f"Unknown env: {name}")


class PolicyNet(nn.Module):
    """策略网络包装器: encoder + actor 头, 仅输出动作 logits.

    用于部署时推理 (无需价值头), 与训练时的 ActorCritic 共享权重.
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.encoder = actor_critic.encoder
        self.actor = actor_critic.actor

    def forward(self, obs):
        h = self.encoder(obs)
        return self.actor(h)


def main():
    parser = argparse.ArgumentParser(description="导出策略网络为 ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--output", type=str, required=True, help="ONNX 输出路径")
    parser.add_argument("--env", type=str, default=None,
                        help="环境名 (用于推断配置与样例观测; 若指定 --config 可省略)")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件 (默认按 --env 推断)")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset 版本")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 ONNX Runtime 一致性校验")
    parser.add_argument("--seed", type=int, default=42, help="环境随机种子")
    args = parser.parse_args()

    # 确定配置: 优先 --config, 其次按 --env 推断
    if args.config:
        config_path = args.config
        env_name = args.env or load_config(args.config).env.name
    elif args.env:
        config_path = _infer_config_path(args.env)
        env_name = args.env
    else:
        parser.error("必须指定 --env 或 --config 之一")

    config = load_config(config_path)
    config.env.name = env_name
    config.env.n_envs = 1
    # 导出只需 Actor-Critic, 关闭好奇心模块以节省资源
    # (encoder 仍使用 config.icm.feature_dim, 与训练权重匹配)
    config.icm.enabled = False
    config.rnd.enabled = False
    config.episodic.enabled = False
    config.use_amp = False

    from curiosity_ppo.utils.seed import set_seed
    from curiosity_ppo.ppo.agent import CuriosityPPOAgent

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vec_env = make_single_env(env_name, seed=args.seed)
    agent = CuriosityPPOAgent(vec_env, config, device=device)
    agent.load(args.checkpoint)
    agent.actor_critic.eval()

    # 构造样例观测: 用 agent._to_tensor 保证与训练时输入 actor_critic 的张量布局一致
    sample_obs = vec_env.reset()
    dummy_input = agent._to_tensor(sample_obs)
    n_actions = agent.n_actions
    print(
        f"已加载检查点: {args.checkpoint} (step={agent.global_step})\n"
        f"  环境={env_name}  输入形状={tuple(dummy_input.shape)}  "
        f"动作数={n_actions}  设备={device}"
    )

    policy_net = PolicyNet(agent.actor_critic).to(device).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dynamic_axes = {"obs": {0: "batch"}, "logits": {0: "batch"}}
    export_kwargs = dict(
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    # PyTorch 2.x 默认使用基于 dynamo 的新导出器 (需 onnxscript);
    # 项目依赖未声明 onnxscript, 因此优先尝试新导出器, 失败时回退到
    # 经典 TorchScript 导出器 (dynamo=False), 保证开箱即用.
    with torch.no_grad():
        try:
            torch.onnx.export(policy_net, dummy_input, args.output, **export_kwargs)
            exporter = "dynamo"
        except Exception as e:
            print(f"新导出器失败 ({e}), 回退到经典导出器 (dynamo=False)...")
            torch.onnx.export(
                policy_net, dummy_input, args.output, dynamo=False, **export_kwargs
            )
            exporter = "legacy"
    print(f"已导出 ONNX: {args.output} (exporter={exporter})")

    # ---------- 可选: ONNX Runtime 一致性校验 ----------
    if not args.no_verify:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name

            # 用不同 batch 大小验证动态轴
            test_input = dummy_input.cpu().numpy()
            with torch.no_grad():
                ref = policy_net(dummy_input).cpu().numpy()

            ort_out = sess.run(None, {input_name: test_input})[0]
            max_diff = float(np.max(np.abs(ref - ort_out)))
            print(
                f"ONNX Runtime 校验通过: logits 形状={ort_out.shape}, "
                f"与 PyTorch 最大绝对误差={max_diff:.2e}"
            )

            # 动态 batch 测试
            batch2 = np.concatenate([test_input, test_input], axis=0)
            ort_out2 = sess.run(None, {input_name: batch2})[0]
            print(f"动态 batch 校验: batch={ort_out2.shape[0]} (输入 batch={batch2.shape[0]})")
        except ImportError:
            print("未安装 onnxruntime, 跳过一致性校验")
        except Exception as e:
            print(f"ONNX Runtime 校验失败 (不影响导出): {e}")

    vec_env.close()


if __name__ == "__main__":
    main()
