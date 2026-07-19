"""从 checkpoint 张量形状反推架构并评测 (不依赖 ckpt 内置 config)。

关键修复 (2026-07-19):
  - DSW 训练时 agent.save 未保存 policy_obs_rms, 导致评测时归一化统计为初始值
    (mean=0, var=1) -> 网络吃到未归一化的原始 uint8 像素, 输入比训练时大 ~50-80 倍,
    卷积层饱和 -> 前向输出退化 -> 每局暴毙得 0 分 (假阴性)。
  - 本脚本在正式评测前, 先用评测环境随机采集观测在线校准 policy_obs_rms,
    逼近训练时的归一化统计量 (输入分布对齐), 再评测。
  - 若未来 checkpoint 已保存 policy_obs_rms, 则直接恢复, 跳过校准。

用法:
  python scripts/eval_ckpt.py --checkpoint <ckpt.pt> --config experiments/atari_montezuma_full.yaml \
      --n-episodes 10 --device cuda
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from curiosity_ppo.config import Config, load_config  # noqa: E402
from curiosity_ppo.ppo.agent import CuriosityPPOAgent  # noqa: E402
from curiosity_ppo.utils.checkpoint import load_checkpoint  # noqa: E402
from benchmarks.eval_atari import evaluate_atari  # noqa: E402
from benchmarks.report import generate_report  # noqa: E402


def infer_architecture(ckpt):
    """从 actor_critic state_dict 反推 in_channels 与 n_actions。"""
    st = ckpt.get("agent_state", ckpt)
    ac = st["actor_critic"]
    conv0 = ac["encoder.conv.0.weight"]            # [out, in, k, k]
    in_ch = conv0.shape[1]
    n_act = ac["actor.2.weight"].shape[0]          # [n_actions, hidden]
    return in_ch, n_act


def calibrate_rms(agent, env, n_steps=3000):
    """用评测环境随机采集观测, 在线更新 policy_obs_rms, 逼近训练时统计量。

    仅在 checkpoint 未保存 policy_obs_rms 时调用 (当前 DSW ckpt 即如此)。
    env 输出 uint8 (84,84,4) HWC, 与训练时 rms.update 的输入一致。
    """
    print(f"[eval_ckpt] 校准 policy_obs_rms ({n_steps} 步, 随机动作覆盖) ...")
    obs = env.reset()
    for _ in range(n_steps):
        action = np.array([env.action_space.sample()])  # 随机动作 -> 广覆盖
        obs, reward, done, info = env.step(action)
        agent.policy_obs_rms.update(np.asarray(obs, dtype=np.float32))
        if bool(done[0]):
            obs = env.reset()
    print(
        f"[eval_ckpt] 校准完成: rms.mean≈{agent.policy_obs_rms.mean.mean():.1f}, "
        f"std≈{agent.policy_obs_rms.std.mean():.1f}, count={agent.policy_obs_rms.count}"
    )


def main():
    parser = argparse.ArgumentParser(description="从 checkpoint 张量反推架构并评测")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点 (.pt)")
    parser.add_argument("--config", type=str, default="experiments/atari_montezuma_full.yaml",
                        help="与训练一致的 yaml")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--calib-steps", type=int, default=3000,
                        help="校准 policy_obs_rms 的步数 (ckpt 无 rms 时生效)")
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cuda"

    # ① 反推架构
    ckpt = load_checkpoint(args.checkpoint, device="cpu")
    in_ch, n_act = infer_architecture(ckpt)
    print(f"[eval_ckpt] 从 checkpoint 推断: in_channels={in_ch}, n_actions={n_act}")

    # ② 选定环境名
    if in_ch == 4:
        env_name = "atari_montezuma"
        expected_encoder = "NatureDQNEncoder"
    elif in_ch == 3:
        env_name = "crafter"
        expected_encoder = "CrafterEncoder"
    else:
        raise SystemExit(f"[eval_ckpt] ERROR: 未知输入通道数 in_channels={in_ch}")

    # ③ 加载 yaml (强制 env.name 与训练一致)
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        cfg = Config()
    cfg.env.name = env_name
    cfg.env.n_envs = 1
    cfg.seed = args.seed

    # ④ 构建环境 + Agent
    from curiosity_ppo.envs.atari_env import make_atari_env  # noqa: E402
    env = make_atari_env(n_envs=1, seed=args.seed)

    agent = CuriosityPPOAgent(env, cfg, device=device)
    print(f"[eval_ckpt] 构造架构: encoder={type(agent.actor_critic.encoder).__name__}, "
          f"in_channels={agent.in_channels}, n_actions={agent.n_actions}")
    if type(agent.actor_critic.encoder).__name__ != expected_encoder:
        raise SystemExit(
            f"[eval_ckpt] ERROR: 构造出的编码器 {type(agent.actor_critic.encoder).__name__} "
            f"与 checkpoint 推断的 {expected_encoder} 不符!"
        )

    # ⑤ 形状校验
    agent_sd = agent.actor_critic.state_dict()
    ckpt_ac = ckpt.get("agent_state", ckpt)["actor_critic"]
    for key in ("encoder.conv.0.weight", "actor.2.weight"):
        a = tuple(agent_sd[key].shape)
        b = tuple(ckpt_ac[key].shape)
        if a != b:
            raise SystemExit(
                f"[eval_ckpt] ERROR: 形状不匹配 key={key} | 构造={a} vs checkpoint={b}"
            )
    print("[eval_ckpt] 形状校验通过 ✅")

    # ⑥ 加载权重
    agent.load(args.checkpoint)
    agent.actor_critic.eval()
    print(f"[eval_ckpt] 已加载权重: global_step={agent.global_step}")

    # ⑦ 恢复或校准 policy_obs_rms (v3 修复: 不依赖 count 值判断)
    st = ckpt.get("agent_state", ckpt)
    has_saved_rms = "policy_obs_rms" in st
    rms_info = f"type={type(agent.policy_obs_rms).__name__}, "
    rms_info += f"isNone={agent.policy_obs_rms is None}"
    if hasattr(agent.policy_obs_rms, 'count'):
        rms_info += f", count={agent.policy_obs_rms.count}"
    print(f"[eval_ckpt] rms 信息: ckpt含rms={has_saved_rms}, 当前={rms_info}")

    if has_saved_rms:
        agent.policy_obs_rms = st["policy_obs_rms"]
        print(f"[eval_ckpt] 已从 checkpoint 恢复 policy_obs_rms")
    elif agent.policy_obs_rms is not None:
        # v3: 只要 ckpt 没存 rms 且当前有 rms 对象, 就强制校准 (不再判断 count==0)
        calib_env = make_atari_env(n_envs=1, seed=args.seed + 999)
        try:
            calibrate_rms(agent, calib_env, n_steps=args.calib_steps)
        finally:
            try:
                calib_env.close()
            except Exception:
                pass
    else:
        print("[eval_ckpt] 警告: policy_obs_rms 为 None 且未保存, 跳过归一化")

    # ⑧ 正式评测
    results = evaluate_atari(agent, n_episodes=args.n_episodes, seed=args.seed)

    generate_report({"atari": results}, output_dir=args.output_dir)
    print(f"[eval_ckpt] 报告: {os.path.join(args.output_dir, 'benchmark_report.md')}")
    print(
        f"[eval_ckpt] 结果: mean={results['mean_score']:.1f} "
        f"max={results['max_score']:.1f} min={results['min_score']:.1f} "
        f"baseline={results['baseline']}"
    )


if __name__ == "__main__":
    main()
