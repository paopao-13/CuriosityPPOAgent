"""录制智能体演示视频

创建单个环境 (非向量化, 通过 n_envs=1 的 DummyVecEnv), 加载模型, 运行 N 步,
逐帧渲染并用 imageio 保存为视频. 支持 crafter / atari_montezuma / minigrid_doorkey 三种环境.

用法:
    python scripts/record_video.py \
        --checkpoint results/checkpoints/last.pt \
        --env crafter \
        --output results/videos/demo.mp4

    # 指定步数 / fps / 随机策略
    python scripts/record_video.py --checkpoint last.pt --env minigrid_doorkey \
        --output demo.mp4 --steps 500 --fps 20 --stochastic

    # 使用自定义配置 (默认按 --env 自动推断 experiments/*.yaml)
    python scripts/record_video.py --checkpoint last.pt --env atari_montezuma \
        --config experiments/atari_montezuma_full.yaml --output demo.mp4
"""
import argparse
import os
import sys

import numpy as np

# 将 src 加入模块搜索路径, 使脚本可独立运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from curiosity_ppo.config import load_config

try:
    import cv2  # opencv-python-headless
    _HAS_CV2 = True
except ImportError:  # 回退到 numpy 最近邻缩放
    cv2 = None
    _HAS_CV2 = False

# imageio 延迟导入, 保证 --help 在未安装依赖时仍可用
imageio = None

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
    """创建 n_envs=1 的向量化环境 (保证与 Agent 接口一致)."""
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


# --------------------------------------------------------------------------
# 观测 → 可显示 RGB 帧
# --------------------------------------------------------------------------


def _to_uint8(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        mx = float(img.max()) if img.size else 0.0
        if mx <= 1.0:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resize(img, size):
    """等比例最近邻缩放, 使短边 >= size (像素艺术风格)."""
    h, w = img.shape[:2]
    if min(h, w) >= size:
        return img
    scale = size / float(min(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if _HAS_CV2:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # numpy 最近邻回退
    row_idx = (np.arange(new_h) * h // new_h).clip(0, h - 1)
    col_idx = (np.arange(new_w) * w // new_w).clip(0, w - 1)
    return img[row_idx][:, col_idx]


def obs_to_rgb(obs):
    """将单条观测转为 (H, W, 3) uint8 RGB 帧.

    支持以下观测布局:
      - (H, W, 3)  RGB 图像        (Crafter / MiniGrid)
      - (H, W, 1)  单通道图像
      - (C, H, W)  堆叠灰度图      (Atari FrameStack), 取最后一帧
      - (H, W)     灰度图
    """
    obs = np.asarray(obs)
    if obs.ndim == 4:  # 去掉 batch 维
        obs = obs[0]

    if obs.ndim == 3:
        h, w, c = obs.shape
        if c in (1, 3):
            img = obs
        else:
            # (C, H, W) → 取最后一帧作为代表帧
            img = obs[-1][..., None]  # (H, W, 1)
    elif obs.ndim == 2:
        img = obs[..., None]  # (H, W, 1)
    else:
        raise ValueError(f"不支持的观测形状: {obs.shape}")

    img = _to_uint8(img)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return np.ascontiguousarray(img)


def grab_frame(obs, min_size):
    """从观测获取一帧并放大到至少 min_size 像素."""
    frame = obs_to_rgb(obs)
    if min_size and min_size > 0:
        frame = _resize(frame, min_size)
    return np.ascontiguousarray(frame)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="录制智能体演示视频")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--env", type=str, required=True,
                        help="环境名: crafter / atari_montezuma / minigrid_doorkey")
    parser.add_argument("--output", type=str, default="results/videos/demo.mp4",
                        help="输出视频路径")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件 (默认按 --env 推断)")
    parser.add_argument("--steps", type=int, default=300, help="录制步数")
    parser.add_argument("--fps", type=int, default=20, help="视频帧率")
    parser.add_argument("--stochastic", action="store_true", help="使用随机策略而非确定性策略")
    parser.add_argument("--min-size", type=int, default=320,
                        help="帧最小短边像素 (放大像素艺术, 0 表示不放大)")
    parser.add_argument("--seed", type=int, default=42, help="环境随机种子")
    args = parser.parse_args()

    # 加载配置 (与训练一致, 以保证编码器结构与权重匹配)
    config_path = args.config or _infer_config_path(args.env)
    config = load_config(config_path)
    config.env.name = args.env
    config.env.n_envs = 1
    # 录制只需 Actor-Critic, 关闭好奇心模块以节省资源
    # (encoder 仍使用 config.icm.feature_dim, 与训练权重匹配)
    config.icm.enabled = False
    config.rnd.enabled = False
    config.episodic.enabled = False
    config.use_amp = False

    import torch
    from curiosity_ppo.utils.seed import set_seed
    from curiosity_ppo.ppo.agent import CuriosityPPOAgent

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vec_env = make_single_env(args.env, seed=args.seed)
    agent = CuriosityPPOAgent(vec_env, config, device=device)
    agent.load(args.checkpoint)
    agent.actor_critic.eval()
    print(f"已加载检查点: {args.checkpoint} (step={agent.global_step})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    frames = []
    obs = vec_env.reset()

    print(f"开始录制 {args.steps} 步, 输出: {args.output}")
    for step in range(args.steps):
        frame = grab_frame(obs, args.min_size)
        frames.append(frame)

        with torch.no_grad():
            action = agent.act(obs, deterministic=not args.stochastic)

        obs, reward, done, info = vec_env.step(action)

        if step % max(1, args.steps // 10) == 0:
            print(f"  step {step}/{args.steps}  done={bool(done[0])}")

    # 追加最后一帧
    frames.append(grab_frame(obs, args.min_size))

    global imageio
    try:
        import imageio as _imageio
        imageio = _imageio
    except ImportError:
        raise ImportError(
            "保存视频需要 imageio (及 imageio-ffmpeg), 请安装: "
            "pip install imageio imageio-ffmpeg"
        )

    imageio.mimsave(args.output, frames, fps=args.fps, macro_block_size=1)
    print(f"已保存视频 ({len(frames)} 帧, {args.fps} fps): {args.output}")
    vec_env.close()


if __name__ == "__main__":
    main()
