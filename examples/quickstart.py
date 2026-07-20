#!/usr/bin/env python
"""Minimal reproduction script – CuriosityPPOAgent quickstart.

Demonstrates how to:
  1. Load an experiment config (YAML)
  2. Build a vectorized MiniGrid environment
  3. Initialise the full CuriosityPPOAgent (policy + curiosity modules)
  4. Run greedy evaluation over N episodes
  5. Report success rate and mean reward

This is a **smoke-test** that validates the pipeline is functional
end-to-end without needing millions of training steps.

Usage:
    python examples/quickstart.py [--n-episodes 20] [--env-id MiniGrid-DoorKey-8x8-v0]

Prerequisites:
    pip install -r requirements.txt
"""

import argparse
import os
import sys

# Ensure project root is on sys.path so src.curiosity_ppo is importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="CuriosityPPOAgent minimal smoke test")
    parser.add_argument(
        "--n-episodes", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--env-id",
        default="MiniGrid-DoorKey-8x8-v0",
        help="Gymnasium environment ID",
    )
    args = parser.parse_args()

    # --- 1. Load config -------------------------------------------------
    from src.curiosity_ppo.config import load_config, Config

    config_path = os.path.join(ROOT, "experiments", "minigrid_potential_shaping.yaml")
    cfg = load_config(config_path) if os.path.exists(config_path) else None

    # --- 2. Build environment -------------------------------------------
    from src.curiosity_ppo.envs.minigrid_env import make_minigrid_env

    n_envs = min(cfg.n_envs, 4) if cfg else 4
    vec_env = make_minigrid_env(
        env_id=args.env_id,
        n_envs=n_envs,
        seed=(cfg.seed if cfg else 42),
        reward_shaping=True,
    )

    # --- 3. Initialise agent --------------------------------------------
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.curiosity_ppo.ppo.agent import CuriosityPPOAgent

    agent = CuriosityPPOAgent(vec_env, config=cfg or Config(), device=device)

    print(f"[ok] Pipeline ready: {args.env_id} | {n_envs} envs | {device}")
    print(f"     Parameters: {sum(p.numel() for p in agent.actor_critic.parameters()):,}")

    # --- 4. Greedy evaluation (built-in) --------------------------------
    success_rate, mean_reward = agent.evaluate(n_episodes=args.n_episodes)

    print(f"\n{'='*50}")
    print(f"Evaluation complete ({args.n_episodes} episodes)")
    print(f"  success_rate : {success_rate:.2%}")
    print(f"  avg_reward   : {mean_reward:.4f}")
    print(f"{'='*50}")

    vec_env.close()


if __name__ == "__main__":
    main()
