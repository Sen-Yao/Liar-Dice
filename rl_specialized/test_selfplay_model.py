"""Utility for benchmarking a trained PPO self-play agent.

This script mirrors the spirit of :mod:`model_playground` but targets the
specialized self-play setup living under ``rl_specialized``.  It loads a
checkpoint produced by :mod:`rl_specialized.training.train_selfplay`, recreates
the default opponent pool, and runs a batch of evaluation episodes.  Outputs
include per-episode rewards and a compact summary so you can sanity-check a
trained policy quickly.

Examples
--------

Evaluate the best model from the latest run for 25 episodes::

    python -m rl_specialized.test_selfplay_model --episodes 25

Point the script at a custom checkpoint and reuse stored VecNormalize stats::

    python -m rl_specialized.test_selfplay_model \
        --model-path runs/rl_selfplay/snapshots/snapshot_step_200000.zip \
        --norm-path runs/rl_selfplay/vecnormalize.pkl

By default the evaluation disables the training-time reward shaping so that the
reported rewards map directly to round outcomes (+1 win / -1 loss).
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
import sys
import types
from typing import List, Tuple

import numpy as np


def _ensure_project_root_on_path() -> None:
    """Allow running the file directly by adding the repo root to ``sys.path``."""

    project_root = Path(__file__).resolve().parents[1]
    project_str = str(project_root)
    if project_str not in sys.path:
        # Prepend so top-level packages (e.g. ``agents``) shadow similarly named
        # site-packages modules when running the script directly.
        sys.path.insert(0, project_str)

    agents_dir = project_root / "agents"
    if agents_dir.is_dir():
        pkg_name = "agents"
        existing = sys.modules.get(pkg_name)

        # Replace or augment any third-party ``agents`` package so local modules win.
        if existing is None or not getattr(existing, "__file__", "").startswith(project_str):
            namespace = types.ModuleType(pkg_name)
            namespace.__path__ = [str(agents_dir)]  # type: ignore[attr-defined]
            namespace.__file__ = str(agents_dir / "__init__.py")
            namespace.__package__ = pkg_name
            sys.modules[pkg_name] = namespace
        else:
            namespace_path = getattr(existing, "__path__", None)
            if namespace_path is not None and str(agents_dir) not in namespace_path:
                try:
                    namespace_path.append(str(agents_dir))
                except AttributeError:
                    existing.__path__ = list(namespace_path) + [str(agents_dir)]  # type: ignore[attr-defined]


_ensure_project_root_on_path()

try:  # Torch is required indirectly by Stable-Baselines3
    import torch
except Exception:  # pragma: no cover - torch is an explicit dependency for PPO
    torch = None

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_specialized.training.train_selfplay import SelfPlayConfig, auto_select_device
from rl_specialized.training.env_wrappers import OpponentPool, LiarDiceSelfPlayEnv


def build_opponent_pool(
    cfg: SelfPlayConfig,
    *,
    snapshots: List[Path],
    load_snapshots: bool,
    device: str,
) -> OpponentPool:
    """Recreate the opponent pool, optionally including policy snapshots."""

    pool = OpponentPool(num_players=cfg.num_players, rule_ratio=cfg.rule_ratio_start)

    for offset in [2, 3, 4, 5]:
        for face in [3, 4, 5]:
            pool.add_rule(start_face=face, challenge_offset=offset)

    for theta in [0.20, 0.25, 0.30]:
        for target in [0.55, 0.60, 0.65]:
            pool.add_prob_rule(theta_challenge=theta, target_raise=target, max_extra_raise=2)

    if load_snapshots:
        for snap in snapshots:
            pool.add_policy(str(snap), device=device)

    return pool


def collect_snapshots(directory: Path) -> List[Path]:
    """Return sorted snapshot paths under ``directory``."""

    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.zip") if path.is_file())


def prompt_snapshot_mode() -> bool:
    """询问用户是否加载策略快照，返回 True 表示加载。"""

    message = (
        "选择对手模式：\n"
        "  1) 仅规则对手\n"
        "  2) 规则 + 策略快照 (默认)\n"
        "请输入选项编号后回车："
    )

    try:
        choice = input(message).strip()
    except EOFError:
        choice = ""

    normalized = choice.lower()
    if normalized in {"1", "rule", "rules", "only_rules", "only"}:
        return False
    if normalized in {"2", "mixed", "policy", "snapshots"}:
        return True

    # 默认行为：加载策略快照
    return True


def make_eval_env(
    pool: OpponentPool,
    cfg: SelfPlayConfig,
    *,
    dense_shaping: bool,
    early_game_penalty: bool,
    norm_path: Path | None,
) -> Tuple[VecNormalize, bool]:
    """Create the vectorized evaluation environment."""

    def _make_env() -> LiarDiceSelfPlayEnv:
        return LiarDiceSelfPlayEnv(
            pool=pool,
            dice_per_player=cfg.dice_per_player,
            dense_shaping=dense_shaping,
            shaping_beta=0.05 if dense_shaping else 0.0,
            shaping_gamma=cfg.gamma,
            early_challenge_min_raises=cfg.early_challenge_min_raises if early_game_penalty else 0,
            early_challenge_penalty=cfg.early_challenge_penalty if early_game_penalty else 0.0,
            guess_step_bonus=cfg.guess_step_bonus if early_game_penalty else 0.0,
            show_opponent_info=False,
        )

    dummy_env = DummyVecEnv([_make_env])

    if norm_path is not None and norm_path.is_file():
        vec_env = VecNormalize.load(norm_path, dummy_env)
        restored = True
    else:
        vec_env = VecNormalize(
            dummy_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            norm_obs_keys=["obs"],
        )
        restored = False

    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env, restored


def run_evaluation(
    model: PPO,
    env: VecNormalize,
    episodes: int,
    deterministic: bool,
) -> List[Tuple[float, int]]:
    """Roll out ``episodes`` episodes and collect (reward, length)."""

    metrics: List[Tuple[float, int]] = []

    for _ in range(episodes):
        obs = env.reset()
        done = np.array([False])
        ep_reward = 0.0
        ep_len = 0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            ep_reward += float(reward[0])
            ep_len += 1

        metrics.append((ep_reward, ep_len))

    return metrics


def format_summary(metrics: List[Tuple[float, int]]) -> str:
    """Pretty-print a quick aggregate summary."""

    rewards = [m[0] for m in metrics]
    lengths = [m[1] for m in metrics]

    avg_reward = statistics.mean(rewards) if rewards else 0.0
    win_rate = (sum(1 for r in rewards if r > 0.0) / len(rewards)) if rewards else 0.0
    avg_len = statistics.mean(lengths) if lengths else 0.0

    return (
        f"Win rate: {win_rate * 100:.1f}%\n"
        f"Average reward: {avg_reward:.2f}\n"
        f"Average steps: {avg_len:.1f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained self-play PPO model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("runs/rl_selfplay/best_model/best_model.zip"),
        help="Checkpoint produced by SelfPlayTrainer (default: best model).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of seats at the table (matches training configuration).",
    )
    parser.add_argument(
        "--dice-per-player",
        type=int,
        default=5,
        help="Dice per player in the environment.",
    )
    parser.add_argument(
        "--rule-ratio",
        type=float,
        default=0.5,
        help="Probability of sampling rule-based opponents from the pool.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (greedy) instead of sampling actions.",
    )
    parser.add_argument(
        "--norm-path",
        type=Path,
        default=None,
        help="Optional VecNormalize statistics produced during training.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("runs/rl_selfplay/snapshots"),
        help="Directory containing snapshot .zip files for opponent sampling.",
    )
    parser.add_argument(
        "--keep-shaping",
        action="store_true",
        help="Retain training-time reward shaping during evaluation.",
    )
    parser.add_argument(
        "--keep-early-penalty",
        action="store_true",
        help="Keep the early-challenge penalty used during training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    cfg = SelfPlayConfig(
        num_players=args.num_players,
        dice_per_player=args.dice_per_player,
    )
    cfg.rule_ratio_start = float(np.clip(args.rule_ratio, 0.0, 1.0))
    cfg.rule_ratio_end = cfg.rule_ratio_start

    load_snapshots = prompt_snapshot_mode()
    if not load_snapshots:
        cfg.rule_ratio_start = cfg.rule_ratio_end = 1.0

    device = auto_select_device()

    snapshots: List[Path] = []
    if load_snapshots:
        snapshots = collect_snapshots(args.snapshot_dir)
        if snapshots:
            print(f"Loaded {len(snapshots)} snapshot opponent(s) from {args.snapshot_dir}.")
        else:
            print("未找到策略快照，对手将仅使用规则对手。")
            load_snapshots = False
            cfg.rule_ratio_start = cfg.rule_ratio_end = 1.0

    pool = build_opponent_pool(
        cfg,
        snapshots=snapshots,
        load_snapshots=load_snapshots,
        device=device,
    )

    env, restored = make_eval_env(
        pool,
        cfg,
        dense_shaping=args.keep_shaping,
        early_game_penalty=args.keep_early_penalty,
        norm_path=args.norm_path,
    )

    if restored:
        print(f"VecNormalize statistics restored from {args.norm_path}.")
    else:
        print("VecNormalize statistics not provided; using fresh statistics.")

    model = PPO.load(args.model_path, device=device)
    print(f"Model loaded from {args.model_path} (device={device}).")

    metrics = run_evaluation(model, env, max(args.episodes, 1), args.deterministic)

    for idx, (reward, length) in enumerate(metrics, start=1):
        print(f"Episode {idx:>3}: reward={reward:+.2f} steps={length}")

    print("\n--- Summary ---")
    print(format_summary(metrics))

    stats = pool.get_usage_stats()
    if load_snapshots:
        print(
            "Opponent usage: "
            f"rules={stats['basic_rule'] + stats['prob_rule']:.1f}% "
            f"snapshots={stats['policy']:.1f}%"
        )
    else:
        print(f"Opponent usage: rules={stats['basic_rule'] + stats['prob_rule']:.1f}%")


if __name__ == "__main__":
    main()
