import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_specialized.training.env_wrappers import OpponentPool, LiarDiceSelfPlayEnv
from rl_specialized.training.masked_policy import MaskedActorCriticPolicy
from rl_specialized.networks.policy_network import make_default_policy_kwargs, PolicyNetConfig


def auto_select_device() -> str:
    if th.cuda.is_available():
        return "cuda"
    try:
        if hasattr(th.backends, "mps") and th.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class SelfPlayConfig:
    num_players: int = 2
    dice_per_player: int = 5
    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    policy_hidden_size: int = 128
    n_steps: int = 1024
    batch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    seed: Optional[int] = 42
    device: Optional[str] = None
    tensorboard_log: Optional[str] = "runs/rl_selfplay"
    eval_episodes: int = 10
    snapshot_freq: int = 200_000
    rule_ratio_start: float = 0.8
    rule_ratio_end: float = 0.2


class SelfPlayCallback(BaseCallback):
    """按进度更新对手池占比，并定期加入策略快照"""

    def __init__(self, pool: OpponentPool, cfg: SelfPlayConfig, save_dir: str):
        super().__init__()
        self.pool = pool
        self.cfg = cfg
        self.save_dir = save_dir
        self._next_snapshot = cfg.snapshot_freq

    def _on_step(self) -> bool:
        # 更新规则体占比：线性从 start 到 end
        progress = min(1.0, self.num_timesteps / max(1, self.cfg.total_timesteps))
        ratio = (1 - progress) * self.cfg.rule_ratio_start + progress * self.cfg.rule_ratio_end
        self.pool.set_rule_ratio(ratio)

        # 到达快照点：保存并注册为对手
        if self.num_timesteps >= self._next_snapshot:
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"snapshot_step_{self.num_timesteps}.zip")
            self.model.save(path)
            # 策略对手推断放CPU
            self.pool.add_policy(path, device="cpu")
            self._next_snapshot += self.cfg.snapshot_freq
        return True


class SelfPlayTrainer:
    def __init__(self, cfg: SelfPlayConfig):
        self.cfg = cfg
        if self.cfg.device is None:
            self.cfg.device = auto_select_device()

        # 构建对手池：初始化若干规则变体（基础 + 概率型）
        self.pool = OpponentPool(num_players=cfg.num_players, rule_ratio=cfg.rule_ratio_start)
        # 基础规则对手：挑战阈值偏移 2..5，起手面值 3/4/5
        for off in [2, 3, 4, 5]:
            for face in [3, 4, 5]:
                self.pool.add_rule(start_face=face, challenge_offset=off)
        # 概率型规则对手：不同阈值组合，提升对手多样性
        for tc in [0.20, 0.25, 0.30]:
            for tr in [0.55, 0.60, 0.65]:
                self.pool.add_prob_rule(theta_challenge=tc, target_raise=tr, max_extra_raise=2)

        def make_env():
            # 默认开启潜在塑形：β=0.05，γ同 PPO 配置（cfg.gamma）
            return LiarDiceSelfPlayEnv(
                pool=self.pool,
                dice_per_player=cfg.dice_per_player,
                dense_shaping=True,
                shaping_beta=0.05,
                shaping_gamma=cfg.gamma,
            )

        self.train_env = DummyVecEnv([make_env])
        self.eval_env = DummyVecEnv([make_env])

        policy_kwargs = make_default_policy_kwargs(PolicyNetConfig(
            features_dim=cfg.policy_hidden_size,
            pi_hidden=cfg.policy_hidden_size,
            vf_hidden=cfg.policy_hidden_size,
        ))

        self.model = PPO(
            policy=MaskedActorCriticPolicy,
            env=self.train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            tensorboard_log=cfg.tensorboard_log,
            policy_kwargs=policy_kwargs,
            seed=cfg.seed,
            device=cfg.device,
        )

    def train(self):
        save_dir = os.path.join(self.cfg.tensorboard_log or "runs", "snapshots")
        sp_cb = SelfPlayCallback(pool=self.pool, cfg=self.cfg, save_dir=save_dir)
        eval_cb = EvalCallback(self.eval_env, best_model_save_path=os.path.join(self.cfg.tensorboard_log or "runs", "best_model"),
                               log_path=self.cfg.tensorboard_log, eval_freq=max(10000, self.cfg.n_steps),
                               deterministic=False, render=False, n_eval_episodes=self.cfg.eval_episodes)
        self.model.learn(total_timesteps=self.cfg.total_timesteps, callback=[sp_cb, eval_cb])

    def evaluate(self, n_episodes: Optional[int] = None) -> float:
        if n_episodes is None:
            n_episodes = self.cfg.eval_episodes
        env = self.eval_env
        total_reward = 0.0
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                done = bool(done[0])
                total_reward += float(reward[0])
        return total_reward / n_episodes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--dice_per_player", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--snapshot_freq", type=int, default=200_000)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu", None])
    parser.add_argument("--logdir", type=str, default="runs/rl_selfplay")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SelfPlayConfig(num_players=args.num_players, dice_per_player=args.dice_per_player,
                         total_timesteps=args.timesteps, snapshot_freq=args.snapshot_freq,
                         device=args.device, tensorboard_log=args.logdir, seed=args.seed)

    trainer = SelfPlayTrainer(cfg)
    print(f"Using device: {cfg.device}")
    trainer.train()
    avg_reward = trainer.evaluate()
    print(f"Eval avg reward over {cfg.eval_episodes} eps: {avg_reward:.3f}")


if __name__ == "__main__":
    main()
