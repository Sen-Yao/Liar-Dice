import os
from dataclasses import dataclass
from typing import Optional

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_specialized.training.env_wrappers import LiarDiceSingleAgentEnv
from rl_specialized.training.masked_policy import MaskedActorCriticPolicy
from rl_specialized.networks.policy_network import make_default_policy_kwargs, PolicyNetConfig


def auto_select_device() -> str:
    """自动选择可用设备：cuda > mps > cpu"""
    if th.cuda.is_available():
        return "cuda"
    # macOS Metal
    try:
        if hasattr(th.backends, "mps") and th.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class TrainConfig:
    num_players: int = 2
    dice_per_player: int = 5
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    policy_hidden_size: int = 64
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    seed: Optional[int] = 42
    device: Optional[str] = None
    tensorboard_log: Optional[str] = "runs/rl_specialized"
    eval_episodes: int = 10


class SpecializedTrainer:
    """基于SB3-PPO的专用模型训练器（带动作掩码）"""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if self.cfg.device is None:
            self.cfg.device = auto_select_device()

        def make_env():
            return LiarDiceSingleAgentEnv(num_players=cfg.num_players, dice_per_player=cfg.dice_per_player)

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
        eval_callback = EvalCallback(self.eval_env, best_model_save_path=os.path.join(self.cfg.tensorboard_log or "runs", "best_model"),
                                     log_path=self.cfg.tensorboard_log, eval_freq=max(10000, self.cfg.n_steps),
                                     deterministic=False, render=False, n_eval_episodes=self.cfg.eval_episodes)
        self.model.learn(total_timesteps=self.cfg.total_timesteps, callback=eval_callback)

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
                # VecEnv：done 为数组
                done = bool(done[0])
                total_reward += float(reward[0])
        return total_reward / n_episodes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--dice_per_player", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu", None])
    parser.add_argument("--logdir", type=str, default="runs/rl_specialized")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(num_players=args.num_players, dice_per_player=args.dice_per_player,
                      total_timesteps=args.timesteps, device=args.device, tensorboard_log=args.logdir,
                      seed=args.seed)

    trainer = SpecializedTrainer(cfg)
    print(f"Using device: {cfg.device}")
    trainer.train()
    avg_reward = trainer.evaluate()
    print(f"Eval avg reward over {cfg.eval_episodes} eps: {avg_reward:.3f}")


if __name__ == "__main__":
    main()
