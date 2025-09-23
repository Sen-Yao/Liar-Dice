import os
import sys
from dataclasses import dataclass
from typing import Optional, Callable

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    total_timesteps: int = 50_000  # 可通过命令行覆盖
    # 学习率与更新规模优化
    learning_rate: float = 3e-4
    learning_rate_end: float = 5e-5
    policy_hidden_size: int = 256
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_range: float = 0.2
    clip_range_end: float = 0.1
    ent_coef_start: float = 0.01
    ent_coef_end: float = 0.002
    vf_coef: float = 0.8
    target_kl: float = 0.03
    seed: Optional[int] = 42
    device: Optional[str] = None
    tensorboard_log: Optional[str] = "runs/rl_selfplay"
    eval_episodes: int = 20
    snapshot_freq: int = 2_000
    rule_ratio_start: float = 0.8
    rule_ratio_end: float = 0.02


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    """SB3 兼容的线性调度：progress_remaining ∈ [0,1]。
    值随训练进度从 start → end 线性变化。
    """
    start = float(start)
    end = float(end)
    def fn(progress_remaining: float) -> float:
        return end + (start - end) * float(progress_remaining)
    return fn


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

        # 熵系数退火：从 ent_coef_start → ent_coef_end
        try:
            if hasattr(self.model, "ent_coef"):
                ent = (1 - progress) * self.cfg.ent_coef_start + progress * self.cfg.ent_coef_end
                self.model.ent_coef = float(ent)
        except Exception:
            pass

        # 每1000步打印一次进度和对手信息
        if self.num_timesteps % 1000 == 0:
            stats = self.pool.get_usage_stats()
            pool_summary = self.pool.get_pool_summary()
            print(f"训练步数: {self.num_timesteps}/{self.cfg.total_timesteps}, 规则对手占比: {ratio:.2f}, 进度: {progress*100:.1f}%")
            print(f"对手池: {pool_summary}")
            if sum(stats.values()) > 0:
                print(f"使用统计: 基础规则{stats['basic_rule']:.0f}% | 概率规则{stats['prob_rule']:.0f}% | 策略快照{stats['policy']:.0f}%")
            print("-" * 50)

        # 到达快照点：保存并注册为对手
        if self.num_timesteps >= self._next_snapshot:
            print(f"保存快照: {self.num_timesteps} 步")
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"snapshot_step_{self.num_timesteps}.zip")
            self.model.save(path)
            # 策略对手推断放CPU
            self.pool.add_policy(path, device="cpu")
            print(f"已保存快照: {path}")
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
                show_opponent_info=False,  # 在训练时不显示详细对手信息
            )

        from stable_baselines3.common.monitor import Monitor

        def make_env_with_monitor():
            env = make_env()
            return Monitor(env)  # 添加Monitor包装解决警告

        # 归一化包装：训练更新统计，评估仅使用统计不更新
        self.train_env = VecNormalize(
            DummyVecEnv([make_env_with_monitor]), norm_obs=True, norm_reward=True, clip_obs=10.0,
            norm_obs_keys=["obs"]  # 只归一化 obs 部分，不归一化 action_mask
        )
        self.eval_env = VecNormalize(
            DummyVecEnv([make_env_with_monitor]), norm_obs=True, norm_reward=False, clip_obs=10.0,
            norm_obs_keys=["obs"]  # 只归一化 obs 部分，不归一化 action_mask
        )
        # 共享统计并关闭评估期统计更新
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        self.eval_env.training = False

        policy_kwargs = make_default_policy_kwargs(PolicyNetConfig(
            features_dim=cfg.policy_hidden_size,
            pi_hidden=cfg.policy_hidden_size,
            vf_hidden=cfg.policy_hidden_size,
        ))

        self.model = PPO(
            policy=MaskedActorCriticPolicy,
            env=self.train_env,
            learning_rate=linear_schedule(cfg.learning_rate, cfg.learning_rate_end),
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=linear_schedule(cfg.clip_range, cfg.clip_range_end),
            ent_coef=cfg.ent_coef_start,
            vf_coef=cfg.vf_coef,
            target_kl=cfg.target_kl,
            tensorboard_log=cfg.tensorboard_log,
            policy_kwargs=policy_kwargs,
            seed=cfg.seed,
            device=cfg.device,
            verbose=1,  # 添加详细输出
        )

    def train(self):
        print(f"开始训练 - 总步数: {self.cfg.total_timesteps}, 玩家数: {self.cfg.num_players}, 设备: {self.cfg.device}")
        print(f"网络配置 - 隐藏层大小: {self.cfg.policy_hidden_size}, 学习率: {self.cfg.learning_rate}")
        print(f"PPO参数 - n_steps: {self.cfg.n_steps}, batch_size: {self.cfg.batch_size}")
        print(f"对手池初始配置: {self.pool.get_pool_summary()}")
        print(f"规则对手占比: {self.cfg.rule_ratio_start:.1f} → {self.cfg.rule_ratio_end:.1f}")
        print("-" * 60)

        save_dir = os.path.join(self.cfg.tensorboard_log or "runs", "snapshots")
        sp_cb = SelfPlayCallback(pool=self.pool, cfg=self.cfg, save_dir=save_dir)
        eval_cb = EvalCallback(self.eval_env, best_model_save_path=os.path.join(self.cfg.tensorboard_log or "runs", "best_model"),
                               log_path=self.cfg.tensorboard_log, eval_freq=max(2000, self.cfg.n_steps),  # 更频繁的评估
                               deterministic=False, render=False, n_eval_episodes=self.cfg.eval_episodes)

        print("开始 PPO 训练...")
        self.model.learn(total_timesteps=self.cfg.total_timesteps, callback=[sp_cb, eval_cb])
        print("训练完成!")

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
    parser.add_argument("--snapshot_freq", type=int, default=20_000)
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