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
    learning_rate: float = 2e-4
    learning_rate_end: float = 5e-5
    policy_hidden_size: int = 256
    n_steps: int = 4096
    n_epochs: int = 6
    batch_size: int = 512
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_end: float = 0.1
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.005
    vf_coef: float = 0.7
    target_kl: float = 0.05
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
        self._log_interval = 2_000
        self._diagnostic_interval = 20_000
        self._next_diagnostic = self._diagnostic_interval

        # 诊断用的历史数据
        self._reward_history = []
        self._ep_len_history = []
        self._loss_history = []
        self._explained_var_history = []

    def _on_step(self) -> bool:
        # 更新规则体占比：线性从 start 到 end
        progress = min(1.0, self.num_timesteps / max(1, self.cfg.total_timesteps))
        ratio = (1 - progress) * self.cfg.rule_ratio_start + progress * self.cfg.rule_ratio_end
        self.pool.set_rule_ratio(ratio)

        # 熵系数退火：从 ent_coef_start → ent_coef_end
        try:
            if hasattr(self.model, "ent_coef"):
                ent = (1 - progress) * self.cfg.ent_coef_start + progress * self.cfg.ent_coef_end
                value = float(ent)
                self.model.ent_coef = value
                if hasattr(self.model, "ent_coef_tensor"):
                    self.model.ent_coef_tensor = th.tensor(value, device=self.model.device)
        except Exception:
            pass

        # 定期输出训练摘要
        if self.num_timesteps % self._log_interval == 0 or self.num_timesteps == 1:
            # 收集指标
            self._collect_metrics()

            stats = self.pool.get_usage_stats()
            pool_summary = self.pool.get_pool_summary()
            print(f"训练步数: {self.num_timesteps:,}/{self.cfg.total_timesteps:,} (进度 {progress*100:.1f}%), 规则对手占比: {ratio:.2f}")
            print(f"对手池: {pool_summary}")
            if sum(stats.values()) > 0:
                print(
                    "使用统计: 基础规则{basic:.0f}% | 概率规则{prob:.0f}% | 策略快照{policy:.0f}%".format(
                        basic=stats.get('basic_rule', 0.0),
                        prob=stats.get('prob_rule', 0.0),
                        policy=stats.get('policy', 0.0),
                    )
                )
            print("当前熵系数: {:.4f}".format(getattr(self.model, "ent_coef", 0.0)))
            print("-" * 60)
            self.pool.reset_stats()

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

        # 到达诊断点：输出训练状态诊断
        if self.num_timesteps >= self._next_diagnostic:
            self._print_diagnostic(progress)
            self._next_diagnostic += self._diagnostic_interval

        return True

    def _on_rollout_end(self) -> None:
        """在每次 rollout 结束时收集指标（此时 logger 已更新）"""
        self._collect_metrics()

    def _collect_metrics(self):
        """收集当前训练指标用于诊断"""
        # 从 locals 和 logger 收集指标（SB3 回调机制）
        try:
            # 方法1: 从 self.locals 获取（在训练循环中可用）
            if hasattr(self, 'locals') and self.locals:
                # 回报和回合长度从 rollout buffer 的 info 获取
                if 'infos' in self.locals:
                    infos = self.locals['infos']
                    for info in infos:
                        if isinstance(info, dict):
                            if 'episode' in info:
                                ep_info = info['episode']
                                if 'r' in ep_info:
                                    self._reward_history.append(float(ep_info['r']))
                                if 'l' in ep_info:
                                    self._ep_len_history.append(float(ep_info['l']))

            # 方法2: 从 logger 的 name_to_value 字典获取
            if hasattr(self.logger, 'name_to_value'):
                metrics = self.logger.name_to_value
                if 'rollout/ep_rew_mean' in metrics:
                    self._reward_history.append(float(metrics['rollout/ep_rew_mean']))
                if 'rollout/ep_len_mean' in metrics:
                    self._ep_len_history.append(float(metrics['rollout/ep_len_mean']))
                if 'train/loss' in metrics:
                    self._loss_history.append(float(metrics['train/loss']))
                if 'train/explained_variance' in metrics:
                    self._explained_var_history.append(float(metrics['train/explained_variance']))

            # 方法3: 从 logger 的 name_to_value 属性（SB3 新版本）
            elif hasattr(self.logger, 'get_log_dict'):
                metrics = self.logger.get_log_dict()
                if 'rollout/ep_rew_mean' in metrics:
                    self._reward_history.append(float(metrics['rollout/ep_rew_mean']))
                if 'rollout/ep_len_mean' in metrics:
                    self._ep_len_history.append(float(metrics['rollout/ep_len_mean']))
                if 'train/loss' in metrics:
                    self._loss_history.append(float(metrics['train/loss']))
                if 'train/explained_variance' in metrics:
                    self._explained_var_history.append(float(metrics['train/explained_variance']))
        except Exception as e:
            # 静默失败，不影响训练
            pass

    def _print_diagnostic(self, progress: float):
        """输出详细的训练诊断信息"""
        self._collect_metrics()

        print("\n" + "=" * 80)
        print(f"🔬 训练诊断报告 - 步数: {self.num_timesteps:,} / {self.cfg.total_timesteps:,} (进度 {progress*100:.1f}%)")
        print("=" * 80)

        # 1. 性能评估
        print("\n📊 【性能评估】")

        # 尝试从多个来源获取回报数据
        avg_reward = None
        reward_std = None

        if self._reward_history:
            recent_rewards = self._reward_history[-10:]
            avg_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)
            reward_trend = "上升" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else "下降"

            print(f"  • 平均回报: {avg_reward:.3f} (std: {reward_std:.3f})")
            print(f"  • 回报趋势: {reward_trend}")
        else:
            # 备用方案：从 logger 直接读取最新值
            try:
                if hasattr(self.logger, 'name_to_value'):
                    metrics = self.logger.name_to_value
                    if 'rollout/ep_rew_mean' in metrics:
                        avg_reward = float(metrics['rollout/ep_rew_mean'])
                        print(f"  • 当前平均回报: {avg_reward:.3f}")
                    else:
                        print("  • 回报数据: 暂无（训练初期或采样中）")
                else:
                    print("  • 回报数据: 暂无（训练初期或采样中）")
            except:
                print("  • 回报数据: 暂无（训练初期或采样中）")

        # 性能评级
        if avg_reward is not None:
            if avg_reward > 0.5:
                status = "✅ 优秀"
            elif avg_reward > 0.0:
                status = "✓ 良好"
            elif avg_reward > -0.5:
                status = "⚠️ 一般"
            else:
                status = "❌ 较差"
            print(f"  • 性能等级: {status}")

        # 2. 稳定性分析
        print("\n🎯 【稳定性分析】")

        loss_mean = None
        if self._loss_history:
            recent_loss = self._loss_history[-10:]
            loss_std = np.std(recent_loss)
            loss_mean = np.mean(recent_loss)

            print(f"  • 平均损失: {loss_mean:.4f} (std: {loss_std:.4f})")
            if loss_std < 0.1:
                print("  • 损失稳定性: ✅ 稳定")
            elif loss_std < 0.3:
                print("  • 损失稳定性: ⚠️ 轻微波动")
            else:
                print("  • 损失稳定性: ❌ 波动较大")
        else:
            try:
                if hasattr(self.logger, 'name_to_value'):
                    metrics = self.logger.name_to_value
                    if 'train/loss' in metrics:
                        loss_mean = float(metrics['train/loss'])
                        print(f"  • 当前损失: {loss_mean:.4f}")
                    else:
                        print("  • 损失数据: 暂无（训练初期）")
                else:
                    print("  • 损失数据: 暂无（训练初期）")
            except:
                print("  • 损失数据: 暂无（训练初期）")

        ev_mean = None
        if self._explained_var_history:
            recent_ev = self._explained_var_history[-10:]
            ev_mean = np.mean(recent_ev)

            print(f"  • 解释方差: {ev_mean:.3f}")
        else:
            try:
                if hasattr(self.logger, 'name_to_value'):
                    metrics = self.logger.name_to_value
                    if 'train/explained_variance' in metrics:
                        ev_mean = float(metrics['train/explained_variance'])
                        print(f"  • 当前解释方差: {ev_mean:.3f}")
                    else:
                        print("  • 解释方差: 暂无（训练初期）")
                else:
                    print("  • 解释方差: 暂无（训练初期）")
            except:
                print("  • 解释方差: 暂无（训练初期）")

        # 值函数质量评估
        if ev_mean is not None:
            if ev_mean > 0.5:
                print("  • 值函数质量: ✅ 优秀")
            elif ev_mean > 0.2:
                print("  • 值函数质量: ✓ 良好")
            elif ev_mean > 0:
                print("  • 值函数质量: ⚠️ 一般（可能需要调整 vf_coef 或 n_steps）")
            else:
                print("  • 值函数质量: ❌ 较差（建议检查奖励塑形和状态表示）")

        # 3. 行为分析
        print("\n🎮 【策略行为】")

        avg_len = None
        if self._ep_len_history:
            recent_len = self._ep_len_history[-10:]
            avg_len = np.mean(recent_len)
            print(f"  • 平均回合长度: {avg_len:.2f} 步")
        else:
            try:
                if hasattr(self.logger, 'name_to_value'):
                    metrics = self.logger.name_to_value
                    if 'rollout/ep_len_mean' in metrics:
                        avg_len = float(metrics['rollout/ep_len_mean'])
                        print(f"  • 当前回合长度: {avg_len:.2f} 步")
                    else:
                        print("  • 回合长度: 暂无（训练初期）")
                else:
                    print("  • 回合长度: 暂无（训练初期）")
            except:
                print("  • 回合长度: 暂无（训练初期）")

        if avg_len is not None:
            expected_len = 2.5 + (self.cfg.num_players - 2) * 0.5

            if avg_len < 1.8:
                print("  • 策略类型: ⚠️ 过于保守（频繁挑战）")
            elif avg_len < expected_len * 0.8:
                print("  • 策略类型: 偏保守")
            elif avg_len > expected_len * 1.3:
                print("  • 策略类型: 偏激进（可能过度叫牌）")
            else:
                print("  • 策略类型: ✓ 均衡")

        # 4. 探索状态
        print("\n🔍 【探索状态】")
        current_ent = getattr(self.model, "ent_coef", 0.0)
        print(f"  • 当前熵系数: {current_ent:.4f}")
        if current_ent > 0.01:
            print("  • 探索程度: 🔥 积极探索中")
        elif current_ent > 0.005:
            print("  • 探索程度: ✓ 适度探索")
        else:
            print("  • 探索程度: ❄️ 收敛阶段（低探索）")

        # 5. 对手池状态
        print("\n👥 【对手池状态】")
        stats = self.pool.get_usage_stats()
        pool_summary = self.pool.get_pool_summary()
        print(f"  • 对手池配置: {pool_summary}")
        if sum(stats.values()) > 0:
            print(f"  • 规则对手: {stats.get('basic_rule', 0.0) + stats.get('prob_rule', 0.0):.0f}%")
            print(f"  • 策略快照: {stats.get('policy', 0.0):.0f}%")

        # 6. 建议
        print("\n💡 【训练建议】")
        issues = []

        if self._reward_history and np.mean(self._reward_history[-10:]) < -0.5:
            issues.append("• 回报偏低：考虑检查奖励塑形系数或增加探索")

        if self._explained_var_history and np.mean(self._explained_var_history[-10:]) < 0.1:
            issues.append("• 值函数质量差：建议提高 vf_coef (当前 {:.1f}) 或增大 n_steps".format(self.cfg.vf_coef))

        if self._ep_len_history and np.mean(self._ep_len_history[-10:]) < 1.8:
            issues.append("• 回合过短：策略过于保守，可能需要降低挑战奖励")

        if self._loss_history and np.std(self._loss_history[-10:]) > 0.5:
            issues.append("• 训练不稳定：考虑降低学习率或减小 target_kl")

        if not issues:
            print("  ✅ 训练状态良好，继续保持！")
        else:
            for issue in issues:
                print(f"  {issue}")

        print("\n" + "=" * 80 + "\n")


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

        # 归一化包装：训练期仅归一化观测，不归一化奖励；评估仅使用统计不更新
        self.train_env = VecNormalize(
            DummyVecEnv([make_env_with_monitor]), norm_obs=True, norm_reward=False, clip_obs=10.0,
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
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=linear_schedule(cfg.clip_range, cfg.clip_range_end),
            clip_range_vf=1.0,
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
        print(f"PPO参数 - n_steps: {self.cfg.n_steps}, n_epochs: {self.cfg.n_epochs}, batch_size: {self.cfg.batch_size}")
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
