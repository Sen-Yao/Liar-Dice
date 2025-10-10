"""
rl_specialized.evaluator - 专用 PPO 模型评估器

用途：
- 评估基于 rl_specialized 训练流程得到的 PPO 策略
- 与 baseline_agents 提供的基线对手进行对战
- 输出详细日志或 JSON 摘要，便于批量实验整合
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _ensure_project_root_on_path() -> None:
    """保证脚本可直接运行，优先加载仓库内模块。"""
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_project_root_on_path()

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - 提示友好错误
    PPO = None

import gymnasium as gym

from env import LiarDiceEnv
from agents.baseline_agents import (
    RandomAgent,
    ConservativeAgent,
    AggressiveAgent,
    OptimizedLLMAgent,
)
from agents.heuristic_agent import HeuristicRuleAgent
from rl_specialized.utils.state_encoder import create_state_encoder


class PPOPolicyAgent:
    """PPO 策略包装器，将 SB3 模型适配为 baseline agent 接口。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        num_players: int = 2,
        dice_per_player: int = 5,
        history_len: int = 0,
        deterministic: bool = True,
    ):
        if PPO is None:
            raise RuntimeError(
                "未检测到 stable-baselines3，请安装后再运行：pip install \"stable-baselines3[extra]\""
            )

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}")

        self.device = device
        self.deterministic = deterministic
        self.encoder = create_state_encoder(
            num_players=num_players,
            dice_per_player=dice_per_player,
            history_length=max(0, history_len),
        )
        self.model = PPO.load(str(self.model_path), device=device)
        self.bound_env: Optional[LiarDiceEnv] = None

        # 维度校验，提前捕获训练配置不一致的问题
        obs_space = getattr(self.model, "observation_space", None)
        if isinstance(obs_space, gym.spaces.Dict):
            trained_dim = int(obs_space["obs"].shape[0])
            current_dim = self.encoder.get_feature_size()
            if trained_dim != current_dim:
                raise RuntimeError(
                    "PPO 模型与当前评估配置不兼容：状态编码维度不一致。\n"
                    f"  训练时特征维度: {trained_dim}\n"
                    f"  当前评估特征维度: {current_dim}\n"
                    "请确认 num_players、dice_per_player、history_len 等配置一致，或重新导出模型。"
                )

        action_space = getattr(self.model, "action_space", None)
        if isinstance(action_space, gym.spaces.Discrete):
            self.trained_action_dim = int(action_space.n)
        else:
            self.trained_action_dim = None

    def bind_environment(self, env: LiarDiceEnv) -> None:
        """绑定环境实例，便于内部访问动作掩码与转换。"""
        self.bound_env = env

        if self.trained_action_dim is not None:
            counts_per_mode = env.total_dice - env.min_count + 1
            counts_per_mode = max(0, counts_per_mode)
            current_action_dim = 1 + (2 * counts_per_mode * 6)
            if current_action_dim != self.trained_action_dim:
                raise RuntimeError(
                    "PPO 模型与当前环境的动作空间不匹配。\n"
                    f"  训练时动作维度: {self.trained_action_dim}\n"
                    f"  当前环境动作维度: {current_action_dim}\n"
                    "请确认 num_players、dice_per_player、历史长度等配置一致。"
                )

    def get_action(self, observation: Dict) -> object:
        """返回 PPO 模型的动作对象（Guess 或 Challenge）。"""
        if self.bound_env is None:
            raise RuntimeError("PPOPolicyAgent 尚未绑定环境，无法获取动作。")

        obs_vec = self.encoder.encode_observation(observation)
        action_mask = self.bound_env.get_action_mask(observation)

        model_obs = {
            "obs": np.expand_dims(obs_vec, axis=0),
            "action_mask": np.expand_dims(action_mask.astype(np.int8), axis=0),
        }

        action_id, _ = self.model.predict(model_obs, deterministic=self.deterministic)
        # 处理 SB3 返回的数组格式（兼容 NumPy 1.25+）
        action_int = int(action_id.item() if hasattr(action_id, 'item') else action_id)
        action_obj = self.bound_env.action_to_object(action_int)
        return action_obj


class SpecializedEvaluator:
    """PPO 模型评估器，复用 baseline 对手。"""

    def __init__(
        self,
        model_path: str,
        num_players: int = 2,
        dice_per_player: int = 5,
        history_len: int = 0,
        device: str = "cpu",
        deterministic: bool = True,
        verbose: bool = True,
    ):
        self.model_path = model_path
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.history_len = history_len
        self.device = device
        self.deterministic = deterministic
        self.verbose = verbose

        history_arg = None if history_len <= 0 else history_len
        self.env = LiarDiceEnv(
            num_players=num_players,
            dice_per_player=dice_per_player,
            history_len=history_arg,
            use_specialized_action_space=True,
        )

        self.policy_agent = PPOPolicyAgent(
            model_path=model_path,
            device=device,
            num_players=num_players,
            dice_per_player=dice_per_player,
            history_len=history_len,
            deterministic=deterministic,
        )
        self.policy_agent.bind_environment(self.env)

        self.results: Dict[str, Dict] = {}

    def evaluate_against_opponent(
        self,
        opponent_name: str,
        opponent_agent,
        num_games: int = 100,
    ) -> Dict:
        """评估 PPO 模型对抗单一对手的表现。"""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"开始评估 PPO 模型 vs {opponent_name}")
            print(f"总对局数: {num_games}")
            print(f"{'=' * 60}")

        wins = 0
        total_rounds = 0
        episode_lengths: List[int] = []
        policy_actions = defaultdict(int)
        opponent_actions = defaultdict(int)

        start_time = time.time()

        for game_idx in range(num_games):
            agents = {"player_0": self.policy_agent, "player_1": opponent_agent}
            self.env.reset()

            # 若对手带有 reset_stats，重置一次
            if hasattr(opponent_agent, "reset_stats"):
                opponent_agent.reset_stats()

            round_count = 0

            for agent_id in self.env.agent_iter(max_iter=200):
                observation = self.env.observe(agent_id)
                terminated = self.env.terminations[agent_id]
                truncated = self.env.truncations[agent_id]

                if terminated or truncated:
                    action = None
                else:
                    agent = agents[agent_id]
                    action = agent.get_action(observation)
                    action_type = type(action).__name__
                    if agent_id == "player_0":
                        policy_actions[action_type] += 1
                    else:
                        opponent_actions[action_type] += 1
                    round_count += 1

                self.env.step(action)

                if all(self.env.terminations.values()) or all(self.env.truncations.values()):
                    break

            reward = self.env.rewards.get("player_0", 0)
            if reward >= 0:
                wins += 1

            total_rounds += round_count
            episode_lengths.append(round_count)

            if self.verbose:
                current = game_idx + 1
                winrate = wins / current
                print(
                    f"[{opponent_name}] 进度 {current}/{num_games} | 当前胜率 {winrate:.1%} | "
                    f"最近一局回合数 {round_count}"
                )

        elapsed = time.time() - start_time
        winrate = wins / num_games
        avg_rounds = total_rounds / num_games if num_games else 0.0

        policy_total = sum(policy_actions.values())
        opp_total = sum(opponent_actions.values())

        policy_challenge_rate = (
            policy_actions["Challenge"] / policy_total if policy_total else 0.0
        )
        opponent_challenge_rate = (
            opponent_actions["Challenge"] / opp_total if opp_total else 0.0
        )

        result = {
            "opponent": opponent_name,
            "num_games": num_games,
            "wins": wins,
            "winrate": winrate,
            "avg_game_length": avg_rounds,
            "total_rounds": total_rounds,
            "policy_actions": dict(policy_actions),
            "opponent_actions": dict(opponent_actions),
            "policy_challenge_rate": policy_challenge_rate,
            "opponent_challenge_rate": opponent_challenge_rate,
            "elapsed_time": elapsed,
            "episode_lengths": episode_lengths,
        }

        if self.verbose:
            self._print_match_summary(result)

        return result

    def _print_match_summary(self, result: Dict) -> None:
        """输出单个对手的评估摘要。"""
        print(f"\n{'-' * 60}")
        print(f"对局总结: PPO vs {result['opponent']}")
        print(f"PPO胜场: {result['wins']} / {result['num_games']} ({result['winrate']:.1%})")
        print(f"平均回合数: {result['avg_game_length']:.2f}")
        print(f"耗时: {result['elapsed_time']:.2f} 秒")
        print("动作统计：")
        print(f"  PPO 挑战率: {result['policy_challenge_rate']:.1%}")
        print(f"  对手挑战率: {result['opponent_challenge_rate']:.1%}")

    def run_full_evaluation(
        self,
        num_games: int = 100,
        include_llm: bool = False,
    ) -> Dict[str, Dict]:
        """遍历所有基线对手并汇总结果。"""
        opponents = [
            ("RandomAgent", RandomAgent("opponent", num_players=self.num_players, seed=42)),
            ("ConservativeAgent", ConservativeAgent("opponent", num_players=self.num_players)),
            ("AggressiveAgent", AggressiveAgent("opponent", num_players=self.num_players)),
            ("HeuristicRuleAgent", HeuristicRuleAgent("opponent", num_players=self.num_players)),
        ]

        if include_llm:
            try:
                llm_agent = OptimizedLLMAgent("opponent", num_players=self.num_players, use_api=True)
                opponents.append(("OptimizedLLMAgent", llm_agent))
            except Exception as exc:
                print(f"⚠️ LLM 对手创建失败（跳过）：{exc}")

        if self.verbose:
            print(f"\n{'#' * 60}")
            print("# PPO 模型评估")
            print(f"# 模型路径: {self.model_path}")
            print(f"# 玩家数: {self.num_players} | 每人骰子: {self.dice_per_player}")
            print(f"# 历史长度: {self.history_len} | 动作决策: {'贪心' if self.deterministic else '采样'}")
            print(f"# 对手数量: {len(opponents)} | 每个对手对局数: {num_games}")
            print(f"{'#' * 60}")

        all_results: Dict[str, Dict] = {}

        for index, (name, agent) in enumerate(opponents, start=1):
            if self.verbose:
                print(f"\n[{index}/{len(opponents)}] 评估 {name} ...")
            result = self.evaluate_against_opponent(
                opponent_name=name,
                opponent_agent=agent,
                num_games=num_games,
            )
            all_results[name] = result

        self.results = all_results

        if self.verbose:
            self._print_overall_summary()

        return all_results

    def _print_overall_summary(self) -> None:
        """打印所有对手的汇总表。"""
        if not self.results:
            return

        print(f"\n{'=' * 60}")
        print("综合评估报告")
        print(f"{'对手':<20} {'对局数':>8} {'胜率':>10} {'平均回合':>10} {'挑战率':>10}")
        print(f"{'-' * 60}")

        total_games = 0
        total_wins = 0
        challenge_rates: List[float] = []

        for name, result in self.results.items():
            print(
                f"{name:<20} "
                f"{result['num_games']:>8} "
                f"{result['winrate']:>9.1%} "
                f"{result['avg_game_length']:>10.2f} "
                f"{result['policy_challenge_rate']:>9.1%}"
            )
            total_games += result["num_games"]
            total_wins += result["wins"]
            challenge_rates.append(result["policy_challenge_rate"])

        overall = total_wins / total_games if total_games else 0.0
        avg_challenge = statistics.mean(challenge_rates) if challenge_rates else 0.0

        print(f"{'-' * 60}")
        print(f"{'总计':<20} {total_games:>8} {overall:>9.1%} {'' :>10} {avg_challenge:>9.1%}")
        print(f"{'=' * 60}")

    def save_results(self, output_path: Optional[str] = None) -> None:
        """保存评估结果到文本文件。"""
        if not self.results:
            print("暂无评估结果可保存。")
            return

        if output_path is None:
            stem = Path(self.model_path).stem
            output_path = f"specialized_eval_{stem}.txt"

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("PPO 模型评估报告\n")
            fh.write("=" * 60 + "\n")
            fh.write(f"模型: {self.model_path}\n")
            fh.write(f"玩家数: {self.num_players}, 每人骰子: {self.dice_per_player}\n")
            fh.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write("=" * 60 + "\n\n")

            for name, result in self.results.items():
                fh.write(f"对手: {name}\n")
                fh.write("-" * 40 + "\n")
                fh.write(f"对局数: {result['num_games']}\n")
                fh.write(f"胜场: {result['wins']} ({result['winrate']:.1%})\n")
                fh.write(f"平均回合数: {result['avg_game_length']:.2f}\n")
                fh.write(f"PPO 挑战率: {result['policy_challenge_rate']:.1%}\n")
                fh.write(f"对手挑战率: {result['opponent_challenge_rate']:.1%}\n\n")

        print(f"\n✓ 评估结果已保存到: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rl_specialized PPO 模型评估器")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="PPO 模型路径（.zip 文件），默认使用 runs/rl_selfplay/best_model/best_model.zip",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="每个对手的评估局数",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="环境玩家数（需与训练一致）",
    )
    parser.add_argument(
        "--dice-per-player",
        type=int,
        default=5,
        help="每位玩家的骰子数量",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=3,
        help="观测中保留的历史长度（需与训练保持一致，默认 3）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="推理设备",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用随机策略（关闭则为贪心）",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="排除 OptimizedLLMAgent 对手（默认包含）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="仅输出 JSON 摘要，适合脚本调用",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="评估完成后保存文本报告",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式，减少控制台输出",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet and not args.json

    # 自动查找模型路径
    model_path = args.model_path
    if model_path is None:
        # 默认使用训练产生的最佳模型
        default_path = Path("runs/rl_selfplay/best_model/best_model.zip")
        if default_path.is_file():
            model_path = str(default_path)
            if verbose:
                print(f"未指定 --model-path，使用默认模型: {model_path}")
        else:
            # 备选：查找最新的快照
            snapshot_dir = Path("runs/rl_selfplay/snapshots")
            if snapshot_dir.exists():
                snapshots = sorted(snapshot_dir.glob("snapshot_step_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
                if snapshots:
                    model_path = str(snapshots[0])
                    if verbose:
                        print(f"未找到 best_model，使用最新快照: {model_path}")
                else:
                    print("❌ 错误: 未找到任何训练模型！")
                    print("请先运行训练：python -m rl_specialized.training.train_selfplay --timesteps 400000")
                    sys.exit(1)
            else:
                print("❌ 错误: 未找到训练输出目录 runs/rl_selfplay/")
                print("请先运行训练：python -m rl_specialized.training.train_selfplay --timesteps 400000")
                sys.exit(1)

    evaluator = SpecializedEvaluator(
        model_path=model_path,
        num_players=args.num_players,
        dice_per_player=args.dice_per_player,
        history_len=args.history_len,
        device=args.device,
        deterministic=not args.stochastic,
        verbose=verbose,
    )

    results = evaluator.run_full_evaluation(
        num_games=max(1, args.num_games),
        include_llm=not args.no_llm,  # 默认包含 LLM，除非指定 --no-llm
    )

    if args.save_results and not args.json:
        evaluator.save_results()

    if args.json:
        payload: List[Dict] = []
        for name, res in results.items():
            payload.append(
                {
                    "opponent": name,
                    "num_games": res["num_games"],
                    "winrate": res["winrate"],
                    "avg_game_length": res["avg_game_length"],
                }
            )
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
