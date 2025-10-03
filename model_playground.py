"""Evaluate a trained DQN model and optionally play against it interactively.

This standalone script lives outside the core training loop so that it can be
copied around or extended without touching the main code base.  It supports two
main workflows:

1. **Benchmarking** – run several episodes with the DQN agent controlling
   ``player_0`` and heuristic agents filling the remaining seats.  The script
   prints per-episode outcomes as well as aggregate win rate, average reward,
   and other quick diagnostics.

2. **Human vs. Model** – jump into a few rounds yourself to feel how the model
   behaves.  You will play as ``player_1``.  Any remaining seats are occupied by
   heuristic agents so that the table is always full.

Example usage::

    # Pure evaluation on 30 episodes, letting the script infer num_players
    python model_playground.py --model-path models/dqn_model.pth --eval-episodes 30

    # Run 10 evaluation episodes then play 3 interactive rounds yourself
    python model_playground.py --model-path models/dqn_model.pth \
        --eval-episodes 10 --play-rounds 3

The script keeps its dependencies minimal and does not mutate any project files
or configuration.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from env import LiarDiceEnv, Guess, Challenge
from agents.DQN_agent import DQNAgent
from agents.heuristic_agent import HeuristicRuleAgent


# ---------------------------------------------------------------------------
# Utilities for loading the trained agent
# ---------------------------------------------------------------------------


@dataclass
class EvalArgs:
    """Simple container that mimics the training arguments needed by DQNAgent."""

    learning_rate: float = 1e-3


def infer_num_players(checkpoint: Dict) -> Optional[int]:
    """Infer number of players from checkpoint metadata or tensor shapes."""

    meta_val = checkpoint.get("num_players")
    if isinstance(meta_val, int) and meta_val >= 2:
        return meta_val

    state = checkpoint.get("model_state_dict", {})
    count_head_weight = state.get("count_head.2.weight")
    if isinstance(count_head_weight, torch.Tensor):
        total_dice = count_head_weight.shape[0]
        if total_dice % 5 == 0:
            inferred = total_dice // 5
            if inferred >= 2:
                return inferred
    return None


def load_dqn_agent(model_path: Path, requested_players: Optional[int]) -> DQNAgent:
    """Instantiate the DQN agent and load weights from ``model_path``."""

    checkpoint = torch.load(model_path, map_location="cpu")

    inferred_players = infer_num_players(checkpoint)
    num_players = requested_players or inferred_players

    if num_players is None:
        raise ValueError(
            "Could not determine num_players from checkpoint. "
            "Please supply --num-players explicitly."
        )

    if (
        requested_players is not None
        and inferred_players is not None
        and requested_players != inferred_players
    ):
        print(
            f"⚠️  Requested num_players={requested_players} conflicts with checkpoint "
            f"({inferred_players}); using {inferred_players}."
        )
        num_players = inferred_players

    lr = checkpoint.get("learning_rate", 1e-3)
    agent = DQNAgent(
        agent_id="player_0",
        num_players=num_players,
        args=EvalArgs(learning_rate=lr),
    )

    agent.q_network.load_state_dict(checkpoint["model_state_dict"])
    agent.target_network.load_state_dict(checkpoint["target_state_dict"])
    agent.q_network.eval()
    agent.target_network.eval()

    return agent


# ---------------------------------------------------------------------------
# Automated evaluation helpers
# ---------------------------------------------------------------------------


def describe_guess(guess: Guess) -> str:
    return f"{guess.count} 个 {guess.face} ({guess.mode})"


def run_eval_episode(env: LiarDiceEnv, rl_agent: DQNAgent, opponents: Dict[str, HeuristicRuleAgent]) -> Dict[str, float]:
    env.reset()
    done = False

    total_reward = 0.0
    steps = 0
    challenges = 0

    while not done:
        agent_id = env.agent_selection
        observation, _, termination, truncation, _ = env.last()

        if termination or truncation:
            env.step(None)
            continue

        if agent_id == rl_agent.agent_id:
            action = rl_agent.get_action(observation)
        else:
            action = opponents[agent_id].get_action(observation)

        if isinstance(action, Challenge):
            challenges += 1

        env.step(action)
        total_reward += env.rewards[rl_agent.agent_id]
        steps += 1

        done = all(env.terminations.values()) or all(env.truncations.values())

    rl_penalty = env.penalties[rl_agent.agent_id]
    min_penalty = min(env.penalties.values())
    win = rl_penalty == min_penalty

    return {
        "reward": total_reward,
        "steps": steps,
        "win": float(win),
        "challenges": float(challenges),
    }


def evaluate_agent(agent: DQNAgent, episodes: int, opponent_confidence: int) -> None:
    if episodes <= 0:
        return

    env = LiarDiceEnv(num_players=agent.num_players)
    opponents = {
        agent_id: HeuristicRuleAgent(
            agent_id,
            agent.num_players,
            confidence_threshold=opponent_confidence,
        )
        for agent_id in env.possible_agents
        if agent_id != agent.agent_id
    }

    metrics = []
    print(f"\n=== Automated Evaluation ({episodes} episodes) ===")
    for idx in range(1, episodes + 1):
        stats = run_eval_episode(env, agent, opponents)
        metrics.append(stats)
        print(
            f"Episode {idx:>3}: win={bool(stats['win'])} "
            f"reward={stats['reward']:+.2f} steps={stats['steps']} "
            f"challenges={int(stats['challenges'])}"
        )

    win_rate = statistics.mean(m["win"] for m in metrics)
    avg_reward = statistics.mean(m["reward"] for m in metrics)
    avg_steps = statistics.mean(m["steps"] for m in metrics)
    avg_challenges = statistics.mean(m["challenges"] for m in metrics)

    print("\n--- Summary ---")
    print(f"Win rate       : {win_rate * 100:.1f}%")
    print(f"Average reward : {avg_reward:.2f}")
    print(f"Average steps  : {avg_steps:.1f}")
    print(f"Challenges/ep  : {avg_challenges:.1f}")


# ---------------------------------------------------------------------------
# Interactive play helpers
# ---------------------------------------------------------------------------


def show_human_prompt(env: LiarDiceEnv, observation: Dict) -> None:
    last_guess = observation.get("last_guess")
    if last_guess is None:
        guess_text = "无"
    else:
        guess_text = describe_guess(last_guess)

    dice_counts = observation["my_dice_counts"]
    hand = list(env.player_hands["player_1"]) if "player_1" in env.player_hands else []

    print("\n--- 你的回合 ---")
    print(f"你手牌统计: {dice_counts}  (实际骰子: {hand})")
    print(f"上一个猜测: {guess_text}")
    print("可选动作: [g]uess / [c]hallenge / [h]elp / [q]uit")


def request_guess(env: LiarDiceEnv, last_guess: Optional[Guess]) -> Guess:
    while True:
        mode_raw = input("模式 (f=飞, z=斋)> ").strip().lower()
        if mode_raw in {"f", "fly", "fei"}:
            mode = "飞"
        elif mode_raw in {"z", "zh", "zhai"}:
            mode = "斋"
        else:
            print("请输入 f 或 z。"); continue

        try:
            count = int(input("数量 count> ").strip())
            face = int(input("点数 face (1-6)> ").strip())
        except ValueError:
            print("请输入合法的整数。"); continue

        guess = Guess(mode=mode, count=count, face=face)
        if not env._is_legal(guess):
            print("该猜测不合法，请重试。")
            continue
        if last_guess is not None and not LiarDiceEnv._is_strictly_greater(guess, last_guess):
            print("该猜测没有比上一位更大，请重试。")
            continue
        return guess


def prompt_human_action(env: LiarDiceEnv, observation: Dict) -> Guess | Challenge:
    last_guess = observation.get("last_guess")

    while True:
        show_human_prompt(env, observation)
        choice = input("选择动作> ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if choice in {"h", "help"}:
            print("提示: 首轮必须 count > 玩家数；飞模式不能喊 1；挑战只有在已有猜测时才合法。")
            continue
        if choice in {"c", "challenge"}:
            if last_guess is None:
                print("目前还没有任何猜测，不能挑战。")
                continue
            return Challenge()
        if choice in {"g", "guess"}:
            return request_guess(env, last_guess)

        print("无法识别的输入，请输入 g/c/h/q。")


def play_against_model(agent: DQNAgent, rounds: int, opponent_confidence: int) -> None:
    if rounds <= 0:
        return

    env = LiarDiceEnv(num_players=agent.num_players)

    # Human 扮演 player_1，其余座位由启发式 agent 填充
    opponents = {}
    for agent_id in env.possible_agents:
        if agent_id in {agent.agent_id, "player_1"}:
            continue
        opponents[agent_id] = HeuristicRuleAgent(
            agent_id,
            agent.num_players,
            confidence_threshold=opponent_confidence,
        )

    print("\n=== 开始互动对局 ===")
    print("你是 player_1，AI 是 player_0。其余玩家由启发式策略扮演。")

    try:
        for round_idx in range(1, rounds + 1):
            env.reset()
            done = False
            print(f"\n>>> Round {round_idx}")

            while not done:
                agent_id = env.agent_selection
                observation, _, termination, truncation, _ = env.last()

                if termination or truncation:
                    env.step(None)
                    continue

                if agent_id == agent.agent_id:
                    action = agent.get_action(observation)
                    print(f"AI 行动: {describe_guess(action) if isinstance(action, Guess) else '挑战!'}")
                elif agent_id == "player_1":
                    action = prompt_human_action(env, observation)
                else:
                    action = opponents[agent_id].get_action(observation)
                    if isinstance(action, Guess):
                        print(f"{agent_id} 猜测: {describe_guess(action)}")
                    else:
                        print(f"{agent_id} 发起挑战！")

                env.step(action)
                done = all(env.terminations.values()) or all(env.truncations.values())

            # 输出本轮结果
            penalties = env.penalties
            loser = max(penalties, key=penalties.get)
            print("本轮结束！罚分情况：")
            for pid, val in penalties.items():
                print(f"  {pid}: {val}")
            if loser == "player_1":
                print("本轮结果：你输掉了 😢")
            elif loser == agent.agent_id:
                print("本轮结果：AI 被击败了 🎉")
            else:
                print(f"本轮结果：{loser} 落败")

    except KeyboardInterrupt:
        print("\n互动对局提前结束。")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or play against a trained DQN model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/dqn_model.pth"),
        help="Path to the saved checkpoint produced by train.py (default: models/dqn_model.pth)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Total number of players (default: 2)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run automatically (default: 10)",
    )
    parser.add_argument(
        "--play-rounds",
        type=int,
        default=0,
        help="Number of interactive rounds to play as player_1",
    )
    parser.add_argument(
        "--opp-confidence",
        type=int,
        default=1,
        help="Challenge aggressiveness of heuristic opponents (default: 1)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if args.num_players is not None and args.num_players < 2:
        raise ValueError("num_players must be at least 2")

    agent = load_dqn_agent(args.model_path, args.num_players)
    print(f"Loaded model for num_players={agent.num_players}")

    if args.eval_episodes > 0:
        evaluate_agent(agent, args.eval_episodes, args.opp_confidence)

    if args.play_rounds > 0:
        play_against_model(agent, args.play_rounds, args.opp_confidence)

    if args.eval_episodes <= 0 and args.play_rounds <= 0:
        print("未指定 --eval-episodes 或 --play-rounds，脚本完成。")


if __name__ == "__main__":
    main()
