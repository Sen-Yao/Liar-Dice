#!/usr/bin/env python3
"""
LLM对战评估脚本

支持LLM代理与多种其他代理类型进行对战，包括：
- Random Agent (随机策略)
- Conservative Agent (保守策略)
- Aggressive Agent (激进策略)
- Heuristic Agent (启发式策略)

统计胜率、平均奖励等指标。
"""

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict
from env import LiarDiceEnv, Challenge
from agents.heuristic_agent import HeuristicRuleAgent
from agents.basic_agent import BasicRuleAgent, ProbabilisticBasicAgent
from agents.baseline_agents import RandomAgent, ConservativeAgent, AggressiveAgent, OptimizedLLMAgent


@dataclass
class GameResult:
    """单局游戏结果"""
    winner: str
    loser: str
    llm_reward: float
    llm_position: int  # LLM代理在游戏中的位置索引
    steps: int
    challenges: int


def is_llm_winner(result: GameResult) -> bool:
    """判断是否由LLM代理获胜"""
    return result.winner == f"player_{result.llm_position}"



def create_agent(agent_type: str, agent_id: str, num_players: int, enable_thinking: bool = False) -> object:
    """创建指定类型的代理"""
    if agent_type == "random":
        return RandomAgent(agent_id, num_players)
    elif agent_type == "conservative":
        return ConservativeAgent(agent_id, num_players)
    elif agent_type == "aggressive":
        return AggressiveAgent(agent_id, num_players)
    elif agent_type == "heuristic":
        return HeuristicRuleAgent(agent_id, num_players, confidence_threshold=1)
    elif agent_type == "basic":
        return BasicRuleAgent(agent_id, num_players)
    elif agent_type == "probabilistic":
        return ProbabilisticBasicAgent(agent_id, num_players)
    elif agent_type == "llm":
        return OptimizedLLMAgent(agent_id, num_players, enable_thinking=enable_thinking)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_single_game(llm_agent: OptimizedLLMAgent, opponent_types: List[str], llm_position: int = 0, enable_thinking: bool = False) -> GameResult:
    """运行单局游戏"""
    num_players = len(opponent_types) + 1
    env = LiarDiceEnv(num_players=num_players)

    # 创建所有代理
    agents = {}
    agent_types = {}

    # 放置LLM代理
    player_names = [f"player_{i}" for i in range(num_players)]
    llm_player = player_names[llm_position]
    agents[llm_player] = llm_agent
    agent_types[llm_player] = "llm"

    # 创建对手代理（如果是LLM也启用思考模式）
    opponent_players = [p for i, p in enumerate(player_names) if i != llm_position]
    for player, opp_type in zip(opponent_players, opponent_types):
        agents[player] = create_agent(opp_type, player, num_players, enable_thinking=enable_thinking)
        agent_types[player] = opp_type

    # 运行游戏
    env.reset()
    steps = 0
    challenges = 0

    while not all(env.terminations.values()) and not all(env.truncations.values()):
        current_player = env.agent_selection
        observation, _, termination, truncation, _ = env.last()

        if termination or truncation:
            env.step(None)
            continue

        agent = agents[current_player]
        action = agent.get_action(observation)

        if isinstance(action, Challenge):
            challenges += 1

        env.step(action)
        steps += 1

    # 确定胜负
    penalties = env.penalties
    min_penalty = min(penalties.values())
    max_penalty = max(penalties.values())

    winner = [pid for pid, penalty in penalties.items() if penalty == min_penalty][0]
    loser = [pid for pid, penalty in penalties.items() if penalty == max_penalty][0]

    return GameResult(
        winner=winner,
        loser=loser,
        llm_reward=env.rewards[llm_player],
        llm_position=llm_position,
        steps=steps,
        challenges=challenges
    )


def run_tournament(llm_agent: OptimizedLLMAgent, opponent_types: List[str], num_games: int, enable_thinking: bool = False) -> Dict[str, List[GameResult]]:
    """运行两人对战锦标赛，LLM与每个对手进行num_games局对战"""
    results = defaultdict(list)

    print(f"开始两人对战锦标赛：每个对手进行 {num_games} 局游戏")
    print(f"LLM代理将对抗的对手类型：{opponent_types}")
    if enable_thinking:
        print("思考模式：已启用")

    for opponent_type in opponent_types:
        print(f"\n对手 {opponent_type} 对战:")

        matchup_results = []

        for game_idx in range(num_games):
            # 随机分配LLM的位置（0或1）
            llm_position = random.randint(0, 1)

            # 运行游戏
            result = run_single_game(llm_agent, [opponent_type], llm_position, enable_thinking=enable_thinking)
            matchup_results.append(result)

            # 计算当前胜率
            wins = sum(1 for r in matchup_results if is_llm_winner(r))
            win_rate = wins / len(matchup_results) * 100

            print(
                f"  第{game_idx + 1}/{num_games}局:"
                f" 胜方 {result.winner}"
                f" | 当前LLM胜率 {win_rate:.1f}% ({wins}/{len(matchup_results)})"
                f" | 平均奖励 {statistics.mean(r.llm_reward for r in matchup_results):+.1f}"
            )

        # 保存结果
        matchup_key = f"LLM_vs_{opponent_type}"
        results[matchup_key] = matchup_results

    return dict(results)


def analyze_results(results: Dict[str, List[GameResult]]) -> None:
    """分析并显示结果统计"""
    print("\n" + "="*60)
    print("对战结果统计")
    print("="*60)

    total_games = sum(len(games) for games in results.values())
    llm_wins = 0
    llm_games = 0

    for matchup, games in results.items():
        if not games:
            continue

        llm_win_count = sum(1 for g in games if is_llm_winner(g))
        win_rate = llm_win_count / len(games)
        avg_reward = statistics.mean(g.llm_reward for g in games)
        avg_steps = statistics.mean(g.steps for g in games)
        avg_challenges = statistics.mean(g.challenges for g in games)

        print(f"\n{matchup}:")
        print(f"  局数: {len(games)}")
        print(f"  LLM胜率: {win_rate*100:.1f}% ({llm_win_count}/{len(games)})")
        print(f"  平均奖励: {avg_reward:+.2f}")
        print(f"  平均步数: {avg_steps:.1f}")
        print(f"  平均挑战次数: {avg_challenges:.1f}")

        llm_wins += llm_win_count
        llm_games += len(games)

    print(f"\n总体统计:")
    print(f"  总局数: {total_games}")
    print(f"  LLM总胜率: {llm_wins/llm_games*100:.1f}% ({llm_wins}/{llm_games})")


def main():
    parser = argparse.ArgumentParser(description="LLM代理两人对战评估")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["random", "conservative", "aggressive", "heuristic"],
        choices=["random", "conservative", "aggressive", "heuristic", "basic", "probabilistic"],
        help="对手代理类型"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="每个对手的游戏局数"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API密钥（如果需要）"
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="启用思考模式（Thinking Mode）"
    )

    args = parser.parse_args()

    if args.api_key:
        import os
        os.environ["OPENAI_API_KEY"] = args.api_key

    # 创建LLM代理（两人模式）
    print("初始化LLM代理...")
    llm_agent = OptimizedLLMAgent("llm_player", 2, enable_thinking=args.enable_thinking)

    # 运行锦标赛
    results = run_tournament(
        llm_agent=llm_agent,
        opponent_types=args.opponents,
        num_games=args.num_games,
        enable_thinking=args.enable_thinking
    )

    # 分析结果
    analyze_results(results)


if __name__ == "__main__":
    main()
