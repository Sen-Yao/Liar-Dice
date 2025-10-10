"""
evaluator.py - DQN模型性能评估器

功能：
- 对训练好的DQN模型进行全面评估
- 与多种baseline agents对战
- 生成详细的统计报告和可视化结果

使用方法：
    python evaluator.py --model-path models/dqn_model.pth --num-games 1000
或：
    python evaluator.py --model-dir models/<exp_tag> --num-games 1000 --json
"""

import argparse
import time
from pathlib import Path
from typing import Dict
from collections import defaultdict

import torch
import json

from env import LiarDiceEnv
from agents.DQN_agent import DQNAgent
from agents.baseline_agents import RandomAgent, ConservativeAgent, AggressiveAgent, OptimizedLLMAgent
from agents.heuristic_agent import HeuristicRuleAgent


class DummyArgs:
    """用于DQNAgent的虚拟参数对象"""
    def __init__(self):
        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.batch_size = 64
        self.buffer_size = 50000
        self.target_update_freq = 1000


class ModelEvaluator:
    """
    DQN模型评估器

    评估指标：
    - 胜率（vs各种baseline）
    - 平均回合数
    - 动作分布（猜测率、挑战率）
    - 决策质量（非法动作率、fallback率）
    """

    def __init__(
        self,
        model_path: str,
        num_players: int = 2,
        device: str = 'cpu',
        history_len: int = 0
    ):
        """
        初始化评估器

        参数：
            model_path: DQN模型文件路径
            num_players: 游戏玩家数量
            device: 计算设备（'cpu' 或 'cuda'）
        """
        self.model_path = Path(model_path)
        self.num_players = num_players
        self.device = device

        # 加载DQN模型
        print(f"正在加载模型: {self.model_path}")

        # 创建虚拟args对象
        dummy_args = DummyArgs()

        self.dqn_agent = DQNAgent(
            agent_id='dqn_player',
            num_players=num_players,
            args=dummy_args
        )

        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            self.dqn_agent.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.dqn_agent.q_network.eval()  # 设置为评估模式
            print(f"✓ 模型加载成功")

            # 显示模型信息
            if 'episode' in checkpoint:
                print(f"  训练轮次: {checkpoint['episode']}")
            if 'epsilon' in checkpoint:
                print(f"  探索率: {checkpoint['epsilon']:.4f}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 创建环境
        self.env = LiarDiceEnv(num_players=num_players, history_len=(history_len if history_len > 0 else None))

        # 评估结果存储
        self.results = {}

    def evaluate_against_opponent(
        self,
        opponent_name: str,
        opponent_agent,
        num_games: int = 1000,
        verbose: bool = True
    ) -> Dict:
        """
        评估DQN模型对抗特定对手的表现

        参数：
            opponent_name: 对手名称
            opponent_agent: 对手智能体实例
            num_games: 对局数量
            verbose: 是否显示详细信息

        返回：
            Dict: 包含胜率、平均回合数等统计信息
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"评估 DQN vs {opponent_name}")
            print(f"对局数量: {num_games}")
            print(f"{'='*60}\n")

        # 统计变量
        dqn_wins = 0
        total_rounds = 0
        dqn_actions = defaultdict(int)
        opponent_actions = defaultdict(int)
        game_lengths = []

        start_time = time.time()

        # 运行多局游戏
        for game_idx in range(num_games):
            # 创建agents字典
            agents = {
                'player_0': self.dqn_agent,
                'player_1': opponent_agent
            }

            # 重置环境
            self.env.reset()
            round_count = 0

            # 运行一局游戏
            for agent_id in self.env.agent_iter(max_iter=200):
                observation = self.env.observe(agent_id)

                # 检查游戏是否结束
                if self.env.terminations[agent_id] or self.env.truncations[agent_id]:
                    action = None
                else:
                    agent = agents[agent_id]

                    # 获取动作（DQN在评估模式下自动使用贪婪策略）
                    action = agent.get_action(observation)
                    action_type = type(action).__name__

                    if isinstance(agent, DQNAgent):
                        dqn_actions[action_type] += 1
                    else:
                        opponent_actions[action_type] += 1

                    round_count += 1

                # 执行动作
                self.env.step(action)

                # 检查所有玩家是否终止
                if all(self.env.terminations.values()):
                    break

            # 统计结果（检查谁赢了）
            # 在Liar's Dice中，rewards为-1表示输，0表示赢或平局
            dqn_reward = self.env.rewards.get('player_0', 0)
            if dqn_reward >= 0:  # DQN没有输
                dqn_wins += 1

            total_rounds += round_count
            game_lengths.append(round_count)

            # 显示进度
            progress_interval = 1 
            if verbose and (game_idx + 1) % progress_interval == 0:
                current_winrate = dqn_wins / (game_idx + 1)
                print(f"进度: {game_idx + 1}/{num_games} | "
                      f"当前胜率: {current_winrate:.1%}")

        elapsed_time = time.time() - start_time

        # 计算统计数据
        winrate = dqn_wins / num_games
        avg_game_length = total_rounds / num_games

        total_dqn_actions = sum(dqn_actions.values())
        total_opponent_actions = sum(opponent_actions.values())

        dqn_challenge_rate = dqn_actions['Challenge'] / total_dqn_actions if total_dqn_actions > 0 else 0
        opponent_challenge_rate = opponent_actions['Challenge'] / total_opponent_actions if total_opponent_actions > 0 else 0

        # 构建结果字典
        result = {
            'opponent': opponent_name,
            'num_games': num_games,
            'dqn_wins': dqn_wins,
            'opponent_wins': num_games - dqn_wins,
            'winrate': winrate,
            'avg_game_length': avg_game_length,
            'total_rounds': total_rounds,
            'dqn_actions': dict(dqn_actions),
            'opponent_actions': dict(opponent_actions),
            'dqn_challenge_rate': dqn_challenge_rate,
            'opponent_challenge_rate': opponent_challenge_rate,
            'elapsed_time': elapsed_time,
            'game_lengths': game_lengths
        }

        # 显示结果
        if verbose:
            self._print_match_result(result)

        return result

    def _print_match_result(self, result: Dict):
        """打印单场对局结果"""
        print(f"\n{'-'*60}")
        print(f"对局结果: DQN vs {result['opponent']}")
        print(f"{'-'*60}")
        print(f"总对局数:   {result['num_games']}")
        print(f"DQN胜场:    {result['dqn_wins']} ({result['winrate']:.1%})")
        print(f"对手胜场:   {result['opponent_wins']} ({1-result['winrate']:.1%})")
        print(f"平均回合数: {result['avg_game_length']:.1f}")
        print(f"总耗时:     {result['elapsed_time']:.2f}秒")
        print(f"\n动作统计:")
        print(f"  DQN挑战率:  {result['dqn_challenge_rate']:.1%}")
        print(f"  对手挑战率: {result['opponent_challenge_rate']:.1%}")

    def run_full_evaluation(
        self,
        num_games: int = 1000,
        include_heuristic: bool = True,
        include_llm: bool = False
    ) -> Dict:
        """
        运行完整评估（对抗所有baseline）

        参数：
            num_games: 每个对手的对局数量
            include_heuristic: 是否包含启发式对手
            include_llm: 是否包含LLM对手（需要API密钥）

        返回：
            Dict: 所有评估结果
        """
        total_opponents = 4 + (1 if include_llm else 0)

        print(f"\n{'#'*60}")
        print(f"# DQN模型完整评估")
        print(f"# 模型: {self.model_path.name}")
        print(f"# 玩家数: {self.num_players}")
        print(f"# 每个对手对局数: {num_games}")
        print(f"# 对手数量: {total_opponents}")
        print(f"{'#'*60}\n")

        all_results = {}

        # 1. vs RandomAgent
        print(f"\n[1/{total_opponents}] 评估 vs RandomAgent")
        random_agent = RandomAgent('opponent', num_players=self.num_players, seed=42)
        all_results['Random'] = self.evaluate_against_opponent(
            'RandomAgent',
            random_agent,
            num_games=num_games
        )

        # 2. vs ConservativeAgent
        print(f"\n[2/{total_opponents}] 评估 vs ConservativeAgent")
        conservative_agent = ConservativeAgent('opponent', num_players=self.num_players)
        all_results['Conservative'] = self.evaluate_against_opponent(
            'ConservativeAgent',
            conservative_agent,
            num_games=num_games
        )

        # 3. vs AggressiveAgent
        print(f"\n[3/{total_opponents}] 评估 vs AggressiveAgent")
        aggressive_agent = AggressiveAgent('opponent', num_players=self.num_players)
        all_results['Aggressive'] = self.evaluate_against_opponent(
            'AggressiveAgent',
            aggressive_agent,
            num_games=num_games
        )

        # 4. vs HeuristicRuleAgent
        if include_heuristic:
            print(f"\n[4/{total_opponents}] 评估 vs HeuristicRuleAgent")
            heuristic_agent = HeuristicRuleAgent('opponent', num_players=self.num_players)
            all_results['Heuristic'] = self.evaluate_against_opponent(
                'HeuristicRuleAgent',
                heuristic_agent,
                num_games=num_games
            )

        # 5. vs OptimizedLLMAgent（可选，需要API密钥）
        if include_llm:
            print(f"\n[5/{total_opponents}] 评估 vs OptimizedLLMAgent")
            try:
                import os
                has_key = bool(os.getenv('DASHSCOPE_API_KEY') and os.getenv('DASHSCOPE_API_KEY') != 'your-api-key-here')
                print(f"API Key 状态: {'已设置' if has_key else '未设置'}")

                print("创建 OptimizedLLMAgent...")
                llm_agent = OptimizedLLMAgent('opponent', num_players=self.num_players, use_api=True)
                print(f"LLM Agent 创建完成: has_api={llm_agent.has_api}, use_api={llm_agent.use_api}")
                
                llm_games = int(num_games / 10)
                print(f"开始 LLM 对局评估 ({llm_games} 局)...")
                all_results['LLM'] = self.evaluate_against_opponent(
                    'OptimizedLLMAgent',
                    llm_agent,
                    num_games=llm_games
                )
            except Exception as e:
                print(f"⚠ LLM对手评估失败: {e}")
                print("  跳过LLM对手评估")

        # 保存结果
        self.results = all_results

        # 打印总结报告
        self._print_summary_report()

        return all_results

    def _print_summary_report(self):
        """打印总结报告"""
        print(f"\n{'='*60}")
        print(f"综合评估报告")
        print(f"{'='*60}\n")

        # 表头
        print(f"{'对手':<20} {'对局数':>8} {'胜率':>10} {'平均回合':>10} {'DQN挑战率':>12}")
        print(f"{'-'*60}")

        # 每个对手的结果
        total_games = 0
        total_wins = 0

        for opponent_name, result in self.results.items():
            print(f"{opponent_name:<20} "
                  f"{result['num_games']:>8} "
                  f"{result['winrate']:>9.1%} "
                  f"{result['avg_game_length']:>10.1f} "
                  f"{result['dqn_challenge_rate']:>11.1%}")

            total_games += result['num_games']
            total_wins += result['dqn_wins']

        print(f"{'-'*60}")

        # 总计
        overall_winrate = total_wins / total_games if total_games > 0 else 0
        print(f"{'总计':<20} "
              f"{total_games:>8} "
              f"{overall_winrate:>9.1%}")

        print(f"\n{'='*60}")

        # 关键发现
        print(f"\n关键发现:")

        # 找出最强和最弱对手
        if self.results:
            sorted_results = sorted(self.results.items(), key=lambda x: x[1]['winrate'])

            weakest_opponent = sorted_results[0]
            strongest_opponent = sorted_results[-1]

            print(f"  • 最难对手: {weakest_opponent[0]} (胜率 {weakest_opponent[1]['winrate']:.1%})")
            print(f"  • 最易对手: {strongest_opponent[0]} (胜率 {strongest_opponent[1]['winrate']:.1%})")
            print(f"  • 综合胜率: {overall_winrate:.1%}")

            # 挑战率分析
            avg_challenge_rate = sum(r['dqn_challenge_rate'] for r in self.results.values()) / len(self.results)
            print(f"  • DQN平均挑战率: {avg_challenge_rate:.1%}")

    def save_results(self, output_path: str = None):
        """
        保存评估结果到文件

        参数：
            output_path: 输出文件路径（默认为evaluation_results.txt）
        """
        if output_path is None:
            output_path = f"evaluation_results_{self.model_path.stem}.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"DQN模型评估报告\n")
            f.write(f"{'='*60}\n")
            f.write(f"模型: {self.model_path}\n")
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")

            # 详细结果
            for opponent_name, result in self.results.items():
                f.write(f"\n对手: {opponent_name}\n")
                f.write(f"{'-'*40}\n")
                f.write(f"对局数: {result['num_games']}\n")
                f.write(f"DQN胜场: {result['dqn_wins']} ({result['winrate']:.1%})\n")
                f.write(f"对手胜场: {result['opponent_wins']} ({1-result['winrate']:.1%})\n")
                f.write(f"平均回合数: {result['avg_game_length']:.2f}\n")
                f.write(f"DQN挑战率: {result['dqn_challenge_rate']:.1%}\n")
                f.write(f"对手挑战率: {result['opponent_challenge_rate']:.1%}\n")

        print(f"\n✓ 评估结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DQN模型评估器')
    parser.add_argument('--model-path', type=str, default='',
                        help='DQN模型文件路径（与 --model-dir 二选一）')
    parser.add_argument('--model-dir', type=str, default='',
                        help='模型目录（包含 model.pth 与 config.json）')
    parser.add_argument('--num-games', type=int, default=1000,
                        help='每个对手的对局数量')
    parser.add_argument('--num-players', type=int, default=2,
                        help='游戏玩家数量')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='计算设备')
    parser.add_argument('--no-heuristic', action='store_true',
                        help='不包含启发式对手评估')
    parser.add_argument('--include-llm', action='store_true', default=False,
                        help='包含LLM对手评估（需要API密钥）')
    parser.add_argument('--save-results', action='store_true',
                        help='保存评估结果到文件')
    parser.add_argument('--json', action='store_true', help='以 JSON 形式输出核心指标（适合批处理）')

    args = parser.parse_args()

    # 解析 model-dir
    history_len = 0
    model_path = args.model_path
    if args.model_dir:
        mdir = Path(args.model_dir)
        cfg_path = mdir / 'config.json'
        model_path = str(mdir / 'model.pth')
        if cfg_path.exists():
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                # 优先从配置复现玩家数与历史长度
                if 'num_players' in cfg:
                    args.num_players = int(cfg['num_players'])
                if 'history_len' in cfg:
                    history_len = int(cfg['history_len'])
            except Exception:
                pass

    # 创建评估器
    evaluator = ModelEvaluator(
        model_path=model_path or 'models/dqn_model.pth',
        num_players=args.num_players,
        device=args.device,
        history_len=history_len
    )

    # 运行评估
    if args.json:
        # 输出除 LLM 外所有内置对手的评估结果（JSON 列表）
        results = evaluator.run_full_evaluation(
            num_games=args.num_games,
            include_heuristic=not args.no_heuristic,
            include_llm=False
        )
        payload = []
        for name, res in results.items():
            payload.append({
                'opponent': name,
                'num_games': res['num_games'],
                'winrate': res['winrate'],
                'avg_game_length': res['avg_game_length']
            })
        print(json.dumps(payload, ensure_ascii=False))
        return
    else:
        results = evaluator.run_full_evaluation(
            num_games=args.num_games,
            include_heuristic=not args.no_heuristic,
            include_llm=args.include_llm
        )

    # 保存结果
    if args.save_results:
        evaluator.save_results()

    print("\n评估完成！")


if __name__ == '__main__':
    main()
