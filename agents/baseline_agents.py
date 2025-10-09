"""
baseline_agents.py - Baseline Agent实现

包含用于性能对比的baseline agents：
- RandomAgent: 从所有合法动作中均匀随机选择

这些agents用于baseline实验，提供性能下界和对比基准。
"""

import random
from typing import Dict, Optional
from env import Action
from utils import get_legal_actions


class RandomAgent:
    """
    随机baseline agent：从所有合法动作中均匀随机选择

    特性：
    - **100%合法性保证**：使用`utils.get_legal_actions()`预先计算所有合法动作
    - **均匀随机选择**：对所有合法动作等概率采样
    - **可重复实验**：支持设置随机种子

    用途：
    - 作为最弱baseline，评估其他agent的最低性能边界
    - 验证环境的公平性（随机agent的胜率应接近1/n）
    - 检测对手agent是否存在可被随机策略利用的弱点

    示例：
        ```python
        agent = RandomAgent(agent_id="random_0", num_players=2, seed=42)
        action = agent.get_action(observation)
        ```
    """

    def __init__(
        self,
        agent_id: str,
        num_players: int,
        seed: Optional[int] = None
    ):
        """
        初始化随机agent

        参数：
            agent_id: agent标识符
            num_players: 游戏玩家数量（用于计算合法动作）
            seed: 随机种子（可选，用于可重复实验）
        """
        self.agent_id = agent_id
        self.num_players = num_players

        # 创建实例级随机数生成器（避免影响全局random）
        self.rng = random.Random(seed)

        # 统计信息（可选，用于分析）
        self.action_count = 0
        self.guess_count = 0
        self.challenge_count = 0

    def get_action(self, observation: Dict) -> Action:
        """
        从所有合法动作中随机选择一个

        流程：
        1. 调用`utils.get_legal_actions()`获取所有合法动作
        2. 使用`random.choice()`均匀随机选择一个动作
        3. 更新统计信息
        4. 返回选中的动作

        参数：
            observation: 游戏观察状态（Dict类型）
                - 'my_dice_counts': 自己的骰子分布
                - 'last_guess': 上一个猜测（None表示首轮）
                - 'total_dice_on_table': 场上总骰子数
                - 其他字段...

        返回：
            Action: 随机选择的合法动作（Guess或Challenge）

        保证：
            返回的动作100%合法（因为从预先计算的合法动作集中选择）
        """
        # 获取所有合法动作
        legal_actions = get_legal_actions(observation, self.num_players)

        # 确保至少有一个合法动作（理论上总是满足）
        if not legal_actions:
            raise RuntimeError(
                f"[RandomAgent] 没有合法动作可选！observation={observation}"
            )

        # 均匀随机选择（使用实例级随机数生成器）
        action = self.rng.choice(legal_actions)

        # 更新统计信息
        self.action_count += 1
        if action.__class__.__name__ == 'Challenge':
            self.challenge_count += 1
        else:  # Guess
            self.guess_count += 1

        return action

    def get_stats(self) -> Dict:
        """
        获取统计信息（用于分析agent行为）

        返回：
            Dict: 包含动作统计的字典
                - action_count: 总动作次数
                - guess_count: 猜测次数
                - challenge_count: 挑战次数
                - guess_rate: 猜测比例
        """
        stats = {
            "action_count": self.action_count,
            "guess_count": self.guess_count,
            "challenge_count": self.challenge_count,
        }

        if self.action_count > 0:
            stats["guess_rate"] = self.guess_count / self.action_count
        else:
            stats["guess_rate"] = 0.0

        return stats

    def reset_stats(self):
        """重置统计信息（用于多局游戏）"""
        self.action_count = 0
        self.guess_count = 0
        self.challenge_count = 0
