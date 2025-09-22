import numpy as np
from typing import Dict, Optional, List

from rl_specialized.action_spaces.player_specific import PlayerSpecificActionSpace


class SpecializedAgent:
    """专用模型智能体（骨架）

    - 使用专用动作空间（计数从 n+1 开始）
    - 目前未依赖深度学习库，提供最小可运行接口
    - 选择策略：从合法动作中均匀采样（可替换为策略网络）
    """

    def __init__(self, num_players: int, dice_per_player: int = 5):
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.action_space = PlayerSpecificActionSpace(num_players, dice_per_player)

    def select_action_id(self, observation: Dict) -> int:
        """基于观察选择动作ID（当前为随机合法动作）"""
        mask = self.action_space.get_action_mask(observation)
        valid_ids = np.flatnonzero(mask)
        if len(valid_ids) == 0:
            # 极端情况：无合法动作，回退到挑战（如果可用）或动作0
            return 0
        return int(np.random.choice(valid_ids))

    def id_to_action(self, action_id: int):
        """动作ID -> 动作对象（Guess 或 Challenge）"""
        return self.action_space.id_to_action(action_id)

    def action_to_id(self, action) -> int:
        """动作对象 -> 动作ID"""
        return self.action_space.action_to_id(action)

    def action_space_size(self) -> int:
        """返回离散动作空间大小"""
        return self.action_space.get_action_space_size()

