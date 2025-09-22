import numpy as np
from typing import Dict, List, Tuple, Optional

from .base import BaseActionSpace
from env import Guess, Challenge, Action, GuessMode, DiceFace


class PlayerSpecificActionSpace(BaseActionSpace):
    """专用模型动作空间 - 为特定玩家数量优化"""

    def __init__(self, num_players: int, dice_per_player: int = 5):
        super().__init__(num_players, dice_per_player)

    def _build_action_mapping(self):
        """构建动作ID到动作对象的映射

        动作空间设计：
        - Challenge: action_id = 0
        - 斋模式猜测: action_id = 1 + (count-1)*6 + (face-1)
        - 飞模式猜测: action_id = 1 + max_zhai_actions + (count-1)*6 + (face-1)
        """
        self.action_to_obj = {}
        self.obj_to_action = {}

        action_id = 0

        # Challenge动作
        challenge = Challenge()
        self.action_to_obj[action_id] = challenge
        self.obj_to_action[challenge] = action_id
        action_id += 1

        # 斋模式猜测动作
        self.zhai_start_id = action_id
        for count in range(1, self.max_dice_count + 1):
            for face in range(1, 7):
                guess = Guess(mode='斋', count=count, face=face)
                self.action_to_obj[action_id] = guess
                self.obj_to_action[guess] = action_id
                action_id += 1

        # 飞模式猜测动作
        self.fei_start_id = action_id
        for count in range(1, self.max_dice_count + 1):
            for face in range(1, 7):
                guess = Guess(mode='飞', count=count, face=face)
                self.action_to_obj[action_id] = guess
                self.obj_to_action[guess] = action_id
                action_id += 1

        self.total_actions = action_id

    def action_to_id(self, action: Action) -> int:
        """将动作对象转换为动作ID"""
        if isinstance(action, Challenge):
            return self.challenge_action_id
        elif isinstance(action, Guess):
            return self.obj_to_action[action]
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

    def id_to_action(self, action_id: int) -> Action:
        """将动作ID转换为动作对象"""
        if action_id < 0 or action_id >= self.total_actions:
            raise ValueError(f"Invalid action_id: {action_id}")
        return self.action_to_obj[action_id]

    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return self.total_actions

    def get_action_mask(self, observation: Dict) -> np.ndarray:
        """获取合法动作掩码

        Args:
            observation: 智能体观察，包含last_guess等信息

        Returns:
            合法动作掩码，1表示合法，0表示非法
        """
        mask = np.zeros(self.total_actions, dtype=np.bool_)

        last_guess = observation.get("last_guess")

        # Challenge总是可以执行（如果不是首轮）
        if last_guess is not None:
            mask[self.challenge_action_id] = True

        # 检查所有猜测动作的合法性
        for action_id in range(1, self.total_actions):
            guess = self.action_to_obj[action_id]
            if self._is_legal_guess(guess, last_guess):
                mask[action_id] = True

        return mask

    def get_valid_actions(self, observation: Dict) -> List[int]:
        """获取合法动作ID列表"""
        mask = self.get_action_mask(observation)
        return np.where(mask)[0].tolist()

    def get_action_distribution_info(self) -> Dict:
        """获取动作分布信息，用于调试和分析"""
        return {
            "total_actions": self.total_actions,
            "challenge_actions": 1,
            "zhai_actions": self.max_dice_count * 6,
            "fei_actions": self.max_dice_count * 6,
            "zhai_start_id": self.zhai_start_id,
            "fei_start_id": self.fei_start_id,
            "max_dice_count": self.max_dice_count,
            "num_players": self.num_players
        }


def create_player_specific_action_space(num_players: int, dice_per_player: int = 5) -> PlayerSpecificActionSpace:
    """创建专用模型动作空间的工厂函数"""
    return PlayerSpecificActionSpace(num_players, dice_per_player)


# 预定义常用配置
def get_2_player_action_space() -> PlayerSpecificActionSpace:
    """获取2人游戏专用动作空间"""
    return create_player_specific_action_space(2)

def get_3_player_action_space() -> PlayerSpecificActionSpace:
    """获取3人游戏专用动作空间"""
    return create_player_specific_action_space(3)

def get_4_player_action_space() -> PlayerSpecificActionSpace:
    """获取4人游戏专用动作空间"""
    return create_player_specific_action_space(4)

def get_5_player_action_space() -> PlayerSpecificActionSpace:
    """获取5人游戏专用动作空间"""
    return create_player_specific_action_space(5)

def get_6_player_action_space() -> PlayerSpecificActionSpace:
    """获取6人游戏专用动作空间"""
    return create_player_specific_action_space(6)