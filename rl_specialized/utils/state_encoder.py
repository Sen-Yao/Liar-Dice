import numpy as np
from typing import Dict, Optional, Tuple

from env import Guess, AgentState


class StateEncoder:
    """状态编码器 - 将复杂的游戏状态转换为神经网络可处理的向量"""

    def __init__(self, num_players: int, dice_per_player: int = 5):
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.total_dice = num_players * dice_per_player

        # 编码方案：6维手牌 + 玩家信息 + 上次猜测 + 游戏状态
        self.dice_features = 6          # 每种点数的数量 [1,2,3,4,5,6]
        self.player_features = num_players  # 每个玩家的罚分
        self.guess_features = 4         # 上次猜测 [mode, count, face, is_valid]
        self.game_features = 3          # 游戏状态 [total_dice, current_player, is_my_turn]

        self.total_features = (self.dice_features + self.player_features +
                              self.guess_features + self.game_features)

    def encode_observation(self, observation: Dict) -> np.ndarray:
        """将观察编码为固定长度的向量

        Args:
            observation: 来自环境的观察字典

        Returns:
            编码后的状态向量
        """
        features = []

        # 1. 手牌特征 (6维)
        dice_counts = observation["my_dice_counts"]
        features.extend(dice_counts)

        # 2. 玩家罚分特征 (num_players维)
        player_penalties = observation["player_penalties"]
        features.extend(player_penalties)

        # 3. 上次猜测特征 (4维)
        last_guess = observation.get("last_guess")
        if last_guess is not None:
            # 模式编码：飞=0, 斋=1
            mode_encoding = 0 if last_guess["mode"] == '飞' else 1
            features.extend([
                mode_encoding,
                last_guess["count"] / self.total_dice,  # 归一化
                last_guess["face"] / 6.0,               # 归一化
                1.0  # 存在有效猜测
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])  # 无猜测

        # 4. 游戏状态特征 (3维)
        features.extend([
            observation["total_dice_on_table"] / self.total_dice,  # 归一化
            observation["current_player_id_idx"] / self.num_players,  # 归一化
            1.0 if observation["is_my_turn"] else 0.0
        ])

        return np.array(features, dtype=np.float32)

    def get_feature_size(self) -> int:
        """获取编码后特征向量的大小"""
        return self.total_features

    def get_feature_names(self) -> list:
        """获取特征名称列表，用于调试和可视化"""
        names = []

        # 手牌特征
        for i in range(1, 7):
            names.append(f"dice_count_{i}")

        # 玩家罚分
        for i in range(self.num_players):
            names.append(f"player_{i}_penalty")

        # 上次猜测
        names.extend(["last_guess_mode", "last_guess_count", "last_guess_face", "has_last_guess"])

        # 游戏状态
        names.extend(["total_dice_ratio", "current_player_ratio", "is_my_turn"])

        return names

    def encode_batch(self, observations: list) -> np.ndarray:
        """批量编码观察"""
        return np.array([self.encode_observation(obs) for obs in observations])


def create_state_encoder(num_players: int, dice_per_player: int = 5) -> StateEncoder:
    """创建状态编码器的工厂函数"""
    return StateEncoder(num_players, dice_per_player)