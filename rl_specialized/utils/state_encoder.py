import numpy as np
from typing import Dict, Optional, Tuple, List

from env import Guess, AgentState


class StateEncoder:
    """状态编码器 - 将复杂的游戏状态转换为神经网络可处理的向量

    扩展：支持编码最近 N 步历史（默认 3 步），每步 4 个特征：
    - mode(飞/斋)、count/总骰子、face/6、player_idx/num_players
    """

    def __init__(self, num_players: int, dice_per_player: int = 5, history_length: int = 3):
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.total_dice = num_players * dice_per_player
        self.history_length = int(max(0, history_length))

        # 编码方案：6维手牌 + 玩家信息 + 上次猜测 + 游戏状态 + 历史(N×4)
        self.dice_features = 6               # 每种点数的数量 [1,2,3,4,5,6]
        self.player_features = num_players   # 每个玩家的罚分
        self.guess_features = 4              # 上次猜测 [mode, count, face, is_valid]
        self.game_features = 3               # 游戏状态 [total_dice, current_player, is_my_turn]
        self.history_features = 4 * self.history_length

        self.total_features = (self.dice_features + self.player_features +
                               self.guess_features + self.game_features +
                               self.history_features)

    def _encode_history(self, observation: Dict) -> List[float]:
        """编码最近 N 步历史：(player_idx, mode, count, face)

        优先使用类型安全的 `game_round_history_encoded`；若不存在则回退到
        `game_round_history`（包含 Guess 对象）。
        """
        if self.history_length <= 0:
            return []

        feats: List[float] = []
        encoded = observation.get("game_round_history_encoded")
        if isinstance(encoded, list):
            recent = encoded[-self.history_length:]
            for item in recent:
                mode = float(item.get("mode", 0))  # 0:飞, 1:斋
                count = float(item.get("count", 0)) / max(1, self.total_dice)
                face = float(item.get("face", 1)) / 6.0
                pid = float(item.get("player_idx", 0)) / max(1, self.num_players)
                feats.extend([mode, count, face, pid])
        else:
            history = observation.get("game_round_history", []) or []
            recent = history[-self.history_length:]
            for (player_idx, guess) in recent:
                # 兼容 Guess 或 dict
                if isinstance(guess, dict):
                    mode_val = guess.get("mode", 0)
                    # 旧版若存中文，需要统一到 0/1，这里做保守转换
                    if isinstance(mode_val, str):
                        mode = 0.0 if mode_val == '飞' else 1.0
                    else:
                        mode = float(mode_val)
                    count = float(guess.get("count", 0)) / max(1, self.total_dice)
                    face = float(guess.get("face", 1)) / 6.0
                else:
                    # Guess dataclass
                    mode = 0.0 if getattr(guess, 'mode', '飞') == '飞' else 1.0
                    count = float(getattr(guess, 'count', 0)) / max(1, self.total_dice)
                    face = float(getattr(guess, 'face', 1)) / 6.0
                pid = float(player_idx) / max(1, self.num_players)
                feats.extend([mode, count, face, pid])

        # 填充到固定长度（填充在前，保证最近历史在固定位置）
        pad = self.history_features - len(feats)
        if pad > 0:
            feats = [0.0] * pad + feats
        return feats

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
            # 兼容 dict 或 Guess dataclass
            if isinstance(last_guess, dict):
                mode_val = last_guess.get("mode")
                count_val = last_guess.get("count", 0)
                face_val = last_guess.get("face", 1)
            else:
                # 视为 Guess 对象
                mode_val = getattr(last_guess, "mode", '飞')
                count_val = getattr(last_guess, "count", 0)
                face_val = getattr(last_guess, "face", 1)

            # 模式编码：飞=0, 斋=1
            mode_encoding = 0 if mode_val == '飞' else 1
            features.extend([
                mode_encoding,
                float(count_val) / max(1, self.total_dice),  # 归一化
                float(face_val) / 6.0,                        # 归一化
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

        # 5. 历史特征 (4 * history_length 维)
        if self.history_length > 0:
            features.extend(self._encode_history(observation))

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

        # 历史特征
        for i in range(self.history_length):
            names.extend([
                f"hist_{i}_mode",
                f"hist_{i}_count_ratio",
                f"hist_{i}_face_ratio",
                f"hist_{i}_player_ratio",
            ])

        return names

    def encode_batch(self, observations: list) -> np.ndarray:
        """批量编码观察"""
        return np.array([self.encode_observation(obs) for obs in observations])


def create_state_encoder(num_players: int, dice_per_player: int = 5, history_length: int = 3) -> StateEncoder:
    """创建状态编码器的工厂函数

    默认编码最近 3 步历史。如需保持旧模型兼容，可将 history_length=0。
    """
    return StateEncoder(num_players, dice_per_player, history_length)
