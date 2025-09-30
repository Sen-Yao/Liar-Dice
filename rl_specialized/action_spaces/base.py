import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

from env import Guess, Challenge, Action, GuessMode, DiceFace


class BaseActionSpace(ABC):
    """动作空间基础类"""

    def __init__(self, num_players: int, dice_per_player: int = 5):
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        # 最大可见骰子总数
        self.max_dice_count = num_players * dice_per_player
        # 专用动作空间的最小可叫个数：n+1（与首轮规则一致）
        # 注意：这会缩小离散动作空间大小，从而得到更小的搜索空间
        self.min_count = num_players + 1

        # 动作ID映射：0为Challenge，其余为Guess
        self.challenge_action_id = 0
        self._build_action_mapping()

    @abstractmethod
    def _build_action_mapping(self):
        """构建动作ID到动作对象的映射"""
        pass

    @abstractmethod
    def action_to_id(self, action: Action) -> int:
        """将动作对象转换为动作ID"""
        pass

    @abstractmethod
    def id_to_action(self, action_id: int) -> Action:
        """将动作ID转换为动作对象"""
        pass

    @abstractmethod
    def get_action_mask(self, observation: Dict) -> np.ndarray:
        """获取合法动作掩码"""
        pass

    @abstractmethod
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        pass

    def _is_legal_guess(self, guess: Guess, last_guess: Optional[Guess]) -> bool:
        """检查猜测是否合法"""
        # 首轮限制：第一个玩家猜测个数 > 玩家人数
        if last_guess is None:
            return guess.count > self.num_players

        # 猜测必须比上一个更大
        return self._is_strictly_greater(guess, last_guess)

    def _is_strictly_greater(self, new_guess: Guess, old_guess: Guess) -> bool:
        """判断新猜测是否严格大于旧猜测"""
        if new_guess.mode == old_guess.mode:
            # 同模式比较
            if new_guess.mode == '飞':
                # 飞模式：新猜测必须更大（个数多或数字大）
                if new_guess.count > old_guess.count:
                    return True
                if new_guess.count == old_guess.count and new_guess.face > old_guess.face:
                    return True
                return False
            else:  # '斋'模式
                # 斋模式：新个数 > 旧个数，或个数相同时面值按斋模式排序
                if new_guess.count > old_guess.count:
                    return True
                if new_guess.count == old_guess.count:
                    # 斋模式数字大小：2 < 3 < 4 < 5 < 6 < 1
                    zhai_order = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 1: 5}
                    return zhai_order[new_guess.face] > zhai_order[old_guess.face]
                return False
        else:
            # 跨模式比较
            if new_guess.mode == '斋' and old_guess.mode == '飞':
                # 飞 → 斋：新个数 ≥ 旧个数/2（向上取整），不比较面值
                return new_guess.count >= (old_guess.count + 1) // 2
            else:  # new_guess.mode == '飞' and old_guess.mode == '斋'
                # 斋 → 飞：新个数 ≥ 旧个数×2，不比较面值
                return new_guess.count >= old_guess.count * 2
