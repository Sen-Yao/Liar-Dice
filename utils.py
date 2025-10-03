from typing import List, Tuple, Dict, Optional
from env import LiarDiceEnv, Guess, Challenge, Action

def is_strictly_greater(new_guess, old_guess) -> bool:
    """实现猜测比较逻辑（与env.py._is_strictly_greater保持完全一致）"""
    if new_guess.mode == old_guess.mode:
        # 同模式比较
        if new_guess.mode == '飞':
            # 飞模式：新猜测必须更大（个数多或个数相同时数字大）
            if new_guess.count > old_guess.count:
                return True
            if new_guess.count == old_guess.count and new_guess.face > old_guess.face:
                return True
            return False
        else:  # '斋'模式
            # 斋模式：新个数 > 旧个数，或个数相同时按斋模式排序（2<3<4<5<6<1）
            if new_guess.count > old_guess.count:
                return True
            if new_guess.count == old_guess.count:
                # 斋模式数字大小：2 < 3 < 4 < 5 < 6 < 1
                zhai_order = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 1: 5}
                return zhai_order[new_guess.face] > zhai_order[old_guess.face]
            return False

    # 跨模式比较：仅比较数量，不比较面值
    if new_guess.mode == '斋' and old_guess.mode == '飞':
        # 飞 → 斋：新个数 ≥ 旧个数/2（向上取整）
        return new_guess.count >= (old_guess.count + 1) // 2
    else:  # new_guess.mode == '飞' and old_guess.mode == '斋'
        # 斋 → 飞：新个数 ≥ 旧个数×2
        return new_guess.count >= old_guess.count * 2

def get_legal_actions(observation: Dict, num_players: int) -> List[Action]:
    """Get all legal actions for current observation"""
    legal_actions = []
    total_dice = 5 * num_players
    last_guess = observation['last_guess']

    # Always add challenge if there's a previous guess
    if last_guess is not None:
        legal_actions.append(Challenge())

    # Add legal guesses
    for mode in ['飞', '斋']:
        for count in range(1, total_dice + 1):
            for face in range(1, 7):
                if mode == '飞' and face == 1:
                    continue  # Flying mode can't guess face 1

                guess = Guess(mode=mode, count=count, face=face)

                if last_guess is None:
                    if count > num_players:  # First guess must be greater than num_players
                        legal_actions.append(guess)
                elif is_strictly_greater(guess, last_guess):
                    legal_actions.append(guess)

    return legal_actions