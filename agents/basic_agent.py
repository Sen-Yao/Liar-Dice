import numpy as np
from typing import Dict

from env import Guess, Challenge, Action

# TODO: 这是一个非常简单的代理，需要进一步优化
# 策略是保守地加注。
# 在猜测数量过高时发起挑战。
# 优先增加点数，保持数量不变。
# 如果点数已经是6，就增加数量，并将点数重置为最低。

class BasicRuleAgent:
    """
    一个基于简单启发式规则的基础 Agent，用于测试环境。
    - 策略是保守地加注。
    - 在猜测数量过高时发起挑战。
    """
    def __init__(self, agent_id: str, num_players: int):
        self.agent_id = agent_id
        self.num_players = num_players
        # 一个非常简单的挑战阈值
        self.challenge_threshold = self.num_players + 3

    def get_action(self, observation: Dict) -> Action:

        if observation["last_guess"] is None:
            return Guess(mode='飞', count=self.num_players + 1, face=4)

        # 如果不是第一个回合，我们可以选择猜测或挑战
        
        # last_guess 已经是 Guess 对象，直接使用
        last_guess = observation["last_guess"]
        
        # 决定是否挑战
        # 如果猜测的数量超过了我们的保守阈值，就发起挑战
        if last_guess.count > self.challenge_threshold:
            return Challenge()

        # 如果不挑战，就必须出一个更高的猜测
        current_mode = last_guess.mode
        current_count = last_guess.count
        current_face = last_guess.face

        # 策略：优先增加点数，保持数量不变
        if current_face < 6:
            # 检查我们是否可以猜 "斋" 模式下的 1
            if current_mode == '斋' and current_face == 1:
                return Guess(mode=current_mode, count=current_count, face=2)
            
            return Guess(mode=current_mode, count=current_count, face=current_face + 1)
        else:
            # 如果点数已经是6，就增加数量，并将点数重置为最低
            # （在“飞”模式下，最低是2；在“斋”模式下，最低是1）
            new_face = 1 if current_mode == '斋' else 2
            return Guess(mode=current_mode, count=current_count + 1, face=new_face)