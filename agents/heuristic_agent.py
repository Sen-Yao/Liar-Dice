import numpy as np
from typing import Dict

from env import Guess, Challenge, Action
from utils import is_strictly_greater

# TODO: 这是一个非常简单的代理，需要进一步优化
# 策略是保守地加注。
# 在猜测数量过高时发起挑战。
# 优先增加点数，保持数量不变。
# 如果点数已经是6，就增加数量，并将点数重置为最低。

class HeuristicRuleAgent:
    """
    一个基于期望的启发式规则的基础 Agent
    """
    def __init__(self, agent_id: str, num_players: int, confidence_threshold=1):
        self.agent_id = agent_id
        self.num_players = num_players
        # 一个非常简单的挑战阈值
        self.challenge_threshold = self.num_players + 3
        self.confidence_threshold = confidence_threshold

    def get_action(self, observation: Dict) -> Action:
        # observation 是字典，其中包括的元素有 ['my_dice_counts'], ["last_guess"] 和 ['game_round_history']
        max_face = 0
        max_num = 0
        for i in range(1, 6):
            if observation['my_dice_counts'][i] > max_num:
                max_num = observation['my_dice_counts'][i]
                max_face = i+1
        # 若首回合，则直接提出一个相对安全的猜测
        if observation["last_guess"] is None:
            if observation['my_dice_counts'][max_face-1] > observation['my_dice_counts'][0]:
                # 最多的点数甚至比 1 还多，可以考虑喊斋
                expect_dice_counts = self.calculate_expectation(observation=observation, mode='斋')
                if int(expect_dice_counts[max_face-1]) > self.num_players:
                    print("int:", int(expect_dice_counts[max_face-1]))
                    return Guess(mode='斋', count=int(expect_dice_counts[max_face-1]), face=max_face)
                else:
                    # 点数不够多,保守喊飞
                    return Guess(mode='飞', count=self.num_players + 1, face=max_face)
            else:
                expect_dice_counts = self.calculate_expectation(observation=observation, mode='飞')
                if int(expect_dice_counts[max_face-1]) > self.num_players:
                    print("int:", int(expect_dice_counts[max_face-1]))
                    return Guess(mode='飞', count=int(expect_dice_counts[max_face-1]), face=max_face)
                else:
                    # 点数不够多,保守喊飞
                    return Guess(mode='飞', count=self.num_players + 1, face=max_face)

        # 如果不是第一个回合，我们可以选择猜测或挑战
        
        # last_guess 已经是 Guess 对象，直接使用
        last_guess = observation["last_guess"]
        
        # 决定是否挑战
        expect_dice_counts = self.calculate_expectation(observation=observation, mode=last_guess.mode)
        # 如果猜测的数量超过了期望+信心阈值, 就发起挑战
        if last_guess.count > expect_dice_counts[last_guess.face-1] + self.confidence_threshold:
            return Challenge()
        else:
            if observation['my_dice_counts'][max_face-1] > observation['my_dice_counts'][0]:
                # 最多的点数甚至比 1 还多，可以考虑喊斋
                expect_dice_counts = self.calculate_expectation(observation=observation, mode='斋')
                try_num = max_num
                while(try_num < 5 * self.num_players):
                    try_guess = Guess(mode='斋', count=try_num, face=max_face)
                    if is_strictly_greater(try_guess, last_guess):
                        return Guess(mode='斋', count=try_num, face=max_face)
                    else:
                        try_num += 1
            else:
                expect_dice_counts = self.calculate_expectation(observation=observation, mode='飞')
                try_num = max_num
                while(try_num < 5 * self.num_players):
                    try_guess = Guess(mode='飞', count=try_num, face=max_face)
                    if is_strictly_greater(try_guess, last_guess):
                        return Guess(mode='飞', count=try_num, face=max_face)
                    else:
                        try_num += 1

    def calculate_expectation(self,  observation: Dict, mode='飞'):
        # 基于手牌和场上情况来计算期望
        unknown_total_dice_num = (self.num_players - 1) * 5
        expect_unknown_every_face_dice_num = unknown_total_dice_num / 6
        expect_dice_counts = list(observation['my_dice_counts'])
        if mode == '飞':
            expect_dice_counts[0] += expect_unknown_every_face_dice_num
            for i in range(1, 6):
                expect_dice_counts[i] += expect_unknown_every_face_dice_num * 2
        else:
            for i in range(6):
                expect_dice_counts[i] += expect_unknown_every_face_dice_num * 1
        return expect_dice_counts