import math
import numpy as np
from typing import Dict, Optional, Tuple

from env import Guess, Challenge, Action


# --------------------------- 简单启发式版本（保留兼容） ---------------------------
class BasicRuleAgent:
    """
    最简单的规则 Agent（向后兼容）。
    - 策略：优先抬面值；若面值已到6，则数量+1并重置面值（飞=2，斋=1）。
    - 超过固定阈值（n+3）则挑战。
    """

    def __init__(self, agent_id: str, num_players: int):
        self.agent_id = agent_id
        self.num_players = num_players
        self.challenge_threshold = self.num_players + 3

    def get_action(self, observation: Dict) -> Action:
        if observation["last_guess"] is None:
            return Guess(mode='飞', count=self.num_players + 1, face=4)

        last_guess = observation["last_guess"]
        if last_guess.count > self.challenge_threshold:
            return Challenge()

        current_mode = last_guess.mode
        current_count = last_guess.count
        current_face = last_guess.face

        if current_face < 6:
            if current_mode == '斋' and current_face == 1:
                return Guess(mode=current_mode, count=current_count, face=2)
            return Guess(mode=current_mode, count=current_count, face=current_face + 1)
        else:
            new_face = 1 if current_mode == '斋' else 2
            return Guess(mode=current_mode, count=current_count + 1, face=new_face)


# --------------------------- 概率驱动版本（推荐） ---------------------------
class ProbabilisticBasicAgent:
    """
    概率驱动的规则 Agent：
    - 挑战：若 P(上家叫点为真) < theta_challenge，则挑战
    - 加注：从一小批候选中选择 P(新叫点为真) ≈ target_raise 的“最小抬价”动作
    - 候选：同模式小步抬价、同模式数量+1（配最佳面）、跨模式转换（飞→斋半 | 斋→飞倍）
    - 首行动：从 n+1 起手，选择期望边际 E-count 最大的模式与面值
    """

    def __init__(
        self,
        agent_id: str,
        num_players: int,
        theta_challenge: float = 0.25,
        target_raise: float = 0.60,
        max_extra_raise: int = 2,
    ):
        self.agent_id = agent_id
        self.num_players = num_players
        self.theta_challenge = float(theta_challenge)
        self.target_raise = float(target_raise)
        self.max_extra_raise = int(max_extra_raise)

    # ---- 概率与候选生成工具 ----
    @staticmethod
    def _binom_tail(n: int, p: float, r: int) -> float:
        """二项分布上尾概率 P[X >= r], X~Bin(n,p)，n<=30时直接求和"""
        if r <= 0:
            return 1.0
        if r > n:
            return 0.0
        q = 1.0 - p
        prob = 0.0
        for k in range(r, n + 1):
            prob += math.comb(n, k) * (p ** k) * (q ** (n - k))
        return float(min(max(prob, 0.0), 1.0))

    @staticmethod
    def _own_success(mode: str, face: int, my_counts: Tuple[int, ...]) -> int:
        """己方手牌对该叫点的“已知成功数”"""
        # my_counts 顺序为 [1..6]
        ones = my_counts[0]
        face_cnt = my_counts[face - 1]
        if mode == '飞':
            # 避免生成“飞+face=1”的极端情况（策略层面不选1）
            return face_cnt + ones if face != 1 else ones
        else:
            return face_cnt

    @staticmethod
    def _best_face(mode: str, my_counts: Tuple[int, ...]) -> int:
        """根据己方手牌选择最优面值（飞：face!=1）"""
        if mode == '飞':
            # 选择 2..6 中 maximizer of (count(face)+count(1))
            ones = my_counts[0]
            best_f, best_v = 2, -1
            for f in range(2, 7):
                v = my_counts[f - 1] + ones
                if v > best_v:
                    best_f, best_v = f, v
            return best_f
        else:
            # 选择 1..6 中 count(face) 最大
            best_f, best_v = 1, -1
            for f in range(1, 7):
                v = my_counts[f - 1]
                if v > best_v:
                    best_f, best_v = f, v
            return best_f

    def _prob_true(self, mode: str, count: int, face: int, observation: Dict) -> float:
        """估计叫点为真的概率（独立骰近似）。"""
        my_counts = tuple(int(x) for x in observation["my_dice_counts"])
        total = int(observation.get("total_dice_on_table", sum(my_counts)))
        unknown = max(0, total - sum(my_counts))

        own = self._own_success(mode, face, my_counts)
        need_from_unknown = max(0, int(count) - own)
        p = (2.0 / 6.0) if mode == '飞' else (1.0 / 6.0)
        return self._binom_tail(unknown, p, need_from_unknown)

    @staticmethod
    def _zhai_rank(face: int) -> int:
        """斋模式面值顺序：2<3<4<5<6<1"""
        order = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 1: 5}
        return order[face]

    def _is_greater(self, new: Guess, old: Guess) -> bool:
        """遵循环境规则判断 new 是否严格大于 old（与 env 一致）。"""
        if new.mode == old.mode:
            if new.mode == '飞':
                if new.count > old.count:
                    return True
                if new.count == old.count and new.face > old.face:
                    return True
                return False
            else:
                if new.count > old.count:
                    return True
                if new.count == old.count and self._zhai_rank(new.face) > self._zhai_rank(old.face):
                    return True
                return False
        # 跨模式
        if new.mode == '斋' and old.mode == '飞':
            return new.count >= (old.count + 1) // 2
        if new.mode == '飞' and old.mode == '斋':
            return new.count >= (old.count * 2)
        return False

    def _min_count(self) -> int:
        return self.num_players + 1

    def _first_move(self, observation: Dict) -> Guess:
        """首行动：从 n+1 起，选择期望边际 E-count 较大的模式与面值"""
        my_counts = tuple(int(x) for x in observation["my_dice_counts"])
        total = int(observation.get("total_dice_on_table", sum(my_counts)))
        unknown = max(0, total - sum(my_counts))
        min_c = self._min_count()

        # 评估飞/斋两个方案
        best_choice = None
        best_margin = -1e9
        for mode in ['飞', '斋']:
            face = self._best_face(mode, my_counts)
            p = (2.0 / 6.0) if mode == '飞' else (1.0 / 6.0)
            own = self._own_success(mode, face, my_counts)
            exp_total = own + unknown * p
            margin = exp_total - min_c
            if margin > best_margin:
                best_margin = margin
                best_choice = Guess(mode=mode, count=min_c, face=face)
        return best_choice

    def _propose_candidates(self, last: Guess, observation: Dict) -> Tuple[Guess, ...]:
        """生成一小批候选叫点（已确保严格更大且满足最低计数）。"""
        min_c = self._min_count()
        total = int(observation.get("total_dice_on_table", self.num_players * 5))
        my_counts = tuple(int(x) for x in observation["my_dice_counts"])
        cands = []

        # 1) 同模式小步抬价（优先面值+1）
        if last.mode == '斋':
            next_face = 2 if last.face == 1 else min(6, last.face + 1)
            g = Guess(mode='斋', count=last.count, face=next_face)
            if self._is_greater(g, last) and g.count >= min_c:
                cands.append(g)
        else:  # 飞
            if last.face < 6:
                next_face = last.face + 1
                if next_face == 1:
                    next_face = 2
                g = Guess(mode='飞', count=last.count, face=next_face)
                if self._is_greater(g, last) and g.face != 1 and g.count >= min_c:
                    cands.append(g)

        # 2) 同模式数量+1（选择当前模式的最佳面值）
        for dc in range(1, self.max_extra_raise + 1):
            new_count = min(total, last.count + dc)
            face = self._best_face(last.mode, my_counts)
            g = Guess(mode=last.mode, count=new_count, face=face)
            if self._is_greater(g, last) and g.count >= min_c:
                cands.append(g)

        # 3) 跨模式切换
        if last.mode == '飞':
            new_c = max(min_c, (last.count + 1) // 2)  # ceil(count/2)
            face = self._best_face('斋', my_counts)
            g = Guess(mode='斋', count=new_c, face=face)
            if self._is_greater(g, last) and g.count >= min_c:
                cands.append(g)
        else:  # last.mode == '斋'
            new_c = max(min_c, last.count * 2)
            face = self._best_face('飞', my_counts)
            g = Guess(mode='飞', count=new_c, face=face)
            if self._is_greater(g, last) and g.face != 1 and g.count >= min_c:
                cands.append(g)

        # 去重（可能不同路径产生相同候选）
        uniq = {}
        for g in cands:
            uniq[(g.mode, g.count, g.face)] = g
        return tuple(uniq.values())

    def get_action(self, observation: Dict) -> Action:
        # 1) 首行动
        last_guess: Optional[Guess] = observation.get("last_guess")
        if last_guess is None:
            return self._first_move(observation)

        # 2) 是否挑战
        p_last = self._prob_true(last_guess.mode, last_guess.count, last_guess.face, observation)
        if p_last < self.theta_challenge:
            return Challenge()

        # 3) 候选生成与筛选
        candidates = self._propose_candidates(last_guess, observation)
        if not candidates:
            return Challenge()

        # 计算每个候选的成立概率
        scored = []
        for g in candidates:
            p = self._prob_true(g.mode, g.count, g.face, observation)
            scored.append((p, g))

        # 先找满足目标阈值的“最小抬价”
        meets = [(p, g) for (p, g) in scored if p >= self.target_raise]
        if meets:
            # 优先数量最小，其次面值最小
            meets.sort(key=lambda x: (x[1].count, x[1].face, -x[0]))
            return meets[0][1]

        # 否则选择概率最高的候选；若仍很低，则改为挑战
        scored.sort(key=lambda x: x[0], reverse=True)
        best_p, best_g = scored[0]
        if best_p < 0.35:
            return Challenge()
        return best_g
