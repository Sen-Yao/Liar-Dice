"""
baseline_agents.py - Baseline Agent实现

包含用于性能对比的baseline agents：
- RandomAgent: 从所有合法动作中均匀随机选择
- ConservativeAgent: 保守策略，倾向于挑战
- AggressiveAgent: 激进策略，倾向于猜测
- OptimizedLLMAgent: 优化的LLM agent（基于通义千问API）

这些agents用于baseline实验，提供性能下界和对比基准。
"""

import random
from typing import Dict, Optional
from env import Action, Guess, Challenge
from utils import get_legal_actions
from agents.llm_agent import LLMAgent


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


class ConservativeAgent:
    """
    保守型baseline agent：倾向于挑战，猜测时选择最小合法猜测

    特性：
    - **保守挑战策略**：使用较低的信心阈值（默认0.5），更容易挑战对手
    - **最小猜测策略**：必须猜测时选择最小合法猜测（count最小，face最小）
    - **100%合法性保证**：使用`utils.get_legal_actions()`保证所有动作合法
    - **概率估计**：基于期望骰子数判断对手猜测是否可信

    策略说明：
    - 挑战条件：对手猜测数量 > 期望数量 + 信心阈值
    - 猜测策略：从所有合法猜测中选择最小的（优先飞模式，count和face尽量小）

    用途：
    - 作为保守型baseline，评估激进策略的优势
    - 测试环境对保守策略的奖励设计
    - 提供性能对比的下界参考

    示例：
        ```python
        agent = ConservativeAgent(agent_id="conservative_0", num_players=2)
        action = agent.get_action(observation)
        ```
    """

    def __init__(
        self,
        agent_id: str,
        num_players: int,
        confidence_threshold: float = 0.5
    ):
        """
        初始化保守型agent

        参数：
            agent_id: agent标识符
            num_players: 游戏玩家数量
            confidence_threshold: 挑战信心阈值（越小越容易挑战，默认0.5）
        """
        self.agent_id = agent_id
        self.num_players = num_players
        self.confidence_threshold = confidence_threshold

        # 统计信息
        self.action_count = 0
        self.guess_count = 0
        self.challenge_count = 0

    def get_action(self, observation: Dict) -> Action:
        """
        保守策略：优先挑战，必须猜测时选择最小合法猜测

        决策流程：
        1. 获取所有合法动作
        2. 如果可以挑战，计算挑战成功概率：
           - 计算期望骰子数
           - 如果对手猜测 > 期望 + 信心阈值，则挑战
        3. 否则从合法猜测中选择最小的

        参数：
            observation: 游戏观察状态

        返回：
            Action: Challenge或最小合法Guess
        """
        # 获取所有合法动作
        legal_actions = get_legal_actions(observation, self.num_players)

        if not legal_actions:
            raise RuntimeError(
                f"[ConservativeAgent] 没有合法动作可选！observation={observation}"
            )

        # 分离挑战和猜测动作
        challenge_actions = [a for a in legal_actions if isinstance(a, Challenge)]
        guess_actions = [a for a in legal_actions if isinstance(a, Guess)]

        # 如果可以挑战，评估是否应该挑战
        if challenge_actions:
            last_guess = observation['last_guess']
            # 计算期望骰子数
            expect_dice_counts = self._calculate_expectation(
                observation, last_guess.mode
            )
            # 保守策略：期望 + 小阈值就挑战
            if last_guess.count > expect_dice_counts[last_guess.face - 1] + self.confidence_threshold:
                self.action_count += 1
                self.challenge_count += 1
                return Challenge()

        # 否则选择最小合法猜测
        if guess_actions:
            # 排序规则：count最小，face最小，优先飞模式
            min_guess = min(guess_actions, key=lambda g: (g.count, g.face, g.mode == '斋'))
            self.action_count += 1
            self.guess_count += 1
            return min_guess

        # 极端情况：只能挑战
        self.action_count += 1
        self.challenge_count += 1
        return Challenge()

    def _calculate_expectation(self, observation: Dict, mode: str) -> list:
        """
        计算期望骰子数（基于概率估计）

        计算方法：
        - 未知骰子数 = (玩家数 - 1) × 5
        - 每个点数的期望 = 未知骰子数 / 6
        - 飞模式：目标点数期望 = 自己的 + 对手期望×2（包括1）
        - 斋模式：目标点数期望 = 自己的 + 对手期望×1

        参数：
            observation: 游戏观察状态
            mode: 计算模式（'飞' 或 '斋'）

        返回：
            list: 长度为6的列表，表示每个点数（1-6）的期望数量
        """
        unknown_total_dice = (self.num_players - 1) * 5
        expect_per_face = unknown_total_dice / 6

        # 从自己的骰子开始
        expect_counts = list(observation['my_dice_counts'])

        if mode == '飞':
            # 飞模式：1是万能牌
            expect_counts[0] += expect_per_face  # 1本身
            for i in range(1, 6):
                # 其他点数包括1的贡献
                expect_counts[i] += expect_per_face * 2
        else:  # '斋'模式
            # 斋模式：每个点数独立
            for i in range(6):
                expect_counts[i] += expect_per_face

        return expect_counts

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "action_count": self.action_count,
            "guess_count": self.guess_count,
            "challenge_count": self.challenge_count,
        }

        if self.action_count > 0:
            stats["challenge_rate"] = self.challenge_count / self.action_count
        else:
            stats["challenge_rate"] = 0.0

        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.action_count = 0
        self.guess_count = 0
        self.challenge_count = 0


class AggressiveAgent:
    """
    激进型baseline agent：倾向于猜测，选择较大的合法猜测

    特性：
    - **激进猜测策略**：使用较高的信心阈值（默认2.0），很少挑战对手
    - **大胆猜测策略**：基于手牌选择较大的合法猜测
    - **100%合法性保证**：使用`utils.get_legal_actions()`保证所有动作合法
    - **手牌优化**：优先选择手里较多的点数进行猜测

    策略说明：
    - 挑战条件：对手猜测数量 > 期望数量 + 高阈值（很少触发）
    - 猜测策略：
      1. 找出手里最多的点数
      2. 在合法猜测中选择包含该点数的
      3. 选择count较大的猜测（但不超过合理范围）
      4. 优先选择与手牌匹配的模式

    用途：
    - 作为激进型baseline，评估保守策略的必要性
    - 测试环境对风险承担的奖励
    - 提供不同风格的性能对比

    示例：
        ```python
        agent = AggressiveAgent(agent_id="aggressive_0", num_players=2)
        action = agent.get_action(observation)
        ```
    """

    def __init__(
        self,
        agent_id: str,
        num_players: int,
        confidence_threshold: float = 2.0
    ):
        """
        初始化激进型agent

        参数：
            agent_id: agent标识符
            num_players: 游戏玩家数量
            confidence_threshold: 挑战信心阈值（越大越不容易挑战，默认2.0）
        """
        self.agent_id = agent_id
        self.num_players = num_players
        self.confidence_threshold = confidence_threshold

        # 统计信息
        self.action_count = 0
        self.guess_count = 0
        self.challenge_count = 0

    def get_action(self, observation: Dict) -> Action:
        """
        激进策略：优先猜测，基于手牌选择较大的合法猜测

        决策流程：
        1. 获取所有合法动作
        2. 如果可以挑战，计算是否应该挑战：
           - 计算期望骰子数
           - 只有对手猜测明显过高时才挑战（高阈值）
        3. 否则基于手牌选择较大的合法猜测

        参数：
            observation: 游戏观察状态

        返回：
            Action: Challenge或较大的合法Guess
        """
        # 获取所有合法动作
        legal_actions = get_legal_actions(observation, self.num_players)

        if not legal_actions:
            raise RuntimeError(
                f"[AggressiveAgent] 没有合法动作可选！observation={observation}"
            )

        # 分离挑战和猜测动作
        challenge_actions = [a for a in legal_actions if isinstance(a, Challenge)]
        guess_actions = [a for a in legal_actions if isinstance(a, Guess)]

        # 如果可以挑战，评估是否应该挑战（高阈值，很少挑战）
        if challenge_actions:
            last_guess = observation['last_guess']
            # 计算期望骰子数
            expect_dice_counts = self._calculate_expectation(
                observation, last_guess.mode
            )
            # 激进策略：只有明显过高才挑战
            if last_guess.count > expect_dice_counts[last_guess.face - 1] + self.confidence_threshold:
                self.action_count += 1
                self.challenge_count += 1
                return Challenge()

        # 否则选择较大的合法猜测（基于手牌）
        if guess_actions:
            chosen_guess = self._choose_aggressive_guess(observation, guess_actions)
            self.action_count += 1
            self.guess_count += 1
            return chosen_guess

        # 极端情况：只能挑战
        self.action_count += 1
        self.challenge_count += 1
        return Challenge()

    def _choose_aggressive_guess(
        self,
        observation: Dict,
        guess_actions: list
    ) -> Guess:
        """
        从合法猜测中选择激进的猜测（基于手牌）

        策略：
        1. 找出手里最多的点数（作为首选）
        2. 计算期望骰子数
        3. 在合法猜测中选择：
           - 包含首选点数的
           - count接近或略低于期望值的（不要太保守）
           - 优先选择与手牌匹配的模式

        参数：
            observation: 游戏观察状态
            guess_actions: 所有合法的Guess动作

        返回：
            Guess: 选中的激进猜测
        """
        my_dice = observation['my_dice_counts']

        # 找出手里最多的点数
        max_count = 0
        best_face = 2  # 默认从2开始（飞模式不能喊1）
        for face in range(1, 7):
            if my_dice[face - 1] > max_count:
                max_count = my_dice[face - 1]
                best_face = face

        # 判断首选模式
        # 如果该点数比1多，优先斋模式；否则优先飞模式
        preferred_mode = '斋' if (best_face != 1 and my_dice[best_face - 1] > my_dice[0]) else '飞'

        # 计算期望值
        fly_expect = self._calculate_expectation(observation, '飞')
        zhai_expect = self._calculate_expectation(observation, '斋')

        # 筛选包含首选点数的合法猜测
        candidate_guesses = []
        for guess in guess_actions:
            if guess.face == best_face:
                candidate_guesses.append(guess)

        # 如果没有首选点数的猜测，使用所有合法猜测
        if not candidate_guesses:
            candidate_guesses = guess_actions

        # 从候选中选择较大的猜测（但不要太过激进）
        # 选择count在期望值附近的（期望 - 1 到 期望 + 1）
        best_guess = None
        best_score = -float('inf')

        for guess in candidate_guesses:
            expect_val = fly_expect[guess.face - 1] if guess.mode == '飞' else zhai_expect[guess.face - 1]

            # 评分：优先选择接近期望值的，不要太保守也不要太激进
            # count越接近期望，分数越高
            score = -abs(guess.count - expect_val)

            # 如果是首选点数，加分
            if guess.face == best_face:
                score += 5

            # 如果是首选模式，加分
            if guess.mode == preferred_mode:
                score += 2

            if score > best_score:
                best_score = score
                best_guess = guess

        # 如果没有找到合适的，选择count最大的
        if best_guess is None:
            best_guess = max(candidate_guesses, key=lambda g: g.count)

        return best_guess

    def _calculate_expectation(self, observation: Dict, mode: str) -> list:
        """
        计算期望骰子数（与ConservativeAgent相同的逻辑）

        参数：
            observation: 游戏观察状态
            mode: 计算模式（'飞' 或 '斋'）

        返回：
            list: 长度为6的列表，表示每个点数（1-6）的期望数量
        """
        unknown_total_dice = (self.num_players - 1) * 5
        expect_per_face = unknown_total_dice / 6

        expect_counts = list(observation['my_dice_counts'])

        if mode == '飞':
            expect_counts[0] += expect_per_face
            for i in range(1, 6):
                expect_counts[i] += expect_per_face * 2
        else:  # '斋'模式
            for i in range(6):
                expect_counts[i] += expect_per_face

        return expect_counts

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "action_count": self.action_count,
            "guess_count": self.guess_count,
            "challenge_count": self.challenge_count,
        }

        if self.action_count > 0:
            stats["challenge_rate"] = self.challenge_count / self.action_count
        else:
            stats["challenge_rate"] = 0.0

        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.action_count = 0
        self.guess_count = 0
        self.challenge_count = 0


class OptimizedLLMAgent(LLMAgent):
    """
    优化的LLM baseline agent：基于通义千问API，使用更确定性的参数

    特性：
    - **继承LLMAgent**：复用完整的三层验证机制
    - **优化参数**：使用更低的temperature（0.3），提高决策确定性
    - **100%合法性保证**：继承父类的Layer 1-3验证机制
    - **统计追踪**：默认启用统计功能

    与LLMAgent的区别：
    - Temperature: 0.7 → 0.3（更确定性，减少随机性）
    - 默认启用统计（enable_stats=True）
    - 作为baseline实验的高级基线

    三层验证机制（继承自LLMAgent）：
    - Layer 1: JSON解析与格式验证
    - Layer 2: 游戏规则合法性验证
    - Layer 3: Fallback策略（从合法动作中随机选择）

    用途：
    - 作为高级baseline，评估RL模型相对于human-like策略的优势
    - 测试LLM在博弈论游戏中的表现
    - 提供可解释的决策过程（reasoning字段）

    示例：
        ```python
        agent = OptimizedLLMAgent(agent_id="llm_0", num_players=2)
        action = agent.get_action(observation)
        stats = agent.get_stats()  # 获取API调用统计
        ```

    环境变量配置：
        - DASHSCOPE_API_KEY: 通义千问API密钥（必需）
        - DASHSCOPE_API_BASE: API端点（可选，默认为阿里云）
        - DASHSCOPE_MODEL: 模型名称（可选，默认qwen-max）
    """

    def __init__(
        self,
        agent_id: str,
        num_players: int,
        temperature: float = 0.3  # 更低的temperature，更确定性
    ):
        """
        初始化优化的LLM agent

        参数：
            agent_id: agent标识符
            num_players: 游戏玩家数量
            temperature: LLM采样温度（默认0.3，更确定性）
        """
        # 调用父类初始化，启用统计
        super().__init__(
            agent_id=agent_id,
            num_players=num_players,
            temperature=temperature,
            enable_stats=True  # 默认启用统计
        )

    # 所有其他功能继承自LLMAgent：
    # - get_action(): 完整的三层验证机制
    # - _state_to_natural_language(): 状态转换为自然语言
    # - _get_legal_actions_hint(): 合法动作提示
    # - _call_llm_api(): API调用
    # - _parse_llm_response(): 响应解析
    # - _is_valid_guess(): 合法性验证
    # - _fallback_action(): Fallback策略
    # - get_stats(): 统计信息
    # - reset_stats(): 重置统计
