import os
import json
import re
import time
from typing import Dict, Optional, List
from env import Guess, Challenge, Action
from openai import OpenAI


class LLMAgent:
    """
    基于LLM的Agent，通过自然语言与游戏交互
    优化版：使用通义千问（Qwen）API，增强合法性保障与测试友好性

    输入流: observation (Dict) → 状态描述 + 合法动作提示
    处理流: LLM推理 → JSON输出
    输出流: 三层验证 → Action (Guess/Challenge)
    """

    def __init__(
        self,
        agent_id: str,
        num_players: int,
        temperature: float = 0.7,
        enable_stats: bool = True,
        use_api: bool = True
    ):
        self.agent_id = agent_id
        self.num_players = num_players
        self.temperature = temperature
        self.use_api = bool(use_api)

        # 从环境变量读取API配置（默认使用Qwen）
        self.api_base = os.getenv(
            'DASHSCOPE_API_BASE',
            'https://dashscope.aliyuncs.com/compatible-mode/v1'
        )
        self.api_key = os.getenv('DASHSCOPE_API_KEY', 'your-api-key-here')
        self.has_api = bool(self.api_key and self.api_key != 'your-api-key-here')
        self.model = os.getenv('DASHSCOPE_MODEL', 'qwen-max')
        
        # 改进：确保 use_api 和 has_api 行为一致
        # 如果明确要求使用API但没有API Key，发出警告
        if use_api and not self.has_api:
            print(f"[LLMAgent] 警告: use_api=True 但未设置有效的API Key，将使用fallback模式")
            self.use_api = False
        else:
            self.use_api = use_api

        # 创建OpenAI兼容客户端
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )

        # 统计追踪（用于测试评估）
        self.stats = {
            "api_calls": 0,
            "total_latency": 0.0,
            "illegal_attempts": 0,
            "fallback_count": 0,
            "parse_errors": 0
        } if enable_stats else None

        # 创建系统提示
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """创建系统提示（完整规则 + 结构化输出）"""
        return f"""你是一个骰子骗子(Liar's Dice)游戏的玩家。请严格遵守以下规则：

## 游戏基本信息
- 玩家人数：{self.num_players}人
- 每位玩家有5颗六面骰子（点数1-6）
- 玩家轮流进行"猜测"或"挑战"
- 你只能看到自己的骰子，无法看到对手的骰子

## 游戏模式
### 飞模式（正常模式）
- 点数1是万能牌（wild），可以代替任何数字
- 计数规则：目标点数数量 + 1点数量
- 例如：猜测"3个4飞"，统计所有4和所有1的总数
- 点数大小排序：1 < 2 < 3 < 4 < 5 < 6（用于同模式比较）
- **重要：飞模式下不能喊1点**（因为1是万能牌）

### 斋模式
- 点数1就是1，不代替其他数字
- 计数规则：只计算目标点数数量
- 例如：猜测"3个4斋"，只统计所有4的数量
- 点数大小排序：2 < 3 < 4 < 5 < 6 < 1（注意：1最大）

## 猜测合法性规则
### 首轮限制
- 本局首位玩家的第一次猜测个数必须**严格大于玩家人数**（> {self.num_players}）
- 首轮不能挑战（因为还没有上一个猜测）

### 同模式比较（飞→飞 或 斋→斋）
- 新猜测必须**严格大于**上一个猜测
- 飞模式：个数更多，或个数相同时点数更大
- 斋模式：个数更多，或个数相同时按斋模式排序更大

### 跨模式转换
- 飞 → 斋：新个数 ≥ 旧个数/2（向上取整），不比较点数
- 斋 → 飞：新个数 ≥ 旧个数×2，不比较点数

## 挑战（开牌）规则
- 挑战后统计场上实际数量（按当前模式计算）
- 挑战者胜：实际数量 < 猜测数量
- 被挑战者胜：实际数量 ≥ 猜测数量

## 输出格式要求
你的回复**必须**是严格的JSON格式：
```json
{{
    "reasoning": "你的推理过程（简短说明为什么这样决策）",
    "action": "GUESS",
    "guess": {{
        "mode": "飞",
        "count": 4,
        "face": 3
    }}
}}
```

或者选择挑战：
```json
{{
    "reasoning": "对手的猜测不可信",
    "action": "CHALLENGE"
}}
```

## 合法性自检清单（在决策前检查）
1. 飞模式下face不能是1
2. 首轮count必须 > {self.num_players}
3. 非首轮必须严格大于上一个猜测
4. count必须在1到{self.num_players * 5}之间
5. face必须在1到6之间

**重要提示**：我会在你的输出后进行合法性验证。如果你的猜测非法，我会强制修正或随机选择合法动作，这会影响你的决策质量。请务必输出合法动作！"""
    
    def get_action(self, observation: Dict) -> Action:
        """获取LLM的动作决策（含统计追踪）"""
        start_time = time.time()

        # 将游戏状态转换为自然语言描述
        state_description = self._state_to_natural_language(observation)

        # 构建对话消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": state_description}
        ]

        # 若无可用API，直接使用fallback，避免阻塞
        if not (self.has_api and self.use_api):
            print(f"[LLMAgent] 使用fallback模式: has_api={self.has_api}, use_api={self.use_api}")
            return self._fallback_action(observation)

        # 调用LLM API（增加保护与超时）
        try:
            response = self._call_llm_api(messages)
        except Exception as e:
            # API 出错时走 fallback，保证评估不中断
            if self.stats is not None:
                self.stats["parse_errors"] += 1
            return self._fallback_action(observation)

        # 记录延迟
        if self.stats is not None:
            elapsed = time.time() - start_time
            self.stats["total_latency"] += elapsed

        # 解析LLM响应（含三层验证）
        action = self._parse_llm_response(response, observation)

        return action
    
    def _state_to_natural_language(self, observation: Dict) -> str:
        """将游戏状态转换为自然语言描述（含合法动作提示）"""
        description = f"""# 当前游戏状态

## 基本信息
- 你的ID：{self.agent_id}
- 玩家人数：{self.num_players}
- 你的手牌：{self._format_dice(observation['my_dice_counts'])}
- 场上总骰子数：{observation['total_dice_on_table']}

## 罚分情况"""

        # 添加罚分信息
        penalties = observation['player_penalties']
        for i, penalty in enumerate(penalties):
            player_name = "你" if i == observation['current_player_id_idx'] else f"玩家{i}"
            description += f"\n- {player_name}: {penalty}分"

        # 添加上一个猜测
        if observation['last_guess'] is not None:
            last_guess = observation['last_guess']
            description += f"\n\n## 上一个猜测\n{last_guess.count}个{last_guess.face}{last_guess.mode}"
        else:
            description += f"\n\n## 上一个猜测\n本回合第一次猜测（必须猜测，个数 > {self.num_players}）"

        # 添加历史记录
        if observation['game_round_history']:
            description += "\n\n## 本轮历史"
            for player_idx, guess in observation['game_round_history']:
                player_name = "你" if player_idx == observation['current_player_id_idx'] else f"玩家{player_idx}"
                description += f"\n- {player_name}: {guess.count}个{guess.face}{guess.mode}"

        # **关键优化**：添加合法动作提示
        legal_actions_hint = self._get_legal_actions_hint(observation)
        description += f"\n\n## 合法动作提示\n{legal_actions_hint}"

        description += "\n\n现在轮到你行动了。请根据上述信息和规则做出决策。"

        return description
    
    def _get_legal_actions_hint(self, observation: Dict) -> str:
        """生成合法动作提示（辅助LLM理解）"""
        from utils import get_legal_actions

        legal_actions = get_legal_actions(observation, self.num_players)

        # 统计合法动作
        guess_actions = [a for a in legal_actions if isinstance(a, Guess)]
        has_challenge = any(isinstance(a, Challenge) for a in legal_actions)

        hint = ""
        if has_challenge:
            hint += "- 可以选择挑战（CHALLENGE）\n"

        if guess_actions:
            # 按模式分组
            fly_guesses = [g for g in guess_actions if g.mode == '飞']
            zhai_guesses = [g for g in guess_actions if g.mode == '斋']

            hint += f"- 可以猜测（GUESS）：共{len(guess_actions)}种合法猜测\n"

            if fly_guesses:
                fly_counts = sorted(set(g.count for g in fly_guesses))
                hint += f"  - 飞模式：个数范围 {fly_counts[0]}-{fly_counts[-1]}，点数2-6\n"

            if zhai_guesses:
                zhai_counts = sorted(set(g.count for g in zhai_guesses))
                hint += f"  - 斋模式：个数范围 {zhai_counts[0]}-{zhai_counts[-1]}，点数1-6\n"

            # 给出推荐示例（最小合法猜测）
            min_guess = min(guess_actions, key=lambda g: (g.count, g.face))
            hint += f"  - 最小合法猜测示例：{min_guess.count}个{min_guess.face}{min_guess.mode}"

        return hint

    def _format_dice(self, dice_counts: tuple) -> str:
        """格式化骰子显示"""
        dice = []
        for face, count in enumerate(dice_counts, 1):
            dice.extend([face] * count)
        return f"[{', '.join(map(str, sorted(dice)))}]"

    def _call_llm_api(self, messages: List[dict]) -> str:
        """调用LLM API（含统计与错误处理）"""
        if self.stats is not None:
            self.stats["api_calls"] += 1

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=480,
                timeout=10
            )

            return response.choices[0].message.content

        except Exception as e:
            if self.stats is not None:
                self.stats["parse_errors"] += 1
            raise Exception(f"LLM API调用失败: {e}")
    
    def _parse_llm_response(self, response: str, observation: Dict) -> Action:
        """
        解析LLM响应并返回Action（三层验证机制）

        Layer 1: JSON解析与格式验证
        Layer 2: 游戏规则合法性验证（调用utils.is_strictly_greater）
        Layer 3: Fallback策略（从合法动作中随机选择）
        """
        try:
            # Layer 1: JSON解析
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("响应中未找到有效的JSON格式")

            json_str = json_match.group()
            data = json.loads(json_str)

            action_type = data.get('action', '').upper()

            # 处理挑战动作
            if action_type == 'CHALLENGE':
                if observation['last_guess'] is None:
                    print(f"[LLMAgent] 非法挑战：首轮不能挑战")
                    if self.stats is not None:
                        self.stats["illegal_attempts"] += 1
                    return self._fallback_action(observation)
                return Challenge()

            # 处理猜测动作
            elif action_type == 'GUESS':
                guess_data = data.get('guess', {})
                if not guess_data:
                    raise ValueError("缺少guess字段")

                mode = guess_data.get('mode')
                count = guess_data.get('count')
                face = guess_data.get('face')

                # 基础参数验证
                if mode not in ['飞', '斋']:
                    raise ValueError(f"无效的模式: {mode}")

                if not isinstance(count, int) or count < 1 or count > self.num_players * 5:
                    raise ValueError(f"无效的数量: {count}")

                if not isinstance(face, int) or face < 1 or face > 6:
                    raise ValueError(f"无效的点数: {face}")

                if mode == '飞' and face == 1:
                    raise ValueError("飞模式下不能选择点数1")

                guess = Guess(mode=mode, count=count, face=face)

                # Layer 2: 规则合法性验证
                if not self._is_valid_guess(guess, observation):
                    print(f"[LLMAgent] 非法猜测: {guess}")
                    if self.stats is not None:
                        self.stats["illegal_attempts"] += 1
                    return self._fallback_action(observation)

                return guess

            else:
                raise ValueError(f"未知的动作类型: {action_type}")

        except json.JSONDecodeError as e:
            print(f"[LLMAgent] JSON解析失败: {e}")
            if self.stats is not None:
                self.stats["parse_errors"] += 1
            return self._fallback_action(observation)

        except Exception as e:
            print(f"[LLMAgent] 响应解析失败: {e}")
            if self.stats is not None:
                self.stats["parse_errors"] += 1
            return self._fallback_action(observation)
    
    def _is_valid_guess(self, guess: Guess, observation: Dict) -> bool:
        """
        验证猜测是否合法（使用utils中的规则）

        这是Layer 2验证的核心，确保与游戏环境规则一致
        """
        from utils import is_strictly_greater

        if observation['last_guess'] is None:
            # 第一个猜测，数量必须大于玩家数
            return guess.count > self.num_players
        else:
            # 必须严格大于上一个猜测
            return is_strictly_greater(guess, observation['last_guess'])

    def _fallback_action(self, observation: Dict) -> Action:
        """
        Layer 3: Fallback策略
        从合法动作集中随机选择（保证100%合法）
        """
        from utils import get_legal_actions
        import random

        print(f"[LLMAgent] 进入fallback_action")
        
        if self.stats is not None:
            self.stats["fallback_count"] += 1

        print(f"[LLMAgent] 调用get_legal_actions，num_players={self.num_players}")
        legal_actions = get_legal_actions(observation, self.num_players)
        print(f"[LLMAgent] get_legal_actions返回，找到{len(legal_actions)}个合法动作")

        if not legal_actions:
            # 极端情况：无合法动作（理论上不应发生）
            print("[LLMAgent] 警告：无合法动作可选")
            return Challenge()

        # 优先选择猜测动作（避免过早挑战）
        guess_actions = [a for a in legal_actions if isinstance(a, Guess)]
        if guess_actions:
            # 选择最小合法猜测（保守策略）
            chosen_action = min(guess_actions, key=lambda g: (g.count, g.face))
            print(f"[LLMAgent] 选择猜测动作: {chosen_action}")
            return chosen_action
        else:
            # 只能挑战
            print("[LLMAgent] 选择挑战动作")
            return Challenge()

    def get_stats(self) -> dict:
        """获取统计数据（用于测试评估）"""
        if self.stats is None:
            return {}

        stats = self.stats.copy()
        if stats["api_calls"] > 0:
            stats["avg_latency"] = stats["total_latency"] / stats["api_calls"]
            stats["illegal_rate"] = stats["illegal_attempts"] / stats["api_calls"]
        else:
            stats["avg_latency"] = 0.0
            stats["illegal_rate"] = 0.0

        return stats

    def reset_stats(self):
        """重置统计数据"""
        if self.stats is not None:
            for key in self.stats:
                self.stats[key] = 0 if isinstance(self.stats[key], int) else 0.0
