import os
import json
import re
from typing import Dict, Optional
from env import Guess, Challenge, Action
from openai import OpenAI


class LLMAgent:
    """
    基于LLM的Agent，通过自然语言与游戏交互
    从环境变量中读取API配置
    """
    
    def __init__(self, agent_id: str, num_players: int):
        self.agent_id = agent_id
        self.num_players = num_players
        
        # 从环境变量读取API配置
        self.api_base = os.getenv('LLM_API_BASE', 'https://api2.aigcbest.top/v1')
        self.api_key = os.getenv('LLM_API_KEY', 'sk-xxx')
        self.model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
        
        # 创建系统提示
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """创建系统提示"""
        return f"""你是一个骰子骗子(Liar Dice)游戏的玩家。游戏规则如下：

游戏基本信息：
- 每位玩家有5颗六面骰子(点数1-6)
- 玩家轮流进行猜测或检验
- 点数1在"飞"模式下可作为万能牌，在"斋"模式下只能算作1

猜测规则：
1. 第一个回合必须猜测，且数量必须大于玩家数({self.num_players})
2. 后续回合必须出大于上一个猜测的猜测
3. 猜测格式：(模式, 数量, 点数)
4. 模式选择："飞"(1作为万能牌) 或 "斋"(1仅作为1)
5. "飞"模式点数范围：2-6，"斋"模式点数范围：1-6

猜测大小比较：
- 同模式：数量大的大，数量相同则点数大的大
- "斋" vs "飞"："斋"的价值大约是"飞"的2倍

你的回复必须严格按照以下JSON格式：
{{
    "thinking": "你的思考过程",
    "action": "GUESS"或"CHALLENGE",
    "guess": {{
        "mode": "飞"或"斋",
        "count": 数量,
        "face": 点数
    }}
}}

如果选择检验，guess字段可以为null。
请确保你的猜测是合法的！"""
    
    def get_action(self, observation: Dict) -> Action:
        """获取LLM的动作决策"""
        # 将游戏状态转换为自然语言描述
        state_description = self._state_to_natural_language(observation)
        
        # 构建对话消息
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": state_description}
        ]
        
        # 调用LLM API
        response = self._call_llm_api(messages)
        
        # 解析LLM响应
        action = self._parse_llm_response(response, observation)
        
        return action
    
    def _state_to_natural_language(self, observation: Dict) -> str:
        """将游戏状态转换为自然语言描述"""
        description = f"""当前游戏状态：

你是玩家 {self.agent_id}，共有 {self.num_players} 名玩家。

你的手牌：{self._format_dice(observation['my_dice_counts'])}
场上总骰子数：{observation['total_dice_on_table']}

当前罚分情况："""
        
        # 添加罚分信息
        penalties = observation['player_penalties']
        for i, penalty in enumerate(penalties):
            player_name = "你" if i == observation['current_player_id_idx'] else f"玩家{i}"
            description += f"\n- {player_name}: {penalty}分"
        
        # 添加上一个猜测
        if observation['last_guess'] is not None:
            last_guess = observation['last_guess']
            description += f"\n\n上一个猜测：{last_guess.mode}模式 {last_guess.count}个{last_guess.face}"
        else:
            description += "\n\n这是本回合的第一个猜测"
        
        # 添加历史记录
        if observation['game_round_history']:
            description += "\n\n本轮历史："
            for player_idx, guess in observation['game_round_history']:
                player_name = "你" if player_idx == observation['current_player_id_idx'] else f"玩家{player_idx}"
                description += f"\n- {player_name}: {guess.mode} {guess.count}个{guess.face}"
        
        description += f"\n\n现在轮到你行动了。请做出你的决策。"
        
        return description
    
    def _format_dice(self, dice_counts: tuple) -> str:
        """格式化骰子显示"""
        dice = []
        for face, count in enumerate(dice_counts, 1):
            dice.extend([face] * count)
        return f"[{', '.join(map(str, sorted(dice)))}]"
    
    def _call_llm_api(self, messages: list) -> str:
        """调用LLM API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            raise Exception(f"LLM调用失败: {e}")
    
    def _parse_llm_response(self, response: str, observation: Dict) -> Action:
        """解析LLM响应并返回Action"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("响应中未找到有效的JSON格式")
            
            json_str = json_match.group()
            data = json.loads(json_str)
            
            action_type = data.get('action', '').upper()
            
            if action_type == 'CHALLENGE':
                # 检查是否可以挑战
                if observation['last_guess'] is None:
                    print(f"LLM尝试在第一个回合挑战，回退到猜测")
                    return self._fallback_guess(observation)
                return Challenge()
            
            elif action_type == 'GUESS':
                guess_data = data.get('guess', {})
                if not guess_data:
                    raise ValueError("缺少guess字段")
                
                mode = guess_data.get('mode')
                count = guess_data.get('count')
                face = guess_data.get('face')
                
                # 验证参数
                if mode not in ['飞', '斋']:
                    raise ValueError(f"无效的模式: {mode}")
                
                if not isinstance(count, int) or count < 1:
                    raise ValueError(f"无效的数量: {count}")
                
                if not isinstance(face, int) or face < 1 or face > 6:
                    raise ValueError(f"无效的点数: {face}")
                
                # 验证模式与点数的匹配
                if mode == '飞' and face == 1:
                    raise ValueError("飞模式下不能选择点数1")
                
                guess = Guess(mode=mode, count=count, face=face)
                
                # 验证猜测是否合法
                if not self._is_valid_guess(guess, observation):
                    print(f"LLM生成了非法猜测: {guess}，尝试修正...")
                    return self._correct_guess(guess, observation)
                
                return guess
            
            else:
                raise ValueError(f"未知的动作类型: {action_type}")
                
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response}")
            return self._fallback_action(observation)
        except Exception as e:
            print(f"响应解析失败: {e}")
            print(f"原始响应: {response}")
            return self._fallback_action(observation)
    
    def _is_valid_guess(self, guess: Guess, observation: Dict) -> bool:
        """验证猜测是否合法"""
        from env import LiarDiceEnv
        
        if observation['last_guess'] is None:
            # 第一个猜测，数量必须大于玩家数
            return guess.count > observation['num_players']
        else:
            # 必须大于上一个猜测
            return LiarDiceEnv._is_strictly_greater(guess, observation['last_guess'])
    
    def _correct_guess(self, invalid_guess: Guess, observation: Dict) -> Action:
        """尝试修正非法猜测"""
        from env import LiarDiceEnv
        
        if observation['last_guess'] is None:
            # 第一个猜测，修正数量
            count = max(observation['num_players'] + 1, invalid_guess.count)
            return Guess(mode=invalid_guess.mode, count=count, face=invalid_guess.face)
        else:
            # 尝试找到最小的合法猜测
            last_guess = observation['last_guess']
            
            # 简单策略：增加数量或点数
            if invalid_guess.count > last_guess.count:
                # 数量已经更大，修正点数
                face = min(6, max(2, invalid_guess.face))
                return Guess(mode=invalid_guess.mode, count=invalid_guess.count, face=face)
            else:
                # 增加数量
                count = last_guess.count + 1
                face = max(2 if invalid_guess.mode == '飞' else 1, invalid_guess.face)
                return Guess(mode=invalid_guess.mode, count=count, face=face)
    
    def _fallback_guess(self, observation: Dict) -> Action:
        """回退到安全的猜测策略"""
        if observation['last_guess'] is None:
            # 第一个猜测
            return Guess(mode='飞', count=self.num_players + 1, face=4)
        else:
            # 保守地增加猜测
            last_guess = observation['last_guess']
            if last_guess.face < 6:
                return Guess(mode=last_guess.mode, count=last_guess.count, face=last_guess.face + 1)
            else:
                new_face = 1 if last_guess.mode == '斋' else 2
                return Guess(mode=last_guess.mode, count=last_guess.count + 1, face=new_face)
    
    def _fallback_action(self, observation: Dict) -> Action:
        """完全回退到基本策略"""
        return self._fallback_guess(observation)