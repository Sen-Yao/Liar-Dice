import numpy as np
from typing import Dict
from env import Guess, Challenge, Action, LiarDiceEnv


class HumanAgent:
    """人类玩家Agent，通过命令行与用户交互"""
    
    def __init__(self, agent_id: str, num_players: int):
        self.agent_id = agent_id
        self.num_players = num_players
    
    def get_action(self, observation: Dict) -> Action:
        """获取人类玩家的动作"""
        self._display_game_state(observation)
        return self._get_user_input(observation)
    
    def _display_game_state(self, observation: Dict):
        """显示游戏状态"""
        print(f"\n{'='*60}")
        print(f"🎲 轮到你了，{self.agent_id}!")
        print(f"{'='*60}")
        
        # 显示玩家手牌
        print(f"你的骰子: {self._format_dice(observation['my_dice_counts'])}")
        print(f"场上总骰子数: {observation['total_dice_on_table']}")
        print(f"玩家数量: {observation['num_players']}")
        
        # 显示当前罚分
        penalties = observation['player_penalties']
        print(f"\n📊 当前罚分:")
        for i, penalty in enumerate(penalties):
            player_name = self._get_player_name(i, observation['current_player_id_idx'])
            print(f"  {player_name}: {penalty} 分")
        
        # 显示上一个猜测
        if observation['last_guess'] is not None:
            last_guess = observation['last_guess']
            print(f"\n🎯 上一个猜测: {last_guess.count} 个 {last_guess.face} {last_guess.mode}")
        
        # 显示操作历史
        if observation['game_round_history']:
            print(f"\n📝 本轮历史:")
            for player_idx, guess in observation['game_round_history']:
                player_name = self._get_player_name(player_idx, observation['current_player_id_idx'])
                print(f"  {player_name}: {guess.count} 个 {guess.face} {guess.mode}")
    
    def _get_player_name(self, player_idx: int, current_player_idx: int) -> str:
        """获取玩家名称"""
        if player_idx == 0:
            return "你"
        elif player_idx == current_player_idx:
            return f"当前玩家({player_idx})"
        else:
            return f"玩家{player_idx}"
    
    def _format_dice(self, dice_counts: tuple) -> str:
        """格式化骰子显示"""
        dice = []
        for face, count in enumerate(dice_counts, 1):
            dice.extend([face] * count)
        return f"[{', '.join(map(str, sorted(dice)))}]"
    
    def _get_user_input(self, observation: Dict) -> Action:
        """获取用户输入"""
        while True:
            print("\n请选择你的操作:")
            print("1. 出猜测 (Guess)")
            print("2. 检验上一个猜测 (Challenge)")
            
            try:
                choice = input("请输入选择 (1/2): ").strip().lower()
                
                if choice in ["2", "challenge", "c"]:
                    if observation['last_guess'] is None:
                        print("❌ 这是第一个回合，无法检验!")
                        continue
                    print("🔍 你选择检验上一个猜测!")
                    return Challenge()
                
                elif choice in ["1", "guess", "g"]:
                    return self._get_guess_input(observation)
                
                else:
                    print("❌ 无效选择，请重新输入!")
                    
            except KeyboardInterrupt:
                print("\n\n游戏已取消!")
                exit()
            except EOFError:
                print("\n\n输入结束，游戏已取消!")
                exit()
    
    def _get_guess_input(self, observation: Dict) -> Guess:
        """获取猜测输入"""
        while True:
            try:
                print("\n请输入你的猜测:")
                
                
                # 输入数量
                max_count = observation['total_dice_on_table']
                count_input = input(f"请输入猜测数量 (1-{max_count}): ").strip()
                count = int(count_input)
                if count < 1 or count > max_count:
                    print(f"❌ 数量必须在1到{max_count}之间!")
                    continue
                
                # 输入点数
                valid_faces = [1, 2, 3, 4, 5, 6]
                face_input = input(f"请输入点数 ({[1, 2, 3, 4, 5, 6]}): ").strip()
                face = int(face_input)
                if face not in valid_faces:
                    print(f"❌ 点数必须是{valid_faces}中的一个!")
                    continue
                
                                # 选择模式
                print("选择模式:")
                print("1. 飞 (点数1作为万能牌)")
                print("2. 斋 (点数1不作为万能牌)")
                
                if face != 1:
                    mode_choice = input("请选择模式 (1/2): ").strip().lower()
                    if mode_choice in ["1", "飞", "fly", ""]:
                        mode = '飞'
                    elif mode_choice in ["2", "斋", "zhai"]:
                        mode = '斋'
                    else:
                        print("❌ 无效模式选择!")
                        continue
                else:
                    mode = '斋'
                
                guess = Guess(mode=mode, count=count, face=face)
                
                # 验证猜测是否合法
                if not self._is_valid_guess(guess, observation):
                    continue
                
                print(f"✅ 你的猜测: {count} 个 {face} {mode}")
                return guess
                
            except ValueError:
                print("❌ 请输入有效的数字!")
            except KeyboardInterrupt:
                print("\n\n游戏已取消!")
                exit()
            except EOFError:
                print("\n\n输入结束，游戏已取消!")
                exit()
    
    def _is_valid_guess(self, guess: Guess, observation: Dict) -> bool:
        """验证猜测是否合法"""
        if observation['last_guess'] is None:
            # 第一个猜测，数量必须大于玩家数
            if guess.count <= observation['num_players']:
                print(f"❌ 第一个猜测的数量必须大于玩家数({observation['num_players']})!")
                return False
        else:
            # 必须大于上一个猜测
            if not LiarDiceEnv._is_strictly_greater(guess, observation['last_guess']):
                print("❌ 你的猜测必须大于上一个猜测!")
                print(f"上一个猜测: {observation['last_guess'].count} 个 {observation['last_guess'].face} {observation['last_guess'].mode}")
                print(f"你的猜测: {guess.count} 个 {guess.face} {guess.mode}")
                return False
        
        return True