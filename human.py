from env import LiarDiceEnv
from agents.human_agent import HumanAgent
from agents.basic_agent import BasicRuleAgent
from agents.llm_agent import LLMAgent

def run_human_vs_ai_mode(args):
    """运行人机对战模式"""
    
    print("🎲 欢迎来到骰子骗子游戏!")
    print("你将与AI进行对战")
    print("="*60)
    
    # 创建环境
    env = LiarDiceEnv(
        num_players=args.num_players, 
        render_mode="human" if args.render else None
    )
    
    # 创建agents字典
    agents = {}
    human_player_idx = 0  # 人类玩家固定为player_0
    
    # 创建人类玩家
    agents[f"player_{human_player_idx}"] = HumanAgent(
        f"player_{human_player_idx}", 
        args.num_players
    )
    
    # 创建AI玩家
    for i in range(args.num_players):
        if i != human_player_idx:
            if args.agent_type == "llm":
                agents[f"player_{i}"] = LLMAgent(f"player_{i}", args.num_players)
            else:
                agents[f"player_{i}"] = BasicRuleAgent(f"player_{i}", args.num_players)
    
    print(f"\n🎮 游戏设置:")
    print(f"你是: player_{human_player_idx} (人类)")
    for i in range(args.num_players):
        if i != human_player_idx:
            print(f"AI是: player_{i} (AI)")
    print(f"总玩家数: {args.num_players}")
    print(f"比赛场数: {args.num_match}")
    
    # 游戏主循环
    for match in range(args.num_match):
        print(f"\n{'='*80}")
        print(f"🏆 第 {match + 1} 场比赛开始!")
        print(f"{'='*80}")
        
        env.reset()
        
        # 单局游戏循环
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                # 本局结束，执行dummy action
                action = None
            else:
                # 获取当前agent的动作
                if agent_id != "player_0":
                    print(f"{agent_id} 思考中...")
                agent_obj = agents[agent_id]
                action = agent_obj.get_action(observation)
                
                # 如果是AI的回合，显示AI的动作（仅在不渲染模式下）
                if not args.render and agent_id != f"player_{human_player_idx}":
                    if isinstance(action, type(None)):
                        pass
                    elif hasattr(action, 'mode'):  # Guess对象
                        print(f"\n🤖 {agent_id} 猜测: {action.mode} {action.count} 个 {action.face}")
                    else:  # Challenge对象
                        print(f"\n🤖 {agent_id} 选择检验!")
            
            env.step(action)
        
        # 显示本局结果
        print(f"\n🏁 第 {match + 1} 场比赛结束!")
        print(f"最终罚分:")
        for agent in env.agents:
            penalty = env.penalties[agent]
            player_type = "你" if agent == f"player_{human_player_idx}" else f"{agent}"
            print(f"  {player_type}: {penalty} 分")
        
        # 显示所有玩家的骰子情况
        print(f"\n🎲 所有玩家的骰子情况:")
        print(f"{"玩家":<12} {"骰子"}")
        print(f"{"-"*12} {"-"*20}")
        
        # 直接遍历player_hands，因为游戏结束后agents列表可能被清空
        for agent in env.player_hands:
            dice_hand = env.player_hands[agent]
            sorted_dice = sorted(dice_hand)
            dice_str = f"[{', '.join(map(str, sorted_dice))}]"
            
            player_name = "你" if agent == f"player_{human_player_idx}" else agent
            print(f"{player_name:<12} {dice_str}")
        
        # 显示最后一个猜测（如果有）
        if env.last_guess is not None:
            print(f"\n🎯 最后一个猜测: {env.last_guess.mode} {env.last_guess.count} 个 {env.last_guess.face}")
            actual_count = env._count_dice(env.last_guess)
            print(f"实际数量: {actual_count}")
            if actual_count >= env.last_guess.count:
                print("✅ 猜测是正确的")
            else:
                print("❌ 猜测是错误的")
        
        # 显示获胜者
        human_penalty = env.penalties[f"player_{human_player_idx}"]
        min_penalty = min(env.penalties.values())
        winners = [agent for agent, penalty in env.penalties.items() if penalty == min_penalty]
        
        if f"player_{human_player_idx}" in winners:
            if len(winners) == 1:
                print("🎉 恭喜你赢得了这场比赛!")
            else:
                print("🤝 你与其他玩家并列第一!")
        else:
            winner_names = ["你" if w == f"player_{human_player_idx}" else w for w in winners]
            print(f"🤖 {'和'.join(winner_names)} 赢得了这场比赛!")
        
        # 询问是否继续（如果不是最后一场）
        if match < args.num_match - 1:
            try:
                continue_game = input("\n是否继续下一场比赛? (y/n): ").strip().lower()
                if continue_game not in ['y', 'yes', '是', '']:
                    break
            except (KeyboardInterrupt, EOFError):
                print("\n\n游戏已取消!")
                break
    
    print(f"\n{'='*80}")
    print("🎮 游戏结束，感谢游玩!")
    print(f"{'='*80}")