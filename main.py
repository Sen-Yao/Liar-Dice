import argparse
from env import LiarDiceEnv
from agents.basic_agent import BasicRuleAgent
from agents.heuristic_agent import HeuristicRuleAgent
from agents.llm_agent import LLMAgent
from human import run_human_vs_ai_mode

def main(args):
    """主游戏入口，支持人类对战模式和AI训练模式"""
    
    if args.mode == "human":
        run_human_vs_ai_mode(args)
    else:
        run_ai_training_mode(args)



def run_ai_training_mode(args):
    """运行AI训练模式"""
    
    print("🤖 AI训练模式启动")
    print("="*40)
    
    env = LiarDiceEnv(num_players=args.num_players, render_mode="human" if args.render else None)
    
    # 根据agent类型创建agents
    agents = {}
    for agent in env.possible_agents:
        if args.agent_type == "llm":
            agents[agent] = LLMAgent(agent, args.num_players)
        elif args.agent_type == "heuristic":
            agents[agent] = HeuristicRuleAgent(agent, args.num_players, confidence_threshold=1)
        elif args.agent_type == "basic":
            agents[agent] = BasicRuleAgent(agent, args.num_players)
        else:
            print(f"Unknow agent name:{args.agent_type}! Exiting...")
            exit()
    
    print(f"🤖 使用 {args.agent_type.upper()} Agent 类型")
    print(f"👥 玩家数量: {args.num_players}")
    print(f"🎯 比赛场数: {args.num_match}")
    
    for match in range(args.num_match):
        print(f"\n\n--- MATCH {match + 1} STARTING ---")
        env.reset()
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                # Agent is done for this round, step with a dummy action
                action = None
            else:
                agent_obj = agents[agent_id]
                action = agent_obj.get_action(observation)
                if hasattr(action, 'mode'):  # Guess对象
                    print(f"Agent {agent_id} acts: {action.count} 个 {action.face} {action.mode}")
                else:  # Challenge对象
                    print(f"Agent {agent_id} acts: Challenge()")
            
            env.step(action)
        
        print(f"--- MATCH {match + 1} ENDED ---")
        print(f"Final Penalties: {env.penalties}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="骰子骗子游戏 - 支持人机对战和AI训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        使用示例:
        # AI训练模式 (默认) - 使用规则Agent
        python main.py
        
        # AI训练模式 - 使用LLM Agent
        python main.py --agent_type llm
        
        # 人机对战模式
        python main.py --mode human
        
        # 人机对战，4个玩家，3场比赛
        python main.py --mode human --num_players 4 --num_match 3
        
        # 启用详细渲染
        python main.py --mode human --render
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["ai", "human"], 
        default="ai",
        help="游戏模式: 'ai'为AI训练模式，'human'为人机对战模式 (默认: ai)"
    )
    parser.add_argument(
        "--agent_type", 
        choices=["basic", "llm", "heuristic"], 
        default="basic",
        help="AI训练模式下的Agent类型: 'basic'为规则Agent，'llm'为LLM Agent (默认: basic)"
    )
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="启用详细渲染显示游戏过程"
    )
    parser.add_argument(
        "--num_players", 
        type=int, 
        default=2, 
        help="玩家数量 (默认: 2)"
    )
    parser.add_argument(
        "--num_match", 
        type=int, 
        default=1, 
        help="比赛场数 (默认: 1)"
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n游戏被用户中断!")
    except EOFError:
        print("\n\n输入结束，游戏退出!")